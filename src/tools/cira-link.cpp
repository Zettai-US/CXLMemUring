//===- cira-link.cpp - Merge several CIR/MLIR modules into one ------------===//
//
// Parses a set of per-translation-unit MLIR modules, links them into a single
// module (resolving symbol collisions the way a linker would), and reports the
// cross-module symbol relationships.
//
//===----------------------------------------------------------------------===//

#include "Dialect/RemoteMemDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;

static llvm::cl::list<std::string> inputFilenames(llvm::cl::Positional, llvm::cl::OneOrMore,
                                                  llvm::cl::desc("<input mlir files>"));
static llvm::cl::opt<std::string> outputFilename("o", llvm::cl::desc("Merged module output"),
                                                 llvm::cl::value_desc("filename"), llvm::cl::init("-"));
static llvm::cl::opt<std::string> reportFilename("report", llvm::cl::desc("Symbol relationship report"),
                                                 llvm::cl::value_desc("filename"), llvm::cl::init(""));
static llvm::cl::opt<bool> noMerge("analyze-only", llvm::cl::desc("Only emit the relationship report"),
                                   llvm::cl::init(false));

namespace {

/// What a module contributes to, and expects from, the link.
struct ModuleFacts {
    std::string name;
    llvm::StringSet<> defined; // symbols with a body / initializer
    llvm::StringSet<> declared; // symbols referenced but left external
    llvm::StringSet<> referenced; // symbols actually used by some op
    unsigned numOps = 0;
    unsigned renamed = 0;
    unsigned droppedDuplicates = 0;
};

bool isDefinition(Operation *op) {
    if (auto func = dyn_cast<FunctionOpInterface>(op))
        return !func.isExternal();
    // Globals without an initial value are tentative declarations.
    return op->getNumRegions() > 0 ? !op->getRegion(0).empty() : op->hasAttr("initial_value");
}

void collectFacts(ModuleOp module, ModuleFacts &facts) {
    for (Operation &op : module.getBody()->getOperations()) {
        ++facts.numOps;
        auto sym = dyn_cast<SymbolOpInterface>(&op);
        if (!sym)
            continue;
        (isDefinition(&op) ? facts.defined : facts.declared).insert(sym.getName());
    }
    // SymbolTable::getSymbolUses bails out on these modules, so read the symbol
    // references straight off the attributes.
    module.walk([&](Operation *op) {
        for (NamedAttribute named : op->getAttrs())
            named.getValue().walk(
                [&](SymbolRefAttr ref) { facts.referenced.insert(ref.getRootReference().getValue()); });
    });
}

} // namespace

int main(int argc, char **argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv, "CIRA module linker\n");

    DialectRegistry registry;
    registerAllDialects(registry);
    registry.insert<::cir::CIRDialect, mlir::cira::RemoteMemDialect>();

    MLIRContext context(registry);
    context.allowUnregisteredDialects();
    context.loadAllAvailableDialects();

    OpBuilder builder(&context);
    auto merged = ModuleOp::create(builder.getUnknownLoc(), "cira_linked");
    SymbolTable mergedSymbols(merged);

    SmallVector<ModuleFacts> allFacts;
    // symbol -> module that defines it, for the cross-module edges.
    llvm::StringMap<std::string> definedBy;
    llvm::StringMap<SmallVector<std::string>> duplicateDefinitions;

    for (const std::string &path : inputFilenames) {
        std::string stem = llvm::sys::path::stem(path).str();
        llvm::errs() << "[cira-link] parsing " << stem << "\n";

        OwningOpRef<ModuleOp> source = parseSourceFile<ModuleOp>(path, &context);
        if (!source) {
            llvm::errs() << "[cira-link] SKIPPING unparsable input " << path << "\n";
            continue;
        }

        ModuleFacts facts;
        facts.name = stem;
        collectFacts(*source, facts);

        for (const auto &entry : facts.defined) {
            StringRef name = entry.getKey();
            auto it = definedBy.find(name);
            if (it == definedBy.end())
                definedBy[name] = stem;
            else
                duplicateDefinitions[name].push_back(stem);
        }

        if (noMerge) {
            allFacts.push_back(std::move(facts));
            continue;
        }

        // Resolve collisions *before* moving anything, so that intra-module
        // references are rewritten while the referencing ops are still here.
        SmallVector<Operation *> toDrop;
        for (Operation &op : source->getBody()->getOperations()) {
            auto sym = dyn_cast<SymbolOpInterface>(&op);
            if (!sym)
                continue;
            Operation *existing = mergedSymbols.lookup(sym.getNameAttr());
            if (!existing)
                continue;

            bool incomingIsDef = isDefinition(&op);
            bool existingIsDef = isDefinition(existing);

            if (!incomingIsDef) {
                toDrop.push_back(&op); // already have something under this name
                continue;
            }
            if (!existingIsDef) {
                mergedSymbols.erase(existing); // definition supersedes declaration
                continue;
            }
            if (sym.isPrivate()) {
                // TU-local symbol: give it a unique name and fix local uses.
                std::string unique = (sym.getName() + "." + stem).str();
                auto newName = builder.getStringAttr(unique);
                if (succeeded(SymbolTable::replaceAllSymbolUses(&op, newName, source->getOperation()))) {
                    SymbolTable::setSymbolName(&op, newName);
                    ++facts.renamed;
                    continue;
                }
            }
            // Two external definitions of the same name: keep the first.
            toDrop.push_back(&op);
            ++facts.droppedDuplicates;
        }
        for (Operation *op : toDrop)
            op->erase();

        for (Operation &op : llvm::make_early_inc_range(source->getBody()->getOperations())) {
            op.remove();
            merged.getBody()->push_back(&op);
            if (auto sym = dyn_cast<SymbolOpInterface>(&op))
                mergedSymbols.insert(&op, merged.getBody()->end());
        }

        allFacts.push_back(std::move(facts));
    }

    // ---- relationship report -------------------------------------------------
    std::string report;
    llvm::raw_string_ostream os(report);

    os << "== per-module ==\n";
    os << llvm::formatv("{0,-28} {1,+8} {2,+8} {3,+8} {4,+8} {5,+8}\n", "module", "topops", "defs", "decls", "refs",
                        "renamed");
    for (const ModuleFacts &f : allFacts)
        os << llvm::formatv("{0,-28} {1,+8} {2,+8} {3,+8} {4,+8} {5,+8}\n", f.name, f.numOps, f.defined.size(),
                            f.declared.size(), f.referenced.size(), f.renamed);

    os << "\n== cross-module dependencies (uses -> provider) ==\n";
    for (const ModuleFacts &f : allFacts) {
        llvm::StringMap<unsigned> edges;
        unsigned unresolved = 0;
        for (const auto &entry : f.referenced) {
            StringRef name = entry.getKey();
            if (f.defined.contains(name))
                continue;
            auto it = definedBy.find(name);
            if (it == definedBy.end()) {
                ++unresolved;
                continue;
            }
            if (it->second != f.name)
                ++edges[it->second];
        }
        os << f.name << ":\n";
        if (edges.empty())
            os << "    (no resolved cross-module symbol uses)\n";
        for (const auto &e : edges)
            os << llvm::formatv("    -> {0,-26} {1} symbols\n", e.getKey(), e.getValue());
        os << llvm::formatv("    unresolved (external/libc): {0}\n", unresolved);
    }

    os << "\n== duplicate definitions ==\n";
    if (duplicateDefinitions.empty())
        os << "    none\n";
    SmallVector<StringRef> dupNames;
    for (const auto &d : duplicateDefinitions)
        dupNames.push_back(d.getKey());
    llvm::sort(dupNames);
    os << llvm::formatv("    {0} symbols defined in more than one module\n", dupNames.size());
    for (StringRef name : llvm::ArrayRef<StringRef>(dupNames).take_front(40)) {
        os << "    " << name << ": " << definedBy[name];
        for (const std::string &other : duplicateDefinitions[name])
            os << ", " << other;
        os << "\n";
    }
    if (dupNames.size() > 40)
        os << llvm::formatv("    ... and {0} more\n", dupNames.size() - 40);

    if (reportFilename.empty()) {
        llvm::errs() << os.str();
    } else {
        std::string error;
        auto file = openOutputFile(reportFilename, &error);
        if (!file) {
            llvm::errs() << error << "\n";
            return 1;
        }
        file->os() << os.str();
        file->keep();
    }

    if (noMerge)
        return 0;

    std::string error;
    auto output = openOutputFile(outputFilename, &error);
    if (!output) {
        llvm::errs() << error << "\n";
        return 1;
    }
    // Locations are what keeps regions attributable to a translation unit after
    // linking, so keep them.
    merged.print(output->os(), OpPrintingFlags().enableDebugInfo(true));
    output->os() << "\n";
    output->keep();
    return 0;
}
