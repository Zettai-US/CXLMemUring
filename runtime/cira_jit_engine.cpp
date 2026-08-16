// CIRA JIT engine — ORC LLJIT-backed specializer. See cira_jit_engine.h.

#include "cira_jit_engine.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cstring>
#include <functional>

namespace cira {

namespace {

std::atomic<bool> g_targetInited{false};

// Walk every load of the sentinel global and rewrite it directly to the
// chosen constant. This is functionally equivalent to "make the global
// internal+constant then run -globalopt -instcombine", but it does the
// fold by hand and avoids LLVM's pass manager entirely — necessary on
// LLVM trunk builds whose PoisoningVH<BasicBlock> machinery is broken.
template <typename ConstantBuilderT>
void foldGlobalUses(llvm::Module &M, const char *name, ConstantBuilderT makeConstant) {
    llvm::GlobalVariable *g = M.getGlobalVariable(name, /*AllowInternal=*/true);
    if (!g)
        return;
    llvm::Constant *C = makeConstant(M.getContext());
    // Snapshot uses first — replacing them mutates the use-list.
    llvm::SmallVector<llvm::User *, 8> users(g->users().begin(), g->users().end());
    for (llvm::User *U : users) {
        if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(U)) {
            LI->replaceAllUsesWith(C);
            LI->eraseFromParent();
        }
    }
    if (g->use_empty())
        g->eraseFromParent();
}

void patchI32(llvm::Module &M, const char *name, uint32_t value) {
    foldGlobalUses(M, name, [value](llvm::LLVMContext &C) -> llvm::Constant * {
        return llvm::ConstantInt::get(llvm::Type::getInt32Ty(C), value);
    });
}

void patchF32(llvm::Module &M, const char *name, float value) {
    foldGlobalUses(M, name, [value](llvm::LLVMContext &C) -> llvm::Constant * {
        return llvm::ConstantFP::get(llvm::Type::getFloatTy(C), (double)value);
    });
}

uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

} // namespace

bool CiraJitEngine::initializeNativeTarget() {
    bool expected = false;
    if (!g_targetInited.compare_exchange_strong(expected, true))
        return true;
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    return true;
}

CiraJitEngine::CiraJitEngine() = default;
CiraJitEngine::~CiraJitEngine() = default;

uint64_t CiraJitEngine::fingerprint(const cira_jit_decision_t &d) {
    // Quantize host_device_split to 1/256 so trivially-different floats
    // collapse onto the same compiled code.
    uint64_t s = (uint64_t)(d.host_device_split * 256.0f) & 0xff;
    uint64_t v = ((uint64_t)d.batch_size & 0xffff) | (((uint64_t)d.traversal_depth & 0xffff) << 16) |
                 (((uint64_t)d.pipeline_distance & 0xffff) << 32) | ((s & 0xff) << 48) |
                 (((uint64_t)(d.should_offload ? 1u : 0u)) << 56);
    return mix64(v);
}

void CiraJitEngine::patchAndOptimize(llvm::Module &M, const cira_jit_decision_t &d) {
    // Rewrite every load of a sentinel global to the literal constant —
    // already enough for the LLJIT MC backend to constant-fold dependent
    // arithmetic and unroll trivially-bounded loops.
    patchI32(M, kSentinelBatchSize, d.batch_size);
    patchI32(M, kSentinelTraversalDepth, d.traversal_depth);
    patchI32(M, kSentinelPipelineDistance, d.pipeline_distance);
    patchF32(M, kSentinelHostDeviceSplit, d.host_device_split);

    // Run the standard O2 pipeline so InstCombine / SimplifyCFG / loop
    // unrolling can additionally fold the patched constants across basic
    // blocks before codegen.
    llvm::PassBuilder PB;
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    auto MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
    MPM.run(M, MAM);
}

bool CiraJitEngine::ensureLLJIT() {
    if (jit_)
        return true;
    initializeNativeTarget();
    auto J = llvm::orc::LLJITBuilder().create();
    if (!J) {
        llvm::errs() << "[cira-jit] LLJIT create failed: " << llvm::toString(J.takeError()) << "\n";
        return false;
    }
    jit_ = std::move(*J);
    // ctx_ is no longer cached at the engine level — modern ORC requires a
    // fresh LLVMContext per ThreadSafeModule. We construct one inside each
    // specialize() call and transfer ownership to LLJIT.
    return true;
}

CiraJitFn CiraJitEngine::specialize(const std::string &bitcodePath, const std::string &kernelName,
                                    const cira_jit_decision_t &decision) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!ensureLLJIT())
        return nullptr;

    CacheKey key{kernelName, bitcodePath, fingerprint(decision)};
    if (auto it = cache_.find(key); it != cache_.end())
        return it->second;

    auto buf = llvm::MemoryBuffer::getFile(bitcodePath);
    if (!buf) {
        llvm::errs() << "[cira-jit] read failed: " << bitcodePath << ": " << buf.getError().message() << "\n";
        return nullptr;
    }
    // Each specialization gets a fresh LLVMContext that we hand to ORC.
    auto ctx = std::make_unique<llvm::LLVMContext>();
    auto modOrErr = llvm::parseBitcodeFile((*buf)->getMemBufferRef(), *ctx);
    if (!modOrErr) {
        llvm::errs() << "[cira-jit] parseBitcodeFile failed: " << llvm::toString(modOrErr.takeError()) << "\n";
        return nullptr;
    }
    std::unique_ptr<llvm::Module> M = std::move(*modOrErr);
    patchAndOptimize(*M, decision);

    if (auto err = jit_->addIRModule(llvm::orc::ThreadSafeModule(std::move(M), std::move(ctx)))) {
        llvm::errs() << "[cira-jit] addIRModule failed: " << llvm::toString(std::move(err)) << "\n";
        return nullptr;
    }

    auto sym = jit_->lookup(kernelName);
    if (!sym) {
        llvm::errs() << "[cira-jit] lookup '" << kernelName << "' failed: " << llvm::toString(sym.takeError()) << "\n";
        return nullptr;
    }
    auto fn = reinterpret_cast<CiraJitFn>(sym->getValue());
    cache_.emplace(std::move(key), fn);
    return fn;
}

CiraJitFn CiraJitEngine::specializeFromIR(const std::string &irText, const std::string &kernelName,
                                          const cira_jit_decision_t &decision) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!ensureLLJIT())
        return nullptr;

    // Use a content hash as the cache "source" so identical IR text shares
    // a cache entry.
    uint64_t srcHash = std::hash<std::string>{}(irText);
    char srcKey[32];
    std::snprintf(srcKey, sizeof(srcKey), "ir@%016llx", (unsigned long long)srcHash);

    CacheKey key{kernelName, srcKey, fingerprint(decision)};
    if (auto it = cache_.find(key); it != cache_.end())
        return it->second;

    auto ctx = std::make_unique<llvm::LLVMContext>();
    llvm::SMDiagnostic err;
    auto buf = llvm::MemoryBuffer::getMemBuffer(irText);
    auto M = llvm::parseIR(buf->getMemBufferRef(), err, *ctx);
    if (!M) {
        llvm::errs() << "[cira-jit] parseIR failed: ";
        err.print("cira-jit", llvm::errs());
        return nullptr;
    }
    patchAndOptimize(*M, decision);

    if (auto e = jit_->addIRModule(llvm::orc::ThreadSafeModule(std::move(M), std::move(ctx)))) {
        llvm::errs() << "[cira-jit] addIRModule failed: " << llvm::toString(std::move(e)) << "\n";
        return nullptr;
    }
    auto sym = jit_->lookup(kernelName);
    if (!sym) {
        llvm::errs() << "[cira-jit] lookup '" << kernelName << "' failed: " << llvm::toString(sym.takeError()) << "\n";
        return nullptr;
    }
    auto fn = reinterpret_cast<CiraJitFn>(sym->getValue());
    cache_.emplace(std::move(key), fn);
    return fn;
}

void CiraJitEngine::resetCache() {
    std::lock_guard<std::mutex> lock(mu_);
    cache_.clear();
    jit_.reset();
}

} // namespace cira
