// CIRA JIT engine — ORC LLJIT-backed specializer. See cira_jit_engine.h.

#include "cira_jit_engine.h"

#include "cira_cxl_job.h"
#include "cira_mmio.h"
#include "cira_mwait.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
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
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <functional>

namespace cira {

namespace {

std::atomic<bool> g_targetInited{false};

// ---------------------------------------------------------------------------
// Runtime ABI exposed to specialized kernels
// ---------------------------------------------------------------------------
//
// Every operation an offload region needs at run time. Addresses are resolved
// with dlsym(RTLD_DEFAULT) so this list stays a pure declaration: a symbol the
// process does not export is simply skipped, and a template that does not call
// it still compiles.
const char *const kRuntimeAbiSymbols[] = {
    // cira.offload / cira.barrier / futures
    "cira_offload_submit",
    "cira_phase_barrier",
    "cira_future_alloc",
    "cira_future_free",
    "cira_future_await",
    "cira_future_pool_alloc",
    "cira_future_pool_free",
    "cira_future_pool_get",
    "cira_future_pool_arm",
    "cira_future_pool_get_device_addr",
    // cira.install_cacheline / cira.evict_hint / LLC tiles
    "cira_install_cacheline_x86",
    "cira_evict_hint_x86",
    "cira_llc_tile_alloc",
    "cira_llc_tile_free",
    "cira_llc_tile_future",
    "cira_llc_tile_install_from_cxl",
    "cira_llc_tile_get_mwait",
    // CXL Type-2 MMIO doorbell path
    "cira_mmio_default",
    "cira_mmio_base",
    "cira_mmio_size",
    "cira_mmio_read32",
    "cira_mmio_read64",
    "cira_mmio_write32",
    "cira_mmio_write64",
    "cira_mmio_write_block",
    "cira_mmio_fence",
    "cira_mmio_submit_job",
    "cira_mmio_submit_call",
    "cira_mmio_wait_seq",
    "cira_mmio_wait_completion",
    "cira_mmio_arm_completion",
    "cira_mmio_signal_completion",
    "cira_mmio_device_func",
    // cache-resident wait (MONITOR/UMWAIT/TPAUSE/WFE)
    "cira_wait_u32",
    "cira_wait_u32_ne",
    "cira_wait_relax",
    "cira_wait_backend",
    // device firmware entry points
    "vortex_cxl_submit_job_mmio",
    "vortex_cxl_submit_call_mmio",
    "__vortex_install_cacheline",
    "__vortex_prefetch_chain",
    "__vortex_prefetch_chain_kernel",
};

// ---------------------------------------------------------------------------
// Constant folding of sentinel globals
// ---------------------------------------------------------------------------

llvm::Type *typeFor(llvm::LLVMContext &C, CiraKnobType t) {
    switch (t) {
    case CiraKnobType::I1:
        return llvm::Type::getInt1Ty(C);
    case CiraKnobType::I8:
        return llvm::Type::getInt8Ty(C);
    case CiraKnobType::I16:
        return llvm::Type::getInt16Ty(C);
    case CiraKnobType::I32:
        return llvm::Type::getInt32Ty(C);
    case CiraKnobType::I64:
        return llvm::Type::getInt64Ty(C);
    case CiraKnobType::F32:
        return llvm::Type::getFloatTy(C);
    case CiraKnobType::F64:
        return llvm::Type::getDoubleTy(C);
    case CiraKnobType::Ptr:
        return llvm::PointerType::getUnqual(C);
    }
    return llvm::Type::getInt32Ty(C);
}

// Build the replacement constant, coerced to the type the template actually
// loads. A template that declares a knob as i64 while the caller supplies an
// i32 still specializes correctly.
llvm::Constant *constantFor(llvm::LLVMContext &C, const CiraJitKnob &knob, llvm::Type *loadTy) {
    llvm::Type *ty = loadTy ? loadTy : typeFor(C, knob.type);

    if (ty->isPointerTy()) {
        llvm::Type *i64 = llvm::Type::getInt64Ty(C);
        return llvm::ConstantExpr::getIntToPtr(llvm::ConstantInt::get(i64, knob.ivalue), ty);
    }
    if (ty->isFloatingPointTy()) {
        double v =
            (knob.type == CiraKnobType::F32 || knob.type == CiraKnobType::F64) ? knob.fvalue : (double)knob.ivalue;
        return llvm::ConstantFP::get(ty, v);
    }
    if (ty->isIntegerTy()) {
        uint64_t v =
            (knob.type == CiraKnobType::F32 || knob.type == CiraKnobType::F64) ? (uint64_t)knob.fvalue : knob.ivalue;
        return llvm::ConstantInt::get(ty, v, /*IsSigned=*/false);
    }
    return nullptr;
}

// Rewrite every load of the sentinel global directly to the chosen constant.
// This is what "make the global internal+constant, then run -globalopt
// -instcombine" would achieve, done by hand so it works even at -O0 and on
// LLVM builds whose global-opt machinery is unavailable.
bool foldKnob(llvm::Module &M, const CiraJitKnob &knob) {
    llvm::GlobalVariable *g = M.getGlobalVariable(knob.name, /*AllowInternal=*/true);
    if (!g)
        return false;

    llvm::LLVMContext &C = M.getContext();
    bool folded = false;

    // Snapshot the use list first — replacing uses mutates it.
    llvm::SmallVector<llvm::User *, 8> users(g->users().begin(), g->users().end());
    for (llvm::User *U : users) {
        auto *LI = llvm::dyn_cast<llvm::LoadInst>(U);
        if (!LI)
            continue;
        llvm::Constant *CV = constantFor(C, knob, LI->getType());
        if (!CV)
            continue;
        LI->replaceAllUsesWith(CV);
        LI->eraseFromParent();
        folded = true;
    }

    // Also give the global itself a constant initializer so any remaining
    // aggregate/atomic access still sees the chosen value.
    if (llvm::Constant *init = constantFor(C, knob, g->getValueType())) {
        g->setInitializer(init);
        g->setConstant(true);
        g->setLinkage(llvm::GlobalValue::PrivateLinkage);
        folded = true;
    }
    if (g->use_empty())
        g->eraseFromParent();
    return folded;
}

llvm::OptimizationLevel optLevelFor(unsigned level) {
    switch (level) {
    case 0:
        return llvm::OptimizationLevel::O0;
    case 1:
        return llvm::OptimizationLevel::O1;
    case 3:
        return llvm::OptimizationLevel::O3;
    default:
        return llvm::OptimizationLevel::O2;
    }
}

uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

uint64_t hashString(const std::string &s) {
    uint64_t h = 1469598103934665603ULL; // FNV-1a
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

std::string contentKey(const void *data, size_t size) {
    uint64_t h = 1469598103934665603ULL;
    auto *bytes = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ULL;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "mem@%016llx", (unsigned long long)h);
    return buf;
}

} // namespace

// ---------------------------------------------------------------------------
// CiraJitKnob
// ---------------------------------------------------------------------------

CiraJitKnob CiraJitKnob::boolean(std::string n, bool v) {
    CiraJitKnob k;
    k.name = std::move(n);
    k.type = CiraKnobType::I1;
    k.ivalue = v ? 1u : 0u;
    return k;
}
CiraJitKnob CiraJitKnob::i32(std::string n, uint32_t v) {
    CiraJitKnob k;
    k.name = std::move(n);
    k.type = CiraKnobType::I32;
    k.ivalue = v;
    return k;
}
CiraJitKnob CiraJitKnob::i64(std::string n, uint64_t v) {
    CiraJitKnob k;
    k.name = std::move(n);
    k.type = CiraKnobType::I64;
    k.ivalue = v;
    return k;
}
CiraJitKnob CiraJitKnob::f32(std::string n, float v) {
    CiraJitKnob k;
    k.name = std::move(n);
    k.type = CiraKnobType::F32;
    k.fvalue = (double)v;
    return k;
}
CiraJitKnob CiraJitKnob::f64(std::string n, double v) {
    CiraJitKnob k;
    k.name = std::move(n);
    k.type = CiraKnobType::F64;
    k.fvalue = v;
    return k;
}
CiraJitKnob CiraJitKnob::ptr(std::string n, const void *v) {
    CiraJitKnob k;
    k.name = std::move(n);
    k.type = CiraKnobType::Ptr;
    k.ivalue = (uint64_t)(uintptr_t)v;
    return k;
}

std::vector<CiraJitKnob> ciraKnobsFromDecision(const cira_jit_decision_t &d) {
    return {
        CiraJitKnob::i32(kSentinelBatchSize, d.batch_size),
        CiraJitKnob::i32(kSentinelTraversalDepth, d.traversal_depth),
        CiraJitKnob::i32(kSentinelPipelineDistance, d.pipeline_distance),
        CiraJitKnob::f32(kSentinelHostDeviceSplit, d.host_device_split),
        CiraJitKnob::boolean(kSentinelShouldOffload, d.should_offload),
    };
}

std::vector<CiraJitKnob> ciraKnobsFromDevice() {
    cira_mmio_window_t *w = cira_mmio_default();
    if (!w)
        return {};
    return {
        CiraJitKnob::ptr(kSentinelMmioBase, cira_mmio_base(w)),
        CiraJitKnob::i64(kSentinelMmioSize, cira_mmio_size(w)),
        CiraJitKnob::ptr(kSentinelDeviceFunc, cira_mmio_device_func()),
        CiraJitKnob::i32(kSentinelWaitBackend, (uint32_t)cira_wait_backend()),
        CiraJitKnob::i32(kSentinelCompletionMagic, CIRA_CXL_COMPLETION_MAGIC),
    };
}

// ---------------------------------------------------------------------------
// CiraJitEngine
// ---------------------------------------------------------------------------

bool CiraJitEngine::initializeNativeTarget() {
    bool expected = false;
    if (!g_targetInited.compare_exchange_strong(expected, true))
        return true;
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    return true;
}

CiraJitEngine &CiraJitEngine::shared() {
    static CiraJitEngine instance;
    return instance;
}

CiraJitEngine::CiraJitEngine() = default;
CiraJitEngine::~CiraJitEngine() = default;

uint64_t CiraJitEngine::fingerprint(const CiraJitSpec &spec) {
    uint64_t h = mix64(spec.optLevel + 1);
    h ^= mix64(hashString(spec.variant) + 0x9e3779b97f4a7c15ULL);
    h ^= mix64(spec.bindRuntimeAbi ? 0x11 : 0x22);
    h ^= mix64(spec.exposeProcessSymbols ? 0x33 : 0x44);

    // Order-independent so callers may build the knob vector however they like.
    for (const CiraJitKnob &k : spec.knobs) {
        uint64_t payload;
        if (k.type == CiraKnobType::F32) {
            // Quantize to 1/256 so trivially-different floats share code.
            payload = (uint64_t)(int64_t)(k.fvalue * 256.0);
        } else if (k.type == CiraKnobType::F64) {
            std::memcpy(&payload, &k.fvalue, sizeof(payload));
        } else {
            payload = k.ivalue;
        }
        h ^= mix64(hashString(k.name) ^ mix64(payload) ^ (uint64_t)k.type);
    }
    return h;
}

void CiraJitEngine::patchAndOptimize(llvm::Module &M, const CiraJitSpec &spec) {
    for (const CiraJitKnob &knob : spec.knobs)
        foldKnob(M, knob);

    // Re-run the optimizer so InstCombine / SimplifyCFG / loop unrolling can
    // propagate the folded constants across basic blocks: this is where the
    // batch loop gets its trip count and the host/device split branch is
    // resolved statically.
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

    llvm::OptimizationLevel level = optLevelFor(spec.optLevel);
    llvm::ModulePassManager MPM = level == llvm::OptimizationLevel::O0 ? PB.buildO0DefaultPipeline(level)
                                                                       : PB.buildPerModuleDefaultPipeline(level);
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
    return true;
}

bool CiraJitEngine::bindRuntimeAbiLocked() {
    if (abiBound_ || !jit_)
        return true;

    auto &jd = jit_->getMainJITDylib();
    llvm::orc::SymbolMap symbols;
    for (const char *name : kRuntimeAbiSymbols) {
        void *addr = dlsym(RTLD_DEFAULT, name);
        if (!addr)
            continue; // not linked into this process — fine
        symbols[jit_->mangleAndIntern(name)] = llvm::orc::ExecutorSymbolDef(
            llvm::orc::ExecutorAddr::fromPtr(addr), llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable);
    }
    if (!symbols.empty()) {
        if (auto err = jd.define(llvm::orc::absoluteSymbols(std::move(symbols)))) {
            llvm::errs() << "[cira-jit] runtime ABI bind failed: " << llvm::toString(std::move(err)) << "\n";
            return false;
        }
    }

    // Anything else the kernel references (libm, libc, the host binary) is
    // resolved straight out of the running process.
    auto gen = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(jit_->getDataLayout().getGlobalPrefix());
    if (gen) {
        jd.addGenerator(std::move(*gen));
    } else {
        llvm::consumeError(gen.takeError());
    }

    abiBound_ = true;
    return true;
}

bool CiraJitEngine::defineSymbol(const std::string &name, void *address) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!ensureLLJIT() || !address)
        return false;

    llvm::orc::SymbolMap symbols;
    symbols[jit_->mangleAndIntern(name)] = llvm::orc::ExecutorSymbolDef(
        llvm::orc::ExecutorAddr::fromPtr(address), llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable);
    if (auto err = jit_->getMainJITDylib().define(llvm::orc::absoluteSymbols(std::move(symbols)))) {
        llvm::errs() << "[cira-jit] defineSymbol('" << name << "') failed: " << llvm::toString(std::move(err)) << "\n";
        return false;
    }
    return true;
}

CiraJitFn CiraJitEngine::lookup(const std::string &name) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!jit_)
        return nullptr;
    auto sym = jit_->lookup(name);
    if (!sym) {
        llvm::consumeError(sym.takeError());
        return nullptr;
    }
    return reinterpret_cast<CiraJitFn>(sym->getValue());
}

CiraJitFn CiraJitEngine::finish(std::unique_ptr<llvm::Module> M, std::unique_ptr<llvm::LLVMContext> ctx,
                                const CiraJitSpec &spec, CacheKey key) {
    patchAndOptimize(*M, spec);

    if (spec.bindRuntimeAbi || spec.exposeProcessSymbols) {
        if (!bindRuntimeAbiLocked())
            return nullptr;
    }

    // Each specialization lands in its own JITDylib. Two schedules of the same
    // region define the same symbol name, so without this the second one would
    // be rejected as a duplicate definition.
    static std::atomic<uint64_t> dylibCounter{0};
    char dylibName[128];
    std::snprintf(dylibName, sizeof(dylibName), "cira.%s.%016llx.%llu", spec.kernel.c_str(),
                  (unsigned long long)key.fingerprint, (unsigned long long)dylibCounter.fetch_add(1));

    auto jd = jit_->createJITDylib(dylibName);
    if (!jd) {
        llvm::errs() << "[cira-jit] createJITDylib failed: " << llvm::toString(jd.takeError()) << "\n";
        return nullptr;
    }
    // Runtime ABI and process symbols live in the main dylib.
    jd->addToLinkOrder(jit_->getMainJITDylib());

    if (auto err = jit_->addIRModule(*jd, llvm::orc::ThreadSafeModule(std::move(M), std::move(ctx)))) {
        llvm::errs() << "[cira-jit] addIRModule failed: " << llvm::toString(std::move(err)) << "\n";
        return nullptr;
    }

    auto sym = jit_->lookup(*jd, spec.kernel);
    if (!sym) {
        llvm::errs() << "[cira-jit] lookup '" << spec.kernel << "' failed: " << llvm::toString(sym.takeError()) << "\n";
        return nullptr;
    }

    auto fn = reinterpret_cast<CiraJitFn>(sym->getValue());
    cache_.emplace(std::move(key), fn);
    return fn;
}

CiraJitFn CiraJitEngine::specializeFile(const std::string &path, const CiraJitSpec &spec) {
    std::lock_guard<std::mutex> lock(mu_);
    if (spec.kernel.empty() || !ensureLLJIT())
        return nullptr;

    CacheKey key{spec.kernel, path, spec.variant, fingerprint(spec)};
    if (auto it = cache_.find(key); it != cache_.end())
        return it->second;

    auto buf = llvm::MemoryBuffer::getFile(path);
    if (!buf) {
        llvm::errs() << "[cira-jit] read failed: " << path << ": " << buf.getError().message() << "\n";
        return nullptr;
    }

    auto ctx = std::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::Module> M;

    // Accept both bitcode and textual IR so templates can be shipped either way.
    if (llvm::isBitcode(reinterpret_cast<const unsigned char *>((*buf)->getBufferStart()),
                        reinterpret_cast<const unsigned char *>((*buf)->getBufferEnd()))) {
        auto modOrErr = llvm::parseBitcodeFile((*buf)->getMemBufferRef(), *ctx);
        if (!modOrErr) {
            llvm::errs() << "[cira-jit] parseBitcodeFile failed: " << llvm::toString(modOrErr.takeError()) << "\n";
            return nullptr;
        }
        M = std::move(*modOrErr);
    } else {
        llvm::SMDiagnostic diag;
        M = llvm::parseIR((*buf)->getMemBufferRef(), diag, *ctx);
        if (!M) {
            diag.print("cira-jit", llvm::errs());
            return nullptr;
        }
    }

    return finish(std::move(M), std::move(ctx), spec, std::move(key));
}

CiraJitFn CiraJitEngine::specializeText(const std::string &irText, const CiraJitSpec &spec) {
    std::lock_guard<std::mutex> lock(mu_);
    if (spec.kernel.empty() || !ensureLLJIT())
        return nullptr;

    CacheKey key{spec.kernel, contentKey(irText.data(), irText.size()), spec.variant, fingerprint(spec)};
    if (auto it = cache_.find(key); it != cache_.end())
        return it->second;

    auto ctx = std::make_unique<llvm::LLVMContext>();
    llvm::SMDiagnostic diag;
    auto buf = llvm::MemoryBuffer::getMemBuffer(irText);
    auto M = llvm::parseIR(buf->getMemBufferRef(), diag, *ctx);
    if (!M) {
        diag.print("cira-jit", llvm::errs());
        return nullptr;
    }
    return finish(std::move(M), std::move(ctx), spec, std::move(key));
}

CiraJitFn CiraJitEngine::specializeBitcode(const void *data, size_t size, const CiraJitSpec &spec) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!data || !size || spec.kernel.empty() || !ensureLLJIT())
        return nullptr;

    CacheKey key{spec.kernel, contentKey(data, size), spec.variant, fingerprint(spec)};
    if (auto it = cache_.find(key); it != cache_.end())
        return it->second;

    auto ctx = std::make_unique<llvm::LLVMContext>();
    llvm::StringRef ref(static_cast<const char *>(data), size);
    auto modOrErr = llvm::parseBitcodeFile(llvm::MemoryBufferRef(ref, "cira-jit-buffer"), *ctx);
    if (!modOrErr) {
        llvm::errs() << "[cira-jit] parseBitcodeFile failed: " << llvm::toString(modOrErr.takeError()) << "\n";
        return nullptr;
    }
    return finish(std::move(*modOrErr), std::move(ctx), spec, std::move(key));
}

CiraJitFn CiraJitEngine::specialize(const std::string &bitcodePath, const std::string &kernelName,
                                    const cira_jit_decision_t &decision) {
    CiraJitSpec spec;
    spec.kernel = kernelName;
    spec.knobs = ciraKnobsFromDecision(decision);
    return specializeFile(bitcodePath, spec);
}

CiraJitFn CiraJitEngine::specializeFromIR(const std::string &irText, const std::string &kernelName,
                                          const cira_jit_decision_t &decision) {
    CiraJitSpec spec;
    spec.kernel = kernelName;
    spec.knobs = ciraKnobsFromDecision(decision);
    return specializeText(irText, spec);
}

void CiraJitEngine::resetCache() {
    std::lock_guard<std::mutex> lock(mu_);
    cache_.clear();
    jit_.reset();
    abiBound_ = false;
}

size_t CiraJitEngine::cacheSize() const {
    std::lock_guard<std::mutex> lock(mu_);
    return cache_.size();
}

} // namespace cira
