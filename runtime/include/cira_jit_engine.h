// CIRA JIT engine — ORC-based runtime specializer for offload regions.
//
// The CIRA paper compiles a region once into a *template*: an LLVM module in
// which every schedule-dependent quantity is an external sentinel global
// rather than a literal.  At run time the cost model (cira_jit.h) observes the
// region and picks concrete values; this engine folds those values into the
// template, re-optimizes, and hands the module to ORC.  The result is a region
// specialized to the schedule the profiler actually saw, without recompiling
// the program.
//
//   1. compiler   : CIRA -> MLIR -> LLVM IR, knobs left as sentinel globals
//   2. cost model : cira_jit_decide() picks batch/depth/distance/split
//   3. this engine: patch sentinels -> optimize -> ORC LLJIT -> function ptr
//   4. cache      : identical decisions (same fingerprint) share compiled code
//
// The knob set is *not* fixed.  ciraKnobsFromDecision() produces the four
// knobs from the paper, but a caller can specialize on any named global of a
// supported type (see CiraKnobType) — device queue depth, tile shape, an
// already-resolved device function address, a mapped MMIO window pointer, and
// so on.  This is what lets a JIT-specialized kernel write the CXL doorbell
// directly: the control-window base is folded in as a Ptr knob and the
// submission helpers are bound as callable symbols.
//
// Specialized kernels may call any symbol in the CIRA runtime ABI (see
// kRuntimeAbiSymbols in the .cpp): cira_offload_submit, cira_future_await,
// cira_mmio_submit_call / cira_mmio_write32 / cira_mmio_wait_completion,
// cira_wait_u32, vortex_cxl_submit_job_mmio, ... The engine binds them as
// absolute symbols and additionally exposes the host process, so an IR
// template can declare them and call them without any extra plumbing.

#ifndef CIRA_JIT_ENGINE_H
#define CIRA_JIT_ENGINE_H

#include "cira_jit.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
namespace orc {
class LLJIT;
} // namespace orc
} // namespace llvm

namespace cira {

// Sentinel globals for the four knobs in the paper. Templates emitted by the
// CIRA backend declare these; ciraKnobsFromDecision() fills them in.
constexpr const char *kSentinelBatchSize = "cira_kBatchSize";
constexpr const char *kSentinelTraversalDepth = "cira_kTraversalDepth";
constexpr const char *kSentinelPipelineDistance = "cira_kPipelineDistance";
constexpr const char *kSentinelHostDeviceSplit = "cira_kHostDeviceSplit";
constexpr const char *kSentinelShouldOffload = "cira_kShouldOffload";

// Sentinels for the device-side dispatch path. Folding these turns a generic
// "submit to whatever device is configured" template into a kernel that writes
// a known control window at a known offset.
constexpr const char *kSentinelMmioBase = "cira_kMmioBase";
constexpr const char *kSentinelMmioSize = "cira_kMmioSize";
constexpr const char *kSentinelDeviceFunc = "cira_kDeviceFunc";
constexpr const char *kSentinelWaitBackend = "cira_kWaitBackend";
constexpr const char *kSentinelCompletionMagic = "cira_kCompletionMagic";

enum class CiraKnobType : uint8_t { I1, I8, I16, I32, I64, F32, F64, Ptr };

// A single (global name -> constant) substitution applied to the template.
struct CiraJitKnob {
    std::string name;
    CiraKnobType type = CiraKnobType::I32;
    uint64_t ivalue = 0; // integer / pointer payload
    double fvalue = 0.0; // floating payload

    static CiraJitKnob boolean(std::string n, bool v);
    static CiraJitKnob i32(std::string n, uint32_t v);
    static CiraJitKnob i64(std::string n, uint64_t v);
    static CiraJitKnob f32(std::string n, float v);
    static CiraJitKnob f64(std::string n, double v);
    static CiraJitKnob ptr(std::string n, const void *v);
};

// The four paper knobs plus should_offload, ready to hand to specialize().
std::vector<CiraJitKnob> ciraKnobsFromDecision(const cira_jit_decision_t &d);

// Knobs describing the currently configured CXL Type-2 control window, device
// entry point, and host wait backend, so a template can be specialized into
// direct MMIO stores. Empty when no device is configured.
std::vector<CiraJitKnob> ciraKnobsFromDevice();

// What to build. Everything except `kernel` has a usable default.
struct CiraJitSpec {
    std::string kernel; // symbol to look up
    std::vector<CiraJitKnob> knobs;
    unsigned optLevel = 2; // 0..3
    bool bindRuntimeAbi = true; // expose cira_*/vortex_*
    bool exposeProcessSymbols = true;
    // Distinguishes cache entries built from the same source with different
    // non-knob settings (e.g. a different device generation).
    std::string variant;
};

// Generic untyped function pointer returned from a successful lookup.
using CiraJitFn = void (*)();

class CiraJitEngine {
public:
    // Initializes the LLVM native target. Idempotent, thread-safe.
    static bool initializeNativeTarget();

    // Process-wide engine. Most callers want this; the cache is shared.
    static CiraJitEngine &shared();

    CiraJitEngine();
    ~CiraJitEngine();

    CiraJitEngine(const CiraJitEngine &) = delete;
    CiraJitEngine &operator=(const CiraJitEngine &) = delete;

    // --- generic entry points -------------------------------------------

    // Specialize from an LLVM bitcode (.bc) or textual IR (.ll) file.
    CiraJitFn specializeFile(const std::string &path, const CiraJitSpec &spec);

    // Specialize from IR text held in memory.
    CiraJitFn specializeText(const std::string &irText, const CiraJitSpec &spec);

    // Specialize from an in-memory bitcode buffer.
    CiraJitFn specializeBitcode(const void *data, size_t size, const CiraJitSpec &spec);

    // Typed convenience wrapper.
    template <typename FnT> FnT specializeFileAs(const std::string &path, const CiraJitSpec &spec) {
        return reinterpret_cast<FnT>(specializeFile(path, spec));
    }

    // Bind an extra symbol that specialized kernels may call.
    bool defineSymbol(const std::string &name, void *address);

    // Look up a symbol already materialized in the JIT. Returns nullptr when
    // absent.
    CiraJitFn lookup(const std::string &name);

    // --- cost-model convenience -----------------------------------------
    // Equivalent to the generic entry points with
    // ciraKnobsFromDecision(decision) as the knob set.

    CiraJitFn specialize(const std::string &bitcodePath, const std::string &kernelName,
                         const cira_jit_decision_t &decision);

    CiraJitFn specializeFromIR(const std::string &irText, const std::string &kernelName,
                               const cira_jit_decision_t &decision);

    // Forget all cached compiled code and drop the LLJIT instance.
    void resetCache();

    // Number of distinct specializations currently cached.
    size_t cacheSize() const;

private:
    struct CacheKey {
        std::string kernel;
        std::string source; // file path or content hash
        std::string variant;
        uint64_t fingerprint; // knobs + optLevel
        bool operator==(const CacheKey &o) const noexcept {
            return fingerprint == o.fingerprint && kernel == o.kernel && source == o.source && variant == o.variant;
        }
    };
    struct CacheKeyHash {
        size_t operator()(const CacheKey &k) const noexcept {
            std::hash<std::string> h;
            return (size_t)k.fingerprint ^ (h(k.kernel) << 1) ^ (h(k.source) << 2) ^ (h(k.variant) << 3);
        }
    };

    // Rewrite loads of each knob's sentinel global to a literal, then run the
    // requested optimization pipeline so the constants propagate.
    static void patchAndOptimize(llvm::Module &module, const CiraJitSpec &spec);

    // Stable hash of the knob set + optLevel. Equal fingerprints share code.
    static uint64_t fingerprint(const CiraJitSpec &spec);

    bool ensureLLJIT();
    bool bindRuntimeAbiLocked();
    CiraJitFn finish(std::unique_ptr<llvm::Module> module, std::unique_ptr<llvm::LLVMContext> context,
                     const CiraJitSpec &spec, CacheKey key);

    std::unique_ptr<llvm::orc::LLJIT> jit_;
    std::unordered_map<CacheKey, CiraJitFn, CacheKeyHash> cache_;
    bool abiBound_ = false;
    mutable std::mutex mu_;
};

} // namespace cira

#endif // CIRA_JIT_ENGINE_H
