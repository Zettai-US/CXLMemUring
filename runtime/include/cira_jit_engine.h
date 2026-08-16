// CIRA JIT engine — ORC-based runtime specializer for offload kernels.
//
// Workflow:
//   1. The compiler (CIRA → MLIR → LLVM IR) emits an offload kernel that
//      reads four sentinel external globals declared with the well-known
//      names below. Until specialization those globals are zero-initialized
//      placeholders — code that uses them is intentionally unoptimised.
//   2. cira_jit_decide() (cira_jit.h) picks concrete knob values from the
//      profiler's view of the workload.
//   3. CiraJitEngine::specialize() patches the placeholders with the chosen
//      values, runs O3 over the patched module so loop bounds, prefetch
//      depths, and split branches fold into constants, then hands the module
//      to LLVM ORC LLJIT. The lookup returns a typed function pointer.
//   4. Subsequent calls with an equivalent decision (same fingerprint) reuse
//      the cached compiled code; the engine never recompiles unnecessarily.
//
// Sentinel global names the IR template MUST declare (any unused ones can
// be dead-stripped, but their presence makes specialization deterministic):
//
//   @cira_kBatchSize         : i32  (= 0)
//   @cira_kTraversalDepth    : i32  (= 0)
//   @cira_kPipelineDistance  : i32  (= 0)
//   @cira_kHostDeviceSplit   : float (= 0.0)
//
// All are external linkage with `internal` visibility after specialization.

#ifndef CIRA_JIT_ENGINE_H
#define CIRA_JIT_ENGINE_H

#include "cira_jit.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace llvm {
class LLVMContext;
class Module;
class MemoryBuffer;
namespace orc {
class LLJIT;
} // namespace orc
} // namespace llvm

namespace cira {

// Well-known global names patched by the engine. Match the header docstring.
constexpr const char *kSentinelBatchSize = "cira_kBatchSize";
constexpr const char *kSentinelTraversalDepth = "cira_kTraversalDepth";
constexpr const char *kSentinelPipelineDistance = "cira_kPipelineDistance";
constexpr const char *kSentinelHostDeviceSplit = "cira_kHostDeviceSplit";

// Generic untyped function pointer returned from a successful lookup.
using CiraJitFn = void (*)();

class CiraJitEngine {
public:
    // Initializes the LLVM native target + ORC LLJIT. Must be called once
    // per process before any specialize() call. Subsequent calls are no-ops.
    static bool initializeNativeTarget();

    CiraJitEngine();
    ~CiraJitEngine();

    CiraJitEngine(const CiraJitEngine &) = delete;
    CiraJitEngine &operator=(const CiraJitEngine &) = delete;

    // Specialize `kernelName` inside the IR module read from `bitcodePath`
    // (LLVM bitcode .bc) using the supplied decision, then return a function
    // pointer to the specialized symbol. Returns nullptr on failure;
    // diagnostics go to stderr.
    //
    // The same (bitcodePath, kernelName, fingerprint(decision)) tuple
    // resolves to the same compiled code on subsequent calls.
    CiraJitFn specialize(const std::string &bitcodePath, const std::string &kernelName,
                         const cira_jit_decision_t &decision);

    // Same as above but takes raw IR text (useful for unit tests).
    CiraJitFn specializeFromIR(const std::string &irText, const std::string &kernelName,
                               const cira_jit_decision_t &decision);

    // Forget all cached compiled code. The underlying LLJIT instance is
    // recreated lazily on the next specialize() call.
    void resetCache();

private:
    // Hash-key for the cache: kernel + IR identity + knob fingerprint.
    struct CacheKey {
        std::string kernel;
        std::string source; // bitcode path or IR-text hash
        uint64_t fingerprint;
        bool operator==(const CacheKey &o) const noexcept {
            return fingerprint == o.fingerprint && kernel == o.kernel && source == o.source;
        }
    };
    struct CacheKeyHash {
        size_t operator()(const CacheKey &k) const noexcept {
            // Mix kernel/source into the fingerprint — fingerprint already
            // encodes the knobs, so the kernel/source parts are tiebreakers.
            std::hash<std::string> h;
            return (size_t)k.fingerprint ^ (h(k.kernel) << 1) ^ (h(k.source) << 2);
        }
    };

    // Patches the sentinel globals in `module` with the decision's knob
    // values, then runs O3 to fold them into surrounding code.
    static void patchAndOptimize(llvm::Module &module, const cira_jit_decision_t &decision);

    // Stable hash of (batch_size, traversal_depth, pipeline_distance,
    // host_device_split, should_offload). Different decisions ⇒ different
    // compiled code; identical decisions ⇒ shared compiled code.
    static uint64_t fingerprint(const cira_jit_decision_t &d);

    // Lazily build the underlying LLJIT instance.
    bool ensureLLJIT();

    std::unique_ptr<llvm::orc::LLJIT> jit_;
    std::unordered_map<CacheKey, CiraJitFn, CacheKeyHash> cache_;
    std::mutex mu_;
};

} // namespace cira

#endif // CIRA_JIT_ENGINE_H
