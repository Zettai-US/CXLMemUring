// Tests for the ORC-backed CIRA JIT specializer (cira_jit_engine.h).
//
// The IR below stands in for a compiler-emitted offload template: knobs are
// external sentinel globals, and the region calls into the CIRA runtime ABI.
// We assert that (a) knobs of every supported type are folded to literals,
// (b) runtime ABI symbols resolve, (c) equal decisions share compiled code
// while different ones do not, and (d) device knobs describe the configured
// CXL control window.

#include "cira_jit_engine.h"

#include "cira_cxl_job.h"
#include "cira_jit.h"
#include "cira_mmio.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

int g_failures = 0;

void check(bool cond, const char *msg) {
    std::printf("  [%s] %s\n", cond ? " ok " : "FAIL", msg);
    if (!cond)
        ++g_failures;
}

// batch * depth, then +1 when should_offload, scaled by the split. The point
// is that after specialization none of the sentinel loads survive.
const char *kKnobTemplate = R"IR(
@cira_kBatchSize        = external global i32
@cira_kTraversalDepth   = external global i32
@cira_kPipelineDistance = external global i32
@cira_kHostDeviceSplit  = external global float
@cira_kShouldOffload    = external global i8

define i32 @cira_region() {
entry:
  %batch = load i32, ptr @cira_kBatchSize
  %depth = load i32, ptr @cira_kTraversalDepth
  %dist  = load i32, ptr @cira_kPipelineDistance
  %off   = load i8,  ptr @cira_kShouldOffload
  %split = load float, ptr @cira_kHostDeviceSplit

  %prod  = mul i32 %batch, %depth
  %sum   = add i32 %prod, %dist
  %offz  = zext i8 %off to i32
  %tot   = add i32 %sum, %offz

  %scale = fmul float %split, 1.000000e+02
  %si    = fptosi float %scale to i32
  %res   = add i32 %tot, %si
  ret i32 %res
}
)IR";

// A template that calls the runtime ABI: proves the JIT resolves cira_* and
// that a specialized kernel can drive the device path itself.
const char *kAbiTemplate = R"IR(
@cira_kCompletionMagic = external global i32

declare void @cira_wait_relax()
declare void @cira_mmio_signal_completion(ptr, i32, i64)
declare i32  @cira_mmio_wait_completion(ptr, i64)

define i32 @cira_region_abi(ptr %completion) {
entry:
  call void @cira_wait_relax()
  call void @cira_mmio_signal_completion(ptr %completion, i32 0, i64 4242)
  %rc = call i32 @cira_mmio_wait_completion(ptr %completion, i64 1000000000)
  %magic = load i32, ptr @cira_kCompletionMagic
  %ok = add i32 %rc, %magic
  ret i32 %ok
}
)IR";

cira_jit_decision_t make_decision(uint32_t batch, uint32_t depth, uint32_t dist, float split, bool offload) {
    cira_jit_decision_t d;
    std::memset(&d, 0, sizeof(d));
    d.batch_size = batch;
    d.traversal_depth = depth;
    d.pipeline_distance = dist;
    d.host_device_split = split;
    d.should_offload = offload;
    return d;
}

void test_knob_folding(cira::CiraJitEngine &engine) {
    std::printf("knob folding\n");
    using RegionFn = int (*)();

    auto d1 = make_decision(16, 4, 7, 0.25f, true);
    auto fn1 = reinterpret_cast<RegionFn>(engine.specializeFromIR(kKnobTemplate, "cira_region", d1));
    check(fn1 != nullptr, "specialization from decision succeeds");
    // 16*4 + 7 + 1 + (0.25 * 100) = 97
    if (fn1)
        check(fn1() == 97, "all knob types fold to the expected literals");

    // Same decision -> same compiled code, no recompilation.
    auto fn1b = reinterpret_cast<RegionFn>(engine.specializeFromIR(kKnobTemplate, "cira_region", d1));
    check(fn1b == fn1, "identical decisions share cached code");

    // Different decision -> different code, and both stay valid.
    auto d2 = make_decision(8, 8, 3, 0.5f, false);
    auto fn2 = reinterpret_cast<RegionFn>(engine.specializeFromIR(kKnobTemplate, "cira_region", d2));
    check(fn2 != nullptr, "a second schedule of the same region specializes");
    check(fn2 != fn1, "different decisions produce different code");
    // 8*8 + 3 + 0 + 50 = 117
    if (fn2)
        check(fn2() == 117, "second specialization folds its own knobs");
    if (fn1)
        check(fn1() == 97, "first specialization is still callable");
}

void test_generic_spec(cira::CiraJitEngine &engine) {
    std::printf("generic knob spec\n");
    using RegionFn = int (*)();

    // Knobs supplied directly rather than through the cost model, including a
    // type the template does not declare (ignored) and an -O0 build.
    cira::CiraJitSpec spec;
    spec.kernel = "cira_region";
    spec.optLevel = 0;
    spec.variant = "unit-test";
    spec.knobs = {
        cira::CiraJitKnob::i32(cira::kSentinelBatchSize, 2),
        cira::CiraJitKnob::i32(cira::kSentinelTraversalDepth, 3),
        cira::CiraJitKnob::i32(cira::kSentinelPipelineDistance, 4),
        cira::CiraJitKnob::f32(cira::kSentinelHostDeviceSplit, 0.0f),
        cira::CiraJitKnob::boolean(cira::kSentinelShouldOffload, false),
        cira::CiraJitKnob::i64("cira_kNotInThisTemplate", 99),
    };
    auto fn = reinterpret_cast<RegionFn>(engine.specializeText(kKnobTemplate, spec));
    check(fn != nullptr, "explicit knob list specializes at -O0");
    if (fn)
        check(fn() == 10, "2*3 + 4 + 0 + 0 == 10"); // unknown knob ignored

    cira::CiraJitSpec missing = spec;
    missing.kernel = "no_such_symbol";
    check(engine.specializeText(kKnobTemplate, missing) == nullptr,
          "missing kernel symbol reports failure instead of crashing");
}

void test_runtime_abi(cira::CiraJitEngine &engine) {
    std::printf("runtime ABI binding\n");
    using AbiFn = int (*)(void *);

    cira::CiraJitSpec spec;
    spec.kernel = "cira_region_abi";
    spec.knobs = {cira::CiraJitKnob::i32(cira::kSentinelCompletionMagic, 1)};

    auto fn = reinterpret_cast<AbiFn>(engine.specializeText(kAbiTemplate, spec));
    check(fn != nullptr, "template calling the CIRA runtime ABI links");

    if (fn) {
        alignas(CIRA_CXL_CACHELINE_SIZE) cira_cxl_completion_t completion;
        cira_mmio_arm_completion(&completion);
        int rc = fn(&completion);
        check(rc == 1, "JIT-ed kernel drove the completion protocol");
        check(completion.result == 4242, "JIT-ed kernel published the completion line");
    }
}

void test_device_knobs() {
    std::printf("device knobs\n");

    cira_mmio_set_default(nullptr);
    auto none = cira::ciraKnobsFromDevice();
    check(none.empty(), "no knobs when no CXL device is configured");

    cira_mmio_config_t cfg = {};
    cfg.emulate = true;
    cfg.size = CIRA_CXL_CONTROL_BYTES;
    cira_mmio_window_t *w = nullptr;
    if (cira_mmio_open(&w, &cfg) != CIRA_MMIO_OK) {
        check(false, "could not open emulated window");
        return;
    }
    cira_mmio_set_default(w);

    auto knobs = cira::ciraKnobsFromDevice();
    check(knobs.size() == 5, "device knob set is populated");

    bool base_ok = false, magic_ok = false;
    for (const auto &k : knobs) {
        if (k.name == cira::kSentinelMmioBase) {
            base_ok = k.ivalue == (uint64_t)(uintptr_t)cira_mmio_base(w);
        }
        if (k.name == cira::kSentinelCompletionMagic) {
            magic_ok = k.ivalue == CIRA_CXL_COMPLETION_MAGIC;
        }
    }
    check(base_ok, "MMIO base knob points at the mapped control window");
    check(magic_ok, "completion magic knob matches the wire protocol");

    cira_mmio_set_default(nullptr);
}

} // namespace

int main() {
    std::printf("=== CIRA JIT engine tests ===\n");

    cira::CiraJitEngine engine;
    test_knob_folding(engine);
    test_generic_spec(engine);
    test_runtime_abi(engine);
    test_device_knobs();

    std::printf("  cached specializations: %zu\n", engine.cacheSize());
    engine.resetCache();
    if (engine.cacheSize() != 0) {
        std::printf("  [FAIL] resetCache did not clear the cache\n");
        ++g_failures;
    }

    if (g_failures) {
        std::printf("\n%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("\nAll checks passed\n");
    return 0;
}
