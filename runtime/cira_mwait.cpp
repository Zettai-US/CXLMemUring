// CIRA cache-resident wait primitives — see cira_mwait.h.
//
// Instruction encodings are emitted by hand rather than through the compiler
// intrinsics so this file builds without -mwaitpkg / -mmwaitx and so the
// decision to execute them stays a runtime one.

#include "cira_mwait.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <ctime>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <x86intrin.h>
#endif

namespace {

std::atomic<int> g_backend{-1}; // -1 = not detected yet
std::atomic<uint32_t> g_spin_budget{4096};
std::atomic<uint64_t> g_tsc_hz{0};

// ---------------------------------------------------------------------------
// Cycle counter
// ---------------------------------------------------------------------------

inline uint64_t read_cycles() {
#if defined(__x86_64__) || defined(__i386__)
    return __rdtsc();
#elif defined(__aarch64__)
    uint64_t v;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return 0;
#endif
}

uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// ---------------------------------------------------------------------------
// x86 monitor-wait encodings
// ---------------------------------------------------------------------------
#if defined(__x86_64__) || defined(__i386__)

// UMONITOR rax  ->  f3 0f ae f0
inline void umonitor(const volatile void *addr) {
    __asm__ volatile(".byte 0xf3, 0x0f, 0xae, 0xf0" : : "a"(addr) : "memory");
}

// UMWAIT ecx  ->  f2 0f ae f1 ; edx:eax = TSC deadline, CF set on timeout.
// control 0 = C0.2 (deeper, slower wake), 1 = C0.1 (shallow, fast wake).
inline bool umwait(uint32_t control, uint64_t deadline) {
    uint8_t timed_out;
    uint32_t lo = (uint32_t)deadline;
    uint32_t hi = (uint32_t)(deadline >> 32);
    __asm__ volatile(".byte 0xf2, 0x0f, 0xae, 0xf1\n\t"
                     "setc %0"
                     : "=r"(timed_out)
                     : "c"(control), "a"(lo), "d"(hi)
                     : "memory", "cc");
    return timed_out != 0;
}

// TPAUSE ecx  ->  66 0f ae f1
inline bool tpause(uint32_t control, uint64_t deadline) {
    uint8_t timed_out;
    uint32_t lo = (uint32_t)deadline;
    uint32_t hi = (uint32_t)(deadline >> 32);
    __asm__ volatile(".byte 0x66, 0x0f, 0xae, 0xf1\n\t"
                     "setc %0"
                     : "=r"(timed_out)
                     : "c"(control), "a"(lo), "d"(hi)
                     : "memory", "cc");
    return timed_out != 0;
}

// MONITORX  ->  0f 01 fa
inline void monitorx(const volatile void *addr) {
    __asm__ volatile(".byte 0x0f, 0x01, 0xfa" : : "a"(addr), "c"(0), "d"(0) : "memory");
}

// MWAITX  ->  0f 01 fb ; ecx bit1 enables the ebx cycle timer.
inline void mwaitx(uint32_t cycles) {
    __asm__ volatile(".byte 0x0f, 0x01, 0xfb" : : "a"(0), "c"(2), "b"(cycles) : "memory");
}

bool cpu_has_waitpkg() {
    unsigned a, b, c, d;
    if (__get_cpuid_max(0, nullptr) < 7)
        return false;
    __cpuid_count(7, 0, a, b, c, d);
    return (c & (1u << 5)) != 0; // CPUID.(EAX=7,ECX=0):ECX[5] = WAITPKG
}

bool cpu_has_monitorx() {
    unsigned a, b, c, d;
    if (__get_cpuid_max(0x80000000u, nullptr) < 0x80000001u)
        return false;
    __cpuid(0x80000001u, a, b, c, d);
    return (c & (1u << 29)) != 0; // CPUID.80000001H:ECX[29] = MONITORX
}

#endif // x86

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

bool backend_supported(cira_wait_backend_t b) {
    switch (b) {
    case CIRA_WAIT_SPIN:
        return true;
#if defined(__x86_64__) || defined(__i386__)
    case CIRA_WAIT_UMWAIT:
    case CIRA_WAIT_TPAUSE:
        return cpu_has_waitpkg();
    case CIRA_WAIT_MWAITX:
        return cpu_has_monitorx();
#endif
#if defined(__aarch64__)
    case CIRA_WAIT_WFE:
        return true;
#endif
    default:
        return false;
    }
}

cira_wait_backend_t backend_from_env() {
    const char *v = std::getenv("CIRA_WAIT_BACKEND");
    if (!v || !*v)
        return (cira_wait_backend_t)-1;
    if (!std::strcmp(v, "spin"))
        return CIRA_WAIT_SPIN;
    if (!std::strcmp(v, "umwait"))
        return CIRA_WAIT_UMWAIT;
    if (!std::strcmp(v, "tpause"))
        return CIRA_WAIT_TPAUSE;
    if (!std::strcmp(v, "mwaitx"))
        return CIRA_WAIT_MWAITX;
    if (!std::strcmp(v, "wfe"))
        return CIRA_WAIT_WFE;
    return (cira_wait_backend_t)-1;
}

cira_wait_backend_t detect_backend() {
    cira_wait_backend_t forced = backend_from_env();
    if ((int)forced >= 0) {
        return backend_supported(forced) ? forced : CIRA_WAIT_SPIN;
    }
#if defined(__x86_64__) || defined(__i386__)
    if (cpu_has_waitpkg())
        return CIRA_WAIT_UMWAIT;
    if (cpu_has_monitorx())
        return CIRA_WAIT_MWAITX;
#elif defined(__aarch64__)
    return CIRA_WAIT_WFE;
#endif
    return CIRA_WAIT_SPIN;
}

cira_wait_backend_t current_backend() {
    int b = g_backend.load(std::memory_order_acquire);
    if (b >= 0)
        return (cira_wait_backend_t)b;
    cira_wait_backend_t detected = detect_backend();
    g_backend.store((int)detected, std::memory_order_release);
    return detected;
}

// ---------------------------------------------------------------------------
// Core wait loop
// ---------------------------------------------------------------------------

// Longest single monitor-wait. The OS caps UMWAIT/TPAUSE through
// IA32_UMWAIT_CONTROL (typically ~100us) and WFE may wake spuriously, so the
// caller-visible predicate is always re-checked in the enclosing loop.
constexpr uint64_t kWaitQuantumCycles = 20000;

// Arm a monitor on `addr` (where the backend has one) and sleep. Returns after
// a device store, a spurious wake, or the quantum expiring.
void monitor_wait_once(const volatile void *addr) {
    switch (current_backend()) {
#if defined(__x86_64__) || defined(__i386__)
    case CIRA_WAIT_UMWAIT:
        umonitor(addr);
        // Re-check happens in the caller; arming before the final read is what
        // closes the lost-wakeup window.
        umwait(1, read_cycles() + kWaitQuantumCycles);
        return;
    case CIRA_WAIT_TPAUSE:
        tpause(1, read_cycles() + kWaitQuantumCycles);
        return;
    case CIRA_WAIT_MWAITX:
        monitorx(addr);
        mwaitx((uint32_t)kWaitQuantumCycles);
        return;
#endif
#if defined(__aarch64__)
    case CIRA_WAIT_WFE: {
        uint32_t scratch;
        __asm__ volatile("ldxr %w0, [%1]" : "=&r"(scratch) : "r"(addr) : "memory");
        (void)scratch;
        __asm__ volatile("wfe" ::: "memory");
        return;
    }
#endif
    default:
        cira_wait_relax();
        return;
    }
}

enum class Predicate { Equal, NotEqual };

bool wait_impl(const volatile uint32_t *flag, uint32_t value, Predicate pred, uint64_t timeout_ns) {
    if (!flag)
        return false;

    auto satisfied = [&]() {
        uint32_t observed = __atomic_load_n(flag, __ATOMIC_RELAXED);
        return pred == Predicate::Equal ? observed == value : observed != value;
    };

    if (satisfied()) {
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        return true;
    }

    const uint64_t deadline_ns = timeout_ns ? now_ns() + timeout_ns : 0;

    // Phase 1: bounded PAUSE spin. Fine-grained CIRA offloads complete inside
    // this window, so they never pay monitor setup cost.
    uint32_t budget = g_spin_budget.load(std::memory_order_relaxed);
    for (uint32_t i = 0; i < budget; ++i) {
        if (satisfied()) {
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            return true;
        }
        cira_wait_relax();
    }

    // Phase 2: monitor-wait until the device stores to the line.
    for (;;) {
        monitor_wait_once(flag);
        if (satisfied()) {
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            return true;
        }
        if (deadline_ns && now_ns() >= deadline_ns)
            return false;
    }
}

} // namespace

extern "C" {

cira_wait_backend_t cira_wait_backend(void) { return current_backend(); }

const char *cira_wait_backend_name(cira_wait_backend_t backend) {
    switch (backend) {
    case CIRA_WAIT_SPIN:
        return "spin";
    case CIRA_WAIT_UMWAIT:
        return "umwait";
    case CIRA_WAIT_TPAUSE:
        return "tpause";
    case CIRA_WAIT_MWAITX:
        return "mwaitx";
    case CIRA_WAIT_WFE:
        return "wfe";
    default:
        return "unknown";
    }
}

void cira_wait_set_backend(cira_wait_backend_t backend) {
    g_backend.store(backend_supported(backend) ? (int)backend : (int)CIRA_WAIT_SPIN, std::memory_order_release);
}

void cira_wait_set_spin_budget(uint32_t iterations) { g_spin_budget.store(iterations, std::memory_order_relaxed); }

uint32_t cira_wait_spin_budget(void) { return g_spin_budget.load(std::memory_order_relaxed); }

void cira_wait_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#elif defined(__riscv)
    // Zihintpause; decodes as a plain FENCE on cores without it.
    __asm__ volatile(".word 0x0100000f" ::: "memory");
#else
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
#endif
}

bool cira_wait_u32(const volatile uint32_t *flag, uint32_t expect, uint64_t timeout_ns) {
    return wait_impl(flag, expect, Predicate::Equal, timeout_ns);
}

bool cira_wait_u32_ne(const volatile uint32_t *flag, uint32_t unexpected, uint64_t timeout_ns) {
    return wait_impl(flag, unexpected, Predicate::NotEqual, timeout_ns);
}

uint64_t cira_wait_tsc_hz(void) {
    uint64_t cached = g_tsc_hz.load(std::memory_order_acquire);
    if (cached)
        return cached;

    uint64_t c0 = read_cycles();
    if (c0 == 0)
        return 0;

    uint64_t t0 = now_ns();
    struct timespec nap = {0, 1000000}; // 1 ms
    nanosleep(&nap, nullptr);
    uint64_t t1 = now_ns();
    uint64_t c1 = read_cycles();

    if (t1 <= t0 || c1 <= c0)
        return 0;
    uint64_t hz = (c1 - c0) * 1000000000ull / (t1 - t0);
    g_tsc_hz.store(hz, std::memory_order_release);
    return hz;
}

} // extern "C"
