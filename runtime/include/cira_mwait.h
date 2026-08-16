// CIRA cache-resident wait primitives.
//
// The CIRA paper specifies that device-to-host completion is signalled by a
// store to a host-visible cache line, and that the host "waits using a
// cache-resident polling mechanism that avoids heavyweight interrupts".  This
// header is the single implementation of that wait for the whole runtime:
// futures, phase barriers, LLC tiles, MMIO doorbell completion, and
// JIT-specialized kernels all funnel through cira_wait_u32*().
//
// Backend selection is a *runtime* decision (CPUID / HWCAP), never a compile
// time one, so a binary built on one host still runs on another:
//
//   x86-64 + WAITPKG : UMONITOR/UMWAIT  (C0.1 — short wake latency)
//   x86-64 + WAITPKG : TPAUSE           (no monitor, used for backoff)
//   x86-64 + MONITORX: MONITORX/MWAITX  (AMD user-mode monitor-wait)
//   aarch64          : LDXR + WFE
//   other            : PAUSE / YIELD spin
//
// All backends are woken by an ordinary store to the monitored line, so the
// device side needs no doorbell-specific signalling path.

#ifndef CIRA_MWAIT_H
#define CIRA_MWAIT_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CIRA_WAIT_SPIN = 0, // PAUSE / YIELD only
    CIRA_WAIT_UMWAIT = 1, // UMONITOR + UMWAIT (Intel WAITPKG)
    CIRA_WAIT_TPAUSE = 2, // TPAUSE (Intel WAITPKG, no address monitor)
    CIRA_WAIT_MWAITX = 3, // MONITORX + MWAITX (AMD)
    CIRA_WAIT_WFE = 4, // LDXR + WFE (aarch64)
} cira_wait_backend_t;

// Backend actually in use on this host. Detected once, then cached.
cira_wait_backend_t cira_wait_backend(void);
const char *cira_wait_backend_name(cira_wait_backend_t backend);

// Force a backend (evaluation / A-B measurement). Requesting a backend the
// host does not support silently degrades to CIRA_WAIT_SPIN. Also settable
// through the CIRA_WAIT_BACKEND environment variable
// ("spin", "umwait", "tpause", "mwaitx", "wfe").
void cira_wait_set_backend(cira_wait_backend_t backend);

// Number of PAUSE iterations executed before the first monitor-wait is armed.
// Short waits stay in the spin phase and never pay the monitor setup cost.
void cira_wait_set_spin_budget(uint32_t iterations);
uint32_t cira_wait_spin_budget(void);

// Block until *flag == expect (cira_wait_u32) or *flag != unexpected
// (cira_wait_u32_ne). timeout_ns == 0 waits forever.
//
// Returns true when the predicate was observed, false on timeout. An acquire
// fence is executed before returning true, so payload bytes written by the
// device before the flag store are visible to the caller.
bool cira_wait_u32(const volatile uint32_t *flag, uint32_t expect, uint64_t timeout_ns);
bool cira_wait_u32_ne(const volatile uint32_t *flag, uint32_t unexpected, uint64_t timeout_ns);

// One iteration of the spin phase. Exposed so callers with their own polling
// loop (e.g. multi-future barriers) still use the right relax instruction.
void cira_wait_relax(void);

// Estimated invariant-TSC frequency, lazily calibrated against
// CLOCK_MONOTONIC. Only used when a timeout is requested; returns 0 when no
// usable cycle counter exists.
uint64_t cira_wait_tsc_hz(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CIRA_MWAIT_H
