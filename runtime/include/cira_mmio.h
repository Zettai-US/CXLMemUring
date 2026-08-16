// CIRA CXL Type-2 MMIO window.
//
// One place that knows how to reach the device control window, so the C++
// runtime (CiraRuntime.cpp), the JIT-specialized kernels (cira_jit_engine.cpp)
// and standalone tools all submit through the same path.
//
// A window can be obtained from, in priority order:
//   1. an explicit cira_mmio_config_t,
//   2. the environment (cira_mmio_open_env / cira_mmio_default),
//   3. an anonymous shared mapping, when CIRA_CXL_MMIO_EMULATE=1 — this makes
//      the whole submit/doorbell/completion path testable without an FPGA.
//
// Environment variables (CIRA_TYPE2_* accepted as legacy aliases):
//   CIRA_CXL_MMIO_PATH      sysfs resource file, /dev/mem, or any mappable file
//   CIRA_CXL_MMIO_OFFSET    mmap offset into that file (default 0)
//   CIRA_CXL_MMIO_SIZE      window size in bytes (default CIRA_CXL_CONTROL_BYTES)
//   CIRA_CXL_MMIO_ADDR      already-mapped virtual address, instead of a path
//   CIRA_CXL_MMIO_EMULATE   1 => back the window with anonymous memory
//   CIRA_CXL_MMIO_WAIT      1 => submit calls block on the completion line
//   CIRA_CXL_MMIO_TIMEOUT_NS  completion timeout, 0 = wait forever
//   CIRA_CXL_DEVICE_FUNC_ADDR device-side entry point for cira.offload

#ifndef CIRA_MMIO_H
#define CIRA_MMIO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "cira_cxl_job.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cira_mmio_window cira_mmio_window_t;

typedef struct {
    const char *path; // mappable file; NULL => use fixed_addr / emulate
    uint64_t offset; // mmap offset into path
    uint64_t size; // window bytes; 0 => CIRA_CXL_CONTROL_BYTES
    uintptr_t fixed_addr; // pre-mapped virtual address (path == NULL)
    bool emulate; // back with anonymous memory (no device present)
    bool read_only;
} cira_mmio_config_t;

#define CIRA_MMIO_OK 0
#define CIRA_MMIO_EINVAL -1
#define CIRA_MMIO_ENODEV -2
#define CIRA_MMIO_EIO -3
#define CIRA_MMIO_ETIMEDOUT -4
#define CIRA_MMIO_ERANGE -5

// ---------------------------------------------------------------------------
// Window lifetime
// ---------------------------------------------------------------------------

int cira_mmio_open(cira_mmio_window_t **out, const cira_mmio_config_t *cfg);
int cira_mmio_open_env(cira_mmio_window_t **out);
void cira_mmio_close(cira_mmio_window_t *window);

// Process-wide window, opened lazily from the environment on first use.
// Returns NULL when no device is configured; callers then fall back to
// software execution. Never freed — it lives for the life of the process.
cira_mmio_window_t *cira_mmio_default(void);

// Install a window as the process-wide default (takes ownership). Passing NULL
// drops the current default so the next cira_mmio_default() re-reads the env.
void cira_mmio_set_default(cira_mmio_window_t *window);

void *cira_mmio_base(const cira_mmio_window_t *window);
uint64_t cira_mmio_size(const cira_mmio_window_t *window);
bool cira_mmio_is_emulated(const cira_mmio_window_t *window);

// ---------------------------------------------------------------------------
// Register access — bounds-checked, volatile, explicitly ordered
// ---------------------------------------------------------------------------

uint32_t cira_mmio_read32(const cira_mmio_window_t *window, uint64_t offset);
uint64_t cira_mmio_read64(const cira_mmio_window_t *window, uint64_t offset);
int cira_mmio_write32(cira_mmio_window_t *window, uint64_t offset, uint32_t value);
int cira_mmio_write64(cira_mmio_window_t *window, uint64_t offset, uint64_t value);
int cira_mmio_write_block(cira_mmio_window_t *window, uint64_t offset, const void *src, uint64_t len);
int cira_mmio_read_block(const cira_mmio_window_t *window, uint64_t offset, void *dst, uint64_t len);

// Order prior MMIO stores against subsequent ones (doorbell commit fence).
void cira_mmio_fence(void);

// ---------------------------------------------------------------------------
// Job submission (cira.offload / cira.prefetch_* lowering targets)
// ---------------------------------------------------------------------------

// Stage `args` into the slot for `job_id` and ring the doorbell. When
// `out_seq` is non-NULL it receives the sequence number that was committed.
int cira_mmio_submit_job(cira_mmio_window_t *window, uint32_t job_id, const void *args, uint64_t arg_len,
                         uint32_t flags, uint64_t *out_seq);

// cira.offload lowering: func(operands, num_operands, completion).
// `func` may be NULL, in which case CIRA_CXL_DEVICE_FUNC_ADDR is used.
int cira_mmio_submit_call(cira_mmio_window_t *window, void *func, void **operands, uint32_t num_operands,
                          void *completion, uint64_t *out_seq);

// Wait for the device to mirror `seq` back into the host-status register.
// timeout_ns == 0 waits forever. Uses the cira_mwait.h backend.
int cira_mmio_wait_seq(const cira_mmio_window_t *window, uint64_t seq, uint64_t timeout_ns);

// Wait on a 64-byte CIRA completion line until magic == CIRA_CXL_COMPLETION_MAGIC.
int cira_mmio_wait_completion(void *completion, uint64_t timeout_ns);

// Reset a completion line so it can be awaited again (cira.future_create).
void cira_mmio_arm_completion(void *completion);

// Publish a completion in software (used by the no-device fallback path and by
// the simulator firmware).
void cira_mmio_signal_completion(void *completion, uint32_t status, uint64_t result);

// Device entry point resolved from CIRA_CXL_DEVICE_FUNC_ADDR, or NULL.
void *cira_mmio_device_func(void);
void cira_mmio_set_device_func(void *func);

// Default completion timeout from CIRA_CXL_MMIO_TIMEOUT_NS (0 = infinite).
uint64_t cira_mmio_default_timeout_ns(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CIRA_MMIO_H
