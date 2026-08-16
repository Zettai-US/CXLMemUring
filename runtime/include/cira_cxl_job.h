// CIRA <-> CXL Type-2 control-window layout.
//
// This is the single definition of the host/device contract described in the
// CIRA paper's runtime section: the host writes a task descriptor into a
// CXL-visible control window, commits a doorbell with a monotonically
// increasing sequence number, and the device firmware polls the doorbell,
// executes the job, and publishes a 64-byte completion line the host waits on
// with cache-resident polling (cira_mwait.h).
//
// Both sides include this header: the host submit path (cira_mmio.cpp,
// CiraRuntime.cpp) and the device firmware service (vortex_device.cpp).
// Nothing here may be reordered or resized without changing both.
//
//   0x0000  doorbell        (cira_cxl_doorbell_t)
//   0x0100  arg slot 0..N   (cira_cxl_arg_slot_t + payload, 0x400 each)
//   0x1f20  host status     (cira_cxl_host_status_t) — device -> host mirror
//   0x2000  end of window

#ifndef CIRA_CXL_JOB_H
#define CIRA_CXL_JOB_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// "VXCXLJOB" / "HGPUPACC" — the second value keeps wire compatibility with the
// hetGPU PACC firmware this protocol was derived from.
#define CIRA_CXL_JOB_MAGIC 0x565843584c4a4f42ULL
#define CIRA_CXL_PACC_JOB_MAGIC 0x4847505550414343ULL
#define CIRA_CXL_JOB_VERSION 1u
#define CIRA_CXL_COMPLETION_MAGIC 0xDEADBEEFu
#define CIRA_CXL_CACHELINE_SIZE 64u

#define CIRA_CXL_CONTROL_BYTES 0x2000ULL
#define CIRA_CXL_DOORBELL_OFF 0x0ULL
#define CIRA_CXL_ARG_BASE_OFF 0x100ULL
#define CIRA_CXL_ARG_SLOT_BYTES 0x400ULL
#define CIRA_CXL_STATUS_OFF 0x1f20ULL

typedef enum {
    CIRA_CXL_JOB_NOP = 0,
    CIRA_CXL_JOB_INSTALL_CACHELINE = 1, // cira.install_cacheline
    CIRA_CXL_JOB_PREFETCH_CHAIN = 2, // cira.prefetch_indirect
    CIRA_CXL_JOB_STREAM_PREFETCH = 3, // cira.prefetch_stream
    CIRA_CXL_JOB_CALL = 4, // cira.offload / cira.speculate
    CIRA_CXL_JOB_MAX = CIRA_CXL_JOB_CALL,
} cira_cxl_job_id_t;

typedef enum {
    CIRA_CXL_STATUS_SUCCESS = 0,
    CIRA_CXL_STATUS_RUNNING = 1,
    CIRA_CXL_STATUS_BAD_VERSION = 0xffff0001u,
    CIRA_CXL_STATUS_BAD_ARGS = 0xffff0002u,
    CIRA_CXL_STATUS_BAD_JOB = 0xffff00ffu,
} cira_cxl_status_t;

// Job flags carried in the doorbell.
#define CIRA_CXL_FLAG_SPECULATIVE (1u << 0) // cira.speculate: lower priority

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t job_id;
    uint32_t flags;
    uint32_t status;
    uint64_t seq;
} cira_cxl_doorbell_t;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t job_id;
    uint64_t seq;
    uint64_t arg_len;
} cira_cxl_arg_slot_t;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t job_id;
    uint32_t status;
    uint64_t seq;
} cira_cxl_host_status_t;

#define CIRA_CXL_ARG_PAYLOAD_BYTES (CIRA_CXL_ARG_SLOT_BYTES - sizeof(cira_cxl_arg_slot_t))

// The completion line the host monitors. `magic` is written last, so observing
// CIRA_CXL_COMPLETION_MAGIC implies the other fields are visible.
typedef struct {
    uint32_t magic;
    uint32_t status;
    uint64_t result;
    uint64_t cycles;
    uint64_t timestamp;
    uint8_t reserved[32];
} __attribute__((aligned(CIRA_CXL_CACHELINE_SIZE))) cira_cxl_completion_t;

// ---------------------------------------------------------------------------
// Job argument payloads (written into the arg slot for the matching job_id)
// ---------------------------------------------------------------------------

typedef struct {
    uint64_t addr;
    uint64_t size;
    uint64_t completion_addr;
    uint32_t cache_level;
    uint32_t reserved;
} cira_cxl_install_cacheline_job_t;

typedef struct {
    uint64_t start_node_addr;
    uint64_t buf_addr;
    uint64_t completion_addr;
    uint32_t depth;
    uint32_t next_ptr_offset;
    uint32_t data_offset;
    uint32_t data_size;
} cira_cxl_prefetch_chain_job_t;

typedef struct {
    uint64_t base_addr;
    uint64_t buf_addr;
    uint64_t completion_addr;
    uint64_t count;
    uint32_t stride;
    uint32_t elem_size;
    uint32_t reserved;
} cira_cxl_stream_prefetch_job_t;

typedef struct {
    uint64_t func_addr;
    uint64_t operands_addr;
    uint64_t completion_addr;
    uint32_t num_operands;
    uint32_t reserved;
} cira_cxl_call_job_t;

static inline int cira_cxl_valid_magic(uint64_t magic) {
    return magic == CIRA_CXL_JOB_MAGIC || magic == CIRA_CXL_PACC_JOB_MAGIC;
}

static inline uint64_t cira_cxl_arg_slot_off(uint32_t job_id) {
    return CIRA_CXL_ARG_BASE_OFF + (uint64_t)job_id * CIRA_CXL_ARG_SLOT_BYTES;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CIRA_CXL_JOB_H
