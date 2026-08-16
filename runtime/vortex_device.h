// Vortex Device Interface for CIRA Runtime
// Implements host-to-device offload ABI for Vortex RISC-V SIMT backend

#ifndef VORTEX_DEVICE_H
#define VORTEX_DEVICE_H

#include "cira_cxl_job.h"
#include "shared_protocol.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Vortex device handle (opaque)
typedef struct vortex_device *vortex_device_h;
typedef struct vortex_buffer *vortex_buffer_h;
typedef struct vortex_kernel *vortex_kernel_h;

// Device capabilities
typedef struct {
    uint32_t num_cores;
    uint32_t num_warps_per_core;
    uint32_t num_threads_per_warp;
    uint64_t global_mem_size;
    uint64_t local_mem_size;
    uint32_t cache_line_size;
    uint32_t isa_flags;
} vortex_device_caps_t;

// Kernel launch parameters (CUDA-like grid/block)
typedef struct {
    uint32_t grid_dim_x;
    uint32_t grid_dim_y;
    uint32_t grid_dim_z;
    uint32_t block_dim_x;
    uint32_t block_dim_y;
    uint32_t block_dim_z;
    uint32_t shared_mem_bytes;
} vortex_launch_params_t;

// Kernel argument types
typedef enum {
    VORTEX_ARG_VALUE, // Pass by value
    VORTEX_ARG_BUFFER, // Device buffer reference
    VORTEX_ARG_LOCAL // Local memory allocation
} vortex_arg_type_t;

// Kernel argument descriptor
typedef struct {
    void *host_ptr; // Host-side pointer to argument
    uint64_t size; // Size in bytes
    uint32_t alignment; // Alignment requirement
    vortex_arg_type_t type; // Argument type
} vortex_kernel_arg_t;

// Memory transfer direction
typedef enum {
    VORTEX_MEMCPY_HOST_TO_DEVICE,
    VORTEX_MEMCPY_DEVICE_TO_HOST,
    VORTEX_MEMCPY_DEVICE_TO_DEVICE
} vortex_memcpy_kind_t;

// Error codes
#define VORTEX_SUCCESS 0
#define VORTEX_ERROR_INIT -1
#define VORTEX_ERROR_MEMORY -2
#define VORTEX_ERROR_KERNEL -3
#define VORTEX_ERROR_LAUNCH -4
#define VORTEX_ERROR_TIMEOUT -5
#define VORTEX_ERROR_INVALID -6

// CXL Type-2 MMIO control window layout.  The wire format itself lives in
// cira_cxl_job.h and is shared with the host submit path in cira_mmio.cpp.
#define VORTEX_CXL_MMIO_CONTROL_BYTES CIRA_CXL_CONTROL_BYTES

// ============================================================================
// Device Management
// ============================================================================

// Initialize Vortex device (connects to RTL simulator or hardware)
int vortex_device_init(vortex_device_h *device, const char *sim_path);

// Get device capabilities
int vortex_device_get_caps(vortex_device_h device, vortex_device_caps_t *caps);

// Cleanup and shutdown device
int vortex_device_destroy(vortex_device_h device);

// ============================================================================
// Memory Management
// ============================================================================

// Allocate device memory
int vortex_malloc(vortex_device_h device, vortex_buffer_h *buffer, size_t size);

// Free device memory
int vortex_free(vortex_buffer_h buffer);

// Get device memory address
int vortex_buffer_get_address(vortex_buffer_h buffer, uint64_t *dev_addr);

// Copy memory between host and device
int vortex_memcpy(void *dst, const void *src, size_t size, vortex_memcpy_kind_t kind);

// Copy with offset
int vortex_memcpy_offset(vortex_buffer_h buffer, uint64_t offset, const void *src, size_t size,
                         vortex_memcpy_kind_t kind);

// ============================================================================
// Kernel Management
// ============================================================================

// Load kernel from compiled RISC-V binary
int vortex_kernel_load(vortex_device_h device, vortex_kernel_h *kernel, const void *binary_data, size_t binary_size);

// Load kernel from file
int vortex_kernel_load_file(vortex_device_h device, vortex_kernel_h *kernel, const char *binary_path);

// Set kernel arguments
int vortex_kernel_set_args(vortex_kernel_h kernel, const vortex_kernel_arg_t *args, uint32_t num_args);

// Unload kernel
int vortex_kernel_unload(vortex_kernel_h kernel);

// ============================================================================
// Kernel Execution
// ============================================================================

// Launch kernel on device
int vortex_kernel_launch(vortex_device_h device, vortex_kernel_h kernel, const vortex_launch_params_t *params);

// Wait for kernel completion (with timeout in ms)
int vortex_device_wait(vortex_device_h device, uint64_t timeout_ms);

// Synchronize device (blocking)
int vortex_device_synchronize(vortex_device_h device);

// ============================================================================
// CXL Type-2 MMIO offload submission
// ============================================================================

// Stage a firmware job into a CXL-visible MMIO/control window and ring the
// doorbell.  `seq == 0` asks the runtime to allocate a monotonic sequence.
int vortex_cxl_submit_job_mmio(void *control_window, uint32_t job_id, const void *arg_data, uint64_t arg_len,
                               uint64_t seq, uint32_t flags);

// Convenience wrapper for the generic device call job:
//   func_ptr(operands, num_operands, completion_ptr)
int vortex_cxl_submit_call_mmio(void *control_window, uint64_t seq, void *func_ptr, void **operands,
                                uint32_t num_operands, void *completion_ptr);

// ============================================================================
// Performance and Debugging
// ============================================================================

// Query device performance counters
int vortex_device_get_perf_counter(vortex_device_h device, const char *counter_name, uint64_t *value);

// Dump all performance counters
int vortex_device_dump_perf(vortex_device_h device, const char *output_path);

// Enable/disable debug trace
int vortex_device_set_debug(vortex_device_h device, int enable);

#ifdef __cplusplus
}
#endif

#endif // VORTEX_DEVICE_H
