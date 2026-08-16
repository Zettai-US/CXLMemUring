// Vortex Device Interface Implementation
// Bridges CIRA runtime (x86_64 host) with Vortex RISC-V SIMT device

#include "vortex_device.h"
#include "cira_cxl_job.h"
#include <atomic>
#include <cinttypes>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#ifdef VORTEX_FOUND
#include <vortex.h> // Vortex runtime header
#else
// Stub definitions when Vortex SDK is not available
typedef void *vx_device_h;
typedef void *vx_buffer_h;
#define VX_CAPS_NUM_THREADS 0x1
#define VX_CAPS_NUM_WARPS 0x2
#define VX_CAPS_NUM_CORES 0x3
#define VX_CAPS_CACHE_LINE_SIZE 0x4
#define VX_CAPS_GLOBAL_MEM_SIZE 0x5
#define VX_CAPS_LOCAL_MEM_SIZE 0x6
#define VX_CAPS_ISA_FLAGS 0x7
#define VX_MEM_READ_WRITE 0x3
#define VX_MEM_READ 0x1
#define VX_MEM_WRITE 0x2
#define VX_MAX_TIMEOUT 0xFFFFFFFFULL
static inline int vx_dev_open(vx_device_h *) { return -1; }
static inline int vx_dev_close(vx_device_h) { return 0; }
static inline int vx_dev_caps(vx_device_h, uint32_t, uint64_t *) { return -1; }
static inline int vx_mem_alloc(vx_device_h, uint64_t, int, vx_buffer_h *) { return -1; }
static inline int vx_mem_free(vx_buffer_h) { return 0; }
static inline int vx_mem_address(vx_buffer_h, uint64_t *) { return -1; }
static inline int vx_copy_to_dev(vx_buffer_h, const void *, uint64_t, uint64_t) { return -1; }
static inline int vx_copy_from_dev(void *, vx_buffer_h, uint64_t, uint64_t) { return -1; }
static inline int vx_start(vx_device_h, vx_buffer_h, vx_buffer_h) { return -1; }
static inline int vx_ready_wait(vx_device_h, uint64_t) { return -1; }
static inline int vx_upload_kernel_file(vx_device_h, const char *, vx_buffer_h *) { return -1; }
static inline int vx_upload_kernel_bytes(vx_device_h, const void *, uint32_t, vx_buffer_h *) { return -1; }
static inline int vx_dump_perf(vx_device_h, FILE *) { return 0; }
#endif

// Internal device structure
struct vortex_device {
    vx_device_h vx_dev;
    vortex_device_caps_t caps;
    bool initialized;
    bool debug_enabled;
    std::map<std::string, uint64_t> perf_counters;
};

// Internal buffer structure
struct vortex_buffer {
    vx_buffer_h vx_buf;
    vortex_device_h device;
    uint64_t dev_addr;
    size_t size;
};

// Internal kernel structure
struct vortex_kernel {
    vortex_device_h device;
    vx_buffer_h kernel_buffer;
    vx_buffer_h args_buffer;
    uint64_t kernel_addr;
    uint64_t args_addr;
    std::vector<vortex_kernel_arg_t> args;
};

// ============================================================================
// Device Management
// ============================================================================

int vortex_device_init(vortex_device_h *device, const char *sim_path) {
    if (!device)
        return VORTEX_ERROR_INVALID;

    // Allocate device structure
    *device = new vortex_device();
    if (!*device)
        return VORTEX_ERROR_MEMORY;

    // Set environment variable for simulator path if provided
    if (sim_path) {
        setenv("VORTEX_SIM_PATH", sim_path, 1);
    }

    // Open Vortex device
    int ret = vx_dev_open(&((*device)->vx_dev));
    if (ret != 0) {
        delete *device;
        *device = nullptr;
        std::cerr << "Failed to open Vortex device: " << ret << std::endl;
        return VORTEX_ERROR_INIT;
    }

    // Query device capabilities
    uint64_t value;

    vx_dev_caps((*device)->vx_dev, VX_CAPS_NUM_CORES, &value);
    (*device)->caps.num_cores = (uint32_t)value;

    vx_dev_caps((*device)->vx_dev, VX_CAPS_NUM_WARPS, &value);
    (*device)->caps.num_warps_per_core = (uint32_t)value;

    vx_dev_caps((*device)->vx_dev, VX_CAPS_NUM_THREADS, &value);
    (*device)->caps.num_threads_per_warp = (uint32_t)value;

    vx_dev_caps((*device)->vx_dev, VX_CAPS_GLOBAL_MEM_SIZE, &value);
    (*device)->caps.global_mem_size = value;

    vx_dev_caps((*device)->vx_dev, VX_CAPS_LOCAL_MEM_SIZE, &value);
    (*device)->caps.local_mem_size = value;

    vx_dev_caps((*device)->vx_dev, VX_CAPS_CACHE_LINE_SIZE, &value);
    (*device)->caps.cache_line_size = (uint32_t)value;

    vx_dev_caps((*device)->vx_dev, VX_CAPS_ISA_FLAGS, &value);
    (*device)->caps.isa_flags = (uint32_t)value;

    (*device)->initialized = true;
    (*device)->debug_enabled = false;

    std::cout << "Vortex Device Initialized:" << std::endl;
    std::cout << "  Cores: " << (*device)->caps.num_cores << std::endl;
    std::cout << "  Warps per core: " << (*device)->caps.num_warps_per_core << std::endl;
    std::cout << "  Threads per warp: " << (*device)->caps.num_threads_per_warp << std::endl;
    std::cout << "  Global memory: " << (*device)->caps.global_mem_size << " bytes" << std::endl;

    return VORTEX_SUCCESS;
}

int vortex_device_get_caps(vortex_device_h device, vortex_device_caps_t *caps) {
    if (!device || !caps)
        return VORTEX_ERROR_INVALID;
    if (!device->initialized)
        return VORTEX_ERROR_INIT;

    *caps = device->caps;
    return VORTEX_SUCCESS;
}

int vortex_device_destroy(vortex_device_h device) {
    if (!device)
        return VORTEX_ERROR_INVALID;

    if (device->initialized) {
        vx_dev_close(device->vx_dev);
    }

    delete device;
    return VORTEX_SUCCESS;
}

// ============================================================================
// Memory Management
// ============================================================================

int vortex_malloc(vortex_device_h device, vortex_buffer_h *buffer, size_t size) {
    if (!device || !buffer || size == 0)
        return VORTEX_ERROR_INVALID;
    if (!device->initialized)
        return VORTEX_ERROR_INIT;

    // Allocate buffer structure
    *buffer = new vortex_buffer();
    if (!*buffer)
        return VORTEX_ERROR_MEMORY;

    (*buffer)->device = device;
    (*buffer)->size = size;

    // Allocate device memory
    int ret = vx_mem_alloc(device->vx_dev, size, VX_MEM_READ_WRITE, &((*buffer)->vx_buf));
    if (ret != 0) {
        delete *buffer;
        *buffer = nullptr;
        return VORTEX_ERROR_MEMORY;
    }

    // Get device address
    ret = vx_mem_address((*buffer)->vx_buf, &((*buffer)->dev_addr));
    if (ret != 0) {
        vx_mem_free((*buffer)->vx_buf);
        delete *buffer;
        *buffer = nullptr;
        return VORTEX_ERROR_MEMORY;
    }

    if (device->debug_enabled) {
        std::cout << "Allocated " << size << " bytes at device address 0x" << std::hex << (*buffer)->dev_addr
                  << std::dec << std::endl;
    }

    return VORTEX_SUCCESS;
}

int vortex_free(vortex_buffer_h buffer) {
    if (!buffer)
        return VORTEX_ERROR_INVALID;

    vx_mem_free(buffer->vx_buf);
    delete buffer;
    return VORTEX_SUCCESS;
}

int vortex_buffer_get_address(vortex_buffer_h buffer, uint64_t *dev_addr) {
    if (!buffer || !dev_addr)
        return VORTEX_ERROR_INVALID;

    *dev_addr = buffer->dev_addr;
    return VORTEX_SUCCESS;
}

int vortex_memcpy(void *dst, const void *src, size_t size, vortex_memcpy_kind_t kind) {
    if (!dst || !src || size == 0)
        return VORTEX_ERROR_INVALID;

    // This is a simplified version - assumes dst/src are buffer handles for device transfers
    switch (kind) {
    case VORTEX_MEMCPY_HOST_TO_DEVICE: {
        vortex_buffer_h dev_buf = static_cast<vortex_buffer_h>(dst);
        int ret = vx_copy_to_dev(dev_buf->vx_buf, src, 0, size);
        return (ret == 0) ? VORTEX_SUCCESS : VORTEX_ERROR_MEMORY;
    }
    case VORTEX_MEMCPY_DEVICE_TO_HOST: {
        vortex_buffer_h dev_buf = static_cast<vortex_buffer_h>(const_cast<void *>(src));
        int ret = vx_copy_from_dev(dst, dev_buf->vx_buf, 0, size);
        return (ret == 0) ? VORTEX_SUCCESS : VORTEX_ERROR_MEMORY;
    }
    case VORTEX_MEMCPY_DEVICE_TO_DEVICE:
        // Not directly supported - would need staging buffer
        return VORTEX_ERROR_INVALID;
    }
    return VORTEX_ERROR_INVALID;
}

int vortex_memcpy_offset(vortex_buffer_h buffer, uint64_t offset, const void *src, size_t size,
                         vortex_memcpy_kind_t kind) {
    if (!buffer || !src || size == 0)
        return VORTEX_ERROR_INVALID;
    if (offset + size > buffer->size)
        return VORTEX_ERROR_INVALID;

    switch (kind) {
    case VORTEX_MEMCPY_HOST_TO_DEVICE: {
        int ret = vx_copy_to_dev(buffer->vx_buf, src, offset, size);
        return (ret == 0) ? VORTEX_SUCCESS : VORTEX_ERROR_MEMORY;
    }
    case VORTEX_MEMCPY_DEVICE_TO_HOST: {
        int ret = vx_copy_from_dev(const_cast<void *>(src), buffer->vx_buf, offset, size);
        return (ret == 0) ? VORTEX_SUCCESS : VORTEX_ERROR_MEMORY;
    }
    default:
        return VORTEX_ERROR_INVALID;
    }
}

// ============================================================================
// Kernel Management
// ============================================================================

int vortex_kernel_load(vortex_device_h device, vortex_kernel_h *kernel, const void *binary_data, size_t binary_size) {
    if (!device || !kernel || !binary_data || binary_size == 0)
        return VORTEX_ERROR_INVALID;
    if (!device->initialized)
        return VORTEX_ERROR_INIT;

    // Allocate kernel structure
    *kernel = new vortex_kernel();
    if (!*kernel)
        return VORTEX_ERROR_MEMORY;

    (*kernel)->device = device;

    // Upload kernel binary to device
    int ret = vx_upload_kernel_bytes(device->vx_dev, binary_data, binary_size, &((*kernel)->kernel_buffer));
    if (ret != 0) {
        delete *kernel;
        *kernel = nullptr;
        return VORTEX_ERROR_KERNEL;
    }

    // Get kernel address
    ret = vx_mem_address((*kernel)->kernel_buffer, &((*kernel)->kernel_addr));
    if (ret != 0) {
        vx_mem_free((*kernel)->kernel_buffer);
        delete *kernel;
        *kernel = nullptr;
        return VORTEX_ERROR_KERNEL;
    }

    (*kernel)->args_buffer = nullptr;
    (*kernel)->args_addr = 0;

    if (device->debug_enabled) {
        std::cout << "Loaded kernel binary (" << binary_size << " bytes) at 0x" << std::hex << (*kernel)->kernel_addr
                  << std::dec << std::endl;
    }

    return VORTEX_SUCCESS;
}

int vortex_kernel_load_file(vortex_device_h device, vortex_kernel_h *kernel, const char *binary_path) {
    if (!device || !kernel || !binary_path)
        return VORTEX_ERROR_INVALID;

    // Read binary file
    int fd = open(binary_path, O_RDONLY);
    if (fd < 0) {
        std::cerr << "Failed to open kernel file: " << binary_path << std::endl;
        return VORTEX_ERROR_KERNEL;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return VORTEX_ERROR_KERNEL;
    }

    size_t file_size = st.st_size;
    std::vector<uint8_t> binary_data(file_size);

    ssize_t bytes_read = read(fd, binary_data.data(), file_size);
    close(fd);

    if (bytes_read != (ssize_t)file_size) {
        return VORTEX_ERROR_KERNEL;
    }

    return vortex_kernel_load(device, kernel, binary_data.data(), file_size);
}

int vortex_kernel_set_args(vortex_kernel_h kernel, const vortex_kernel_arg_t *args, uint32_t num_args) {
    if (!kernel || !args || num_args == 0)
        return VORTEX_ERROR_INVALID;

    // Store arguments
    kernel->args.clear();
    kernel->args.reserve(num_args);

    // Calculate total argument buffer size
    size_t total_size = 0;
    for (uint32_t i = 0; i < num_args; i++) {
        // Align each argument
        size_t aligned_size = (args[i].size + args[i].alignment - 1) & ~(args[i].alignment - 1);
        total_size += aligned_size;
        kernel->args.push_back(args[i]);
    }

    // Allocate argument buffer on device
    if (kernel->args_buffer != nullptr) {
        vx_mem_free(kernel->args_buffer);
    }

    int ret = vx_mem_alloc(kernel->device->vx_dev, total_size, VX_MEM_READ, &(kernel->args_buffer));
    if (ret != 0) {
        return VORTEX_ERROR_MEMORY;
    }

    // Get argument buffer address
    ret = vx_mem_address(kernel->args_buffer, &(kernel->args_addr));
    if (ret != 0) {
        vx_mem_free(kernel->args_buffer);
        kernel->args_buffer = nullptr;
        return VORTEX_ERROR_MEMORY;
    }

    // Pack arguments into buffer
    std::vector<uint8_t> arg_data(total_size);
    size_t offset = 0;

    for (uint32_t i = 0; i < num_args; i++) {
        // Align offset
        offset = (offset + args[i].alignment - 1) & ~(args[i].alignment - 1);

        if (args[i].type == VORTEX_ARG_VALUE) {
            // Copy value
            memcpy(arg_data.data() + offset, args[i].host_ptr, args[i].size);
        } else if (args[i].type == VORTEX_ARG_BUFFER) {
            // Copy buffer address
            vortex_buffer_h buf = static_cast<vortex_buffer_h>(args[i].host_ptr);
            uint64_t addr = buf->dev_addr;
            memcpy(arg_data.data() + offset, &addr, sizeof(uint64_t));
        }

        offset += args[i].size;
    }

    // Upload arguments to device
    ret = vx_copy_to_dev(kernel->args_buffer, arg_data.data(), 0, total_size);
    if (ret != 0) {
        return VORTEX_ERROR_MEMORY;
    }

    if (kernel->device->debug_enabled) {
        std::cout << "Set " << num_args << " kernel arguments (" << total_size << " bytes total)" << std::endl;
    }

    return VORTEX_SUCCESS;
}

int vortex_kernel_unload(vortex_kernel_h kernel) {
    if (!kernel)
        return VORTEX_ERROR_INVALID;

    if (kernel->kernel_buffer) {
        vx_mem_free(kernel->kernel_buffer);
    }
    if (kernel->args_buffer) {
        vx_mem_free(kernel->args_buffer);
    }

    delete kernel;
    return VORTEX_SUCCESS;
}

// ============================================================================
// Kernel Execution
// ============================================================================

int vortex_kernel_launch(vortex_device_h device, vortex_kernel_h kernel, const vortex_launch_params_t *params) {
    if (!device || !kernel || !params)
        return VORTEX_ERROR_INVALID;
    if (!device->initialized)
        return VORTEX_ERROR_INIT;

    // Note: Vortex handles threading internally via vx_spawn_threads() in the kernel
    // The grid/block parameters are passed as part of the kernel arguments
    // We don't need to write DCR registers - vx_start() handles everything

    if (device->debug_enabled) {
        std::cout << "Launching kernel:" << std::endl;
        std::cout << "  Grid: (" << params->grid_dim_x << ", " << params->grid_dim_y << ", " << params->grid_dim_z
                  << ")" << std::endl;
        std::cout << "  Block: (" << params->block_dim_x << ", " << params->block_dim_y << ", " << params->block_dim_z
                  << ")" << std::endl;
    }

    // Start execution
    int ret = vx_start(device->vx_dev, kernel->kernel_buffer, kernel->args_buffer);
    if (ret != 0) {
        std::cerr << "Failed to start kernel execution: " << ret << std::endl;
        return VORTEX_ERROR_LAUNCH;
    }

    // Wait for completion
    ret = vx_ready_wait(device->vx_dev, VX_MAX_TIMEOUT);
    if (ret != 0) {
        std::cerr << "Kernel execution timed out or failed" << std::endl;
        return VORTEX_ERROR_TIMEOUT;
    }

    return VORTEX_SUCCESS;
}

int vortex_device_wait(vortex_device_h device, uint64_t timeout_ms) {
    if (!device)
        return VORTEX_ERROR_INVALID;
    if (!device->initialized)
        return VORTEX_ERROR_INIT;

    int ret = vx_ready_wait(device->vx_dev, timeout_ms);
    if (ret != 0) {
        return VORTEX_ERROR_TIMEOUT;
    }

    return VORTEX_SUCCESS;
}

int vortex_device_synchronize(vortex_device_h device) {
    // Wait with maximum timeout
    return vortex_device_wait(device, VX_MAX_TIMEOUT);
}

// ============================================================================
// Performance and Debugging
// ============================================================================

int vortex_device_get_perf_counter(vortex_device_h device, const char *counter_name, uint64_t *value) {
    if (!device || !counter_name || !value)
        return VORTEX_ERROR_INVALID;

    // Check if we have this counter cached
    auto it = device->perf_counters.find(counter_name);
    if (it != device->perf_counters.end()) {
        *value = it->second;
        return VORTEX_SUCCESS;
    }

    return VORTEX_ERROR_INVALID;
}

int vortex_device_dump_perf(vortex_device_h device, const char *output_path) {
    if (!device)
        return VORTEX_ERROR_INVALID;

    FILE *f = stdout;
    if (output_path) {
        f = fopen(output_path, "w");
        if (!f)
            return VORTEX_ERROR_INVALID;
    }

    // Stub implementation: print basic performance info
    fprintf(f, "=== Vortex Performance Counters ===\n");
    fprintf(f, "Device: %p\n", device->vx_dev);
    fprintf(f, "Note: Full performance counters require Vortex SDK\n");

    // Print any cached counters we have
    for (const auto &counter : device->perf_counters) {
        fprintf(f, "%s: %" PRIu64 "\n", counter.first.c_str(), counter.second);
    }

    if (output_path) {
        fclose(f);
    }

    return VORTEX_SUCCESS;
}

int vortex_device_set_debug(vortex_device_h device, int enable) {
    if (!device)
        return VORTEX_ERROR_INVALID;

    device->debug_enabled = (enable != 0);
    return VORTEX_SUCCESS;
}

// ============================================================================
// CXL Type-2 device-side firmware helpers
// ============================================================================
//
// This section mirrors the hetGPU PACC firmware shape:
//   * host stages job arguments in a CXL-visible control window,
//   * host commits a small doorbell with a monotonically increasing seq,
//   * device polls/consumes the doorbell, performs the memory operation, and
//     writes both a control-window status and an optional CIRA completion line.
//
// The exported __vortex_* symbols are the device hooks emitted by CiraToLLVM
// for RISCV_VORTEX.  They are also valid host-side fallbacks when this runtime
// is linked without an actual Vortex firmware build.

namespace {

// Wire layout lives in cira_cxl_job.h — shared with the host submit path.
constexpr uint64_t VORTEX_CXL_JOB_MAGIC = CIRA_CXL_JOB_MAGIC;
constexpr uint32_t VORTEX_CXL_JOB_VERSION = CIRA_CXL_JOB_VERSION;
constexpr uint32_t VORTEX_CXL_COMPLETION_MAGIC = CIRA_CXL_COMPLETION_MAGIC;
constexpr uint32_t VORTEX_CXL_CACHELINE_SIZE = CIRA_CXL_CACHELINE_SIZE;

constexpr uint64_t VORTEX_CXL_DOORBELL_OFF = CIRA_CXL_DOORBELL_OFF;
constexpr uint64_t VORTEX_CXL_ARG_SLOT_BYTES = CIRA_CXL_ARG_SLOT_BYTES;
constexpr uint64_t VORTEX_CXL_COMPLETION_OFF = CIRA_CXL_STATUS_OFF;
constexpr uint64_t VORTEX_CXL_ARG_PAYLOAD_BYTES = CIRA_CXL_ARG_PAYLOAD_BYTES;

constexpr uint32_t VORTEX_CXL_JOB_NOP = CIRA_CXL_JOB_NOP;
constexpr uint32_t VORTEX_CXL_JOB_INSTALL_CACHELINE = CIRA_CXL_JOB_INSTALL_CACHELINE;
constexpr uint32_t VORTEX_CXL_JOB_PREFETCH_CHAIN = CIRA_CXL_JOB_PREFETCH_CHAIN;
constexpr uint32_t VORTEX_CXL_JOB_STREAM_PREFETCH = CIRA_CXL_JOB_STREAM_PREFETCH;
constexpr uint32_t VORTEX_CXL_JOB_CALL = CIRA_CXL_JOB_CALL;

constexpr uint32_t VORTEX_CXL_STATUS_SUCCESS = CIRA_CXL_STATUS_SUCCESS;
constexpr uint32_t VORTEX_CXL_STATUS_RUNNING = CIRA_CXL_STATUS_RUNNING;
constexpr uint32_t VORTEX_CXL_STATUS_BAD_VERSION = CIRA_CXL_STATUS_BAD_VERSION;
constexpr uint32_t VORTEX_CXL_STATUS_BAD_ARGS = CIRA_CXL_STATUS_BAD_ARGS;
constexpr uint32_t VORTEX_CXL_STATUS_BAD_JOB = CIRA_CXL_STATUS_BAD_JOB;

using VortexCxlDoorbell = cira_cxl_doorbell_t;
using VortexCxlArgSlotHeader = cira_cxl_arg_slot_t;
using VortexCxlHostStatus = cira_cxl_host_status_t;
using VortexCxlCompletion = cira_cxl_completion_t;
using VortexCxlInstallCachelineJob = cira_cxl_install_cacheline_job_t;
using VortexCxlPrefetchChainJob = cira_cxl_prefetch_chain_job_t;
using VortexCxlStreamPrefetchJob = cira_cxl_stream_prefetch_job_t;
using VortexCxlCallJob = cira_cxl_call_job_t;

using VortexCxlOffloadFn = void (*)(void **operands, uint32_t num_operands, void *completion);

static inline void vortex_cxl_fence() {
#if defined(__riscv)
    __asm__ volatile("fence iorw, iorw" ::: "memory");
#else
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
}

static inline void vortex_cxl_wait_for_interrupt() {
#if defined(__riscv)
    __asm__ volatile("wfi" ::: "memory");
#else
    usleep(50);
#endif
}

static inline uint64_t vortex_cxl_cycles() {
#if defined(__riscv)
    uint64_t value;
    __asm__ volatile("rdcycle %0" : "=r"(value));
    return value;
#else
    return 0;
#endif
}

static inline bool vortex_cxl_valid_magic(uint64_t magic) { return cira_cxl_valid_magic(magic) != 0; }

static inline uintptr_t vortex_cxl_align_down(uintptr_t value, uintptr_t align) { return value & ~(align - 1); }

static inline uintptr_t vortex_cxl_align_up(uintptr_t value, uintptr_t align) {
    return (value + align - 1) & ~(align - 1);
}

static void vortex_cxl_copy_bytes(volatile uint8_t *dst, const volatile uint8_t *src, uint64_t size) {
    uint64_t off = 0;
    for (; off + sizeof(uint64_t) <= size; off += sizeof(uint64_t)) {
        uint64_t value = *reinterpret_cast<const volatile uint64_t *>(src + off);
        *reinterpret_cast<volatile uint64_t *>(dst + off) = value;
    }
    for (; off < size; ++off) {
        dst[off] = src[off];
    }
}

static void vortex_cxl_store_bytes(volatile void *dst, const void *src, uint64_t size) {
    auto *out = static_cast<volatile uint8_t *>(dst);
    auto *in = static_cast<const uint8_t *>(src);
    for (uint64_t off = 0; off < size; ++off) {
        out[off] = in[off];
    }
}

static uint64_t vortex_cxl_install_range(void *addr, uint64_t size) {
    if (!addr || size == 0)
        return 0;

    uintptr_t begin = vortex_cxl_align_down(reinterpret_cast<uintptr_t>(addr), VORTEX_CXL_CACHELINE_SIZE);
    uintptr_t end = vortex_cxl_align_up(reinterpret_cast<uintptr_t>(addr) + size, VORTEX_CXL_CACHELINE_SIZE);
    uint64_t installed = 0;

    for (uintptr_t line = begin; line < end; line += VORTEX_CXL_CACHELINE_SIZE) {
        auto *ptr = reinterpret_cast<volatile uint8_t *>(line);

        // Read the full line, then write it back unchanged.  On a CXL Type-2
        // DCOH path the store takes device ownership and makes the line visible
        // to the host coherency domain without changing payload bytes.
        uint64_t words[VORTEX_CXL_CACHELINE_SIZE / sizeof(uint64_t)];
        for (uint32_t i = 0; i < VORTEX_CXL_CACHELINE_SIZE / sizeof(uint64_t); ++i) {
            words[i] = *reinterpret_cast<volatile uint64_t *>(ptr + i * sizeof(uint64_t));
        }
        vortex_cxl_fence();
        for (uint32_t i = 0; i < VORTEX_CXL_CACHELINE_SIZE / sizeof(uint64_t); ++i) {
            *reinterpret_cast<volatile uint64_t *>(ptr + i * sizeof(uint64_t)) = words[i];
        }
        ++installed;
    }

    vortex_cxl_fence();
    return installed;
}

static uint64_t vortex_cxl_prefetch_chain_impl(void *start_node, uint64_t next_ptr_offset, uint64_t depth) {
    uint64_t visited = 0;
    auto *node = reinterpret_cast<volatile uint8_t *>(start_node);

    while (node && visited < depth) {
        vortex_cxl_install_range(const_cast<uint8_t *>(reinterpret_cast<const volatile uint8_t *>(node)),
                                 VORTEX_CXL_CACHELINE_SIZE);

        auto *next_slot = reinterpret_cast<volatile uintptr_t *>(
            const_cast<uint8_t *>(reinterpret_cast<const volatile uint8_t *>(node)) + next_ptr_offset);
        node = reinterpret_cast<volatile uint8_t *>(*next_slot);
        ++visited;
    }

    return visited;
}

static void vortex_cxl_write_completion(uint64_t completion_addr, uint32_t status, uint64_t result,
                                        uint64_t start_cycles) {
    if (!completion_addr)
        return;

    auto *completion = reinterpret_cast<volatile VortexCxlCompletion *>(completion_addr);
    completion->result = result;
    completion->cycles = vortex_cxl_cycles() - start_cycles;
    completion->timestamp = vortex_cxl_cycles();
    completion->status = status;
    vortex_cxl_fence();
    completion->magic = VORTEX_CXL_COMPLETION_MAGIC;
    vortex_cxl_fence();
}

static void vortex_cxl_mirror_status(volatile uint8_t *control, uint32_t job_id, uint64_t seq, uint32_t status) {
    if (!control)
        return;
    auto *host = reinterpret_cast<volatile VortexCxlHostStatus *>(control + VORTEX_CXL_COMPLETION_OFF);
    host->magic = VORTEX_CXL_JOB_MAGIC;
    host->version = VORTEX_CXL_JOB_VERSION;
    host->job_id = job_id;
    host->status = status;
    host->seq = seq;
    vortex_cxl_fence();
}

static volatile VortexCxlArgSlotHeader *vortex_cxl_arg_slot(volatile uint8_t *control, uint32_t job_id) {
    if (!control)
        return nullptr;
    if (job_id > VORTEX_CXL_JOB_CALL)
        return nullptr;

    return reinterpret_cast<volatile VortexCxlArgSlotHeader *>(control + cira_cxl_arg_slot_off(job_id));
}

static const void *vortex_cxl_arg_payload(const volatile VortexCxlArgSlotHeader *slot) {
    return reinterpret_cast<const void *>(reinterpret_cast<uintptr_t>(slot) + sizeof(VortexCxlArgSlotHeader));
}

static volatile uint8_t *vortex_cxl_mutable_arg_payload(volatile VortexCxlArgSlotHeader *slot) {
    return reinterpret_cast<volatile uint8_t *>(
        reinterpret_cast<uintptr_t>(const_cast<VortexCxlArgSlotHeader *>(slot)) + sizeof(VortexCxlArgSlotHeader));
}

template <typename T>
static const T *vortex_cxl_checked_arg(const volatile VortexCxlArgSlotHeader *slot, const VortexCxlDoorbell &doorbell) {
    if (!slot || !vortex_cxl_valid_magic(slot->magic) || slot->version != VORTEX_CXL_JOB_VERSION ||
        slot->job_id != doorbell.job_id || slot->seq != doorbell.seq || slot->arg_len < sizeof(T)) {
        return nullptr;
    }
    return reinterpret_cast<const T *>(vortex_cxl_arg_payload(slot));
}

static uint32_t vortex_cxl_run_install(const VortexCxlInstallCachelineJob *job, uint64_t start_cycles) {
    if (!job || !job->addr || job->size == 0) {
        return VORTEX_CXL_STATUS_BAD_ARGS;
    }
    uint64_t lines = vortex_cxl_install_range(reinterpret_cast<void *>(job->addr), job->size);
    vortex_cxl_write_completion(job->completion_addr, VORTEX_CXL_STATUS_SUCCESS, lines, start_cycles);
    return VORTEX_CXL_STATUS_SUCCESS;
}

static uint32_t vortex_cxl_run_prefetch_chain(const VortexCxlPrefetchChainJob *job, uint64_t start_cycles) {
    if (!job || !job->start_node_addr) {
        return VORTEX_CXL_STATUS_BAD_ARGS;
    }

    uint64_t visited = 0;
    auto *node = reinterpret_cast<volatile uint8_t *>(job->start_node_addr);
    auto *dst = reinterpret_cast<volatile uint8_t *>(job->buf_addr);

    while (node && visited < job->depth) {
        vortex_cxl_install_range(const_cast<uint8_t *>(reinterpret_cast<const volatile uint8_t *>(node)),
                                 VORTEX_CXL_CACHELINE_SIZE);

        if (dst && job->data_size != 0) {
            const volatile uint8_t *src = node + job->data_offset;
            volatile uint8_t *out = dst + visited * job->data_size;
            vortex_cxl_copy_bytes(out, src, job->data_size);
        }

        auto *next_slot = reinterpret_cast<volatile uintptr_t *>(
            const_cast<uint8_t *>(reinterpret_cast<const volatile uint8_t *>(node)) + job->next_ptr_offset);
        node = reinterpret_cast<volatile uint8_t *>(*next_slot);
        ++visited;
    }

    vortex_cxl_write_completion(job->completion_addr, VORTEX_CXL_STATUS_SUCCESS, visited, start_cycles);
    return VORTEX_CXL_STATUS_SUCCESS;
}

static uint32_t vortex_cxl_run_stream_prefetch(const VortexCxlStreamPrefetchJob *job, uint64_t start_cycles) {
    if (!job || !job->base_addr || job->count == 0 || job->elem_size == 0) {
        return VORTEX_CXL_STATUS_BAD_ARGS;
    }

    auto *base = reinterpret_cast<volatile uint8_t *>(job->base_addr);
    auto *dst = reinterpret_cast<volatile uint8_t *>(job->buf_addr);
    uint64_t stride = job->stride ? job->stride : job->elem_size;

    for (uint64_t i = 0; i < job->count; ++i) {
        volatile uint8_t *src = base + i * stride;
        vortex_cxl_install_range(const_cast<uint8_t *>(reinterpret_cast<const volatile uint8_t *>(src)),
                                 job->elem_size);
        if (dst) {
            vortex_cxl_copy_bytes(dst + i * job->elem_size, src, job->elem_size);
        }
    }

    vortex_cxl_write_completion(job->completion_addr, VORTEX_CXL_STATUS_SUCCESS, job->count, start_cycles);
    return VORTEX_CXL_STATUS_SUCCESS;
}

static uint32_t vortex_cxl_run_call(const VortexCxlCallJob *job, uint64_t start_cycles) {
    if (!job || !job->func_addr) {
        return VORTEX_CXL_STATUS_BAD_ARGS;
    }

    auto fn = reinterpret_cast<VortexCxlOffloadFn>(job->func_addr);
    auto operands = reinterpret_cast<void **>(job->operands_addr);
    fn(operands, job->num_operands, reinterpret_cast<void *>(job->completion_addr));

    // If the callee did not use the CIRA completion line itself, make the
    // generic call complete successfully.
    vortex_cxl_write_completion(job->completion_addr, VORTEX_CXL_STATUS_SUCCESS, 0, start_cycles);
    return VORTEX_CXL_STATUS_SUCCESS;
}

static uint32_t vortex_cxl_run_job(volatile uint8_t *control, const VortexCxlDoorbell &doorbell) {
    if (doorbell.version != VORTEX_CXL_JOB_VERSION) {
        return VORTEX_CXL_STATUS_BAD_VERSION;
    }
    if (doorbell.job_id == VORTEX_CXL_JOB_NOP) {
        return VORTEX_CXL_STATUS_SUCCESS;
    }

    uint64_t start_cycles = vortex_cxl_cycles();
    auto *slot = vortex_cxl_arg_slot(control, doorbell.job_id);

    switch (doorbell.job_id) {
    case VORTEX_CXL_JOB_INSTALL_CACHELINE:
        return vortex_cxl_run_install(vortex_cxl_checked_arg<VortexCxlInstallCachelineJob>(slot, doorbell),
                                      start_cycles);
    case VORTEX_CXL_JOB_PREFETCH_CHAIN:
        return vortex_cxl_run_prefetch_chain(vortex_cxl_checked_arg<VortexCxlPrefetchChainJob>(slot, doorbell),
                                             start_cycles);
    case VORTEX_CXL_JOB_STREAM_PREFETCH:
        return vortex_cxl_run_stream_prefetch(vortex_cxl_checked_arg<VortexCxlStreamPrefetchJob>(slot, doorbell),
                                              start_cycles);
    case VORTEX_CXL_JOB_CALL:
        return vortex_cxl_run_call(vortex_cxl_checked_arg<VortexCxlCallJob>(slot, doorbell), start_cycles);
    default:
        return VORTEX_CXL_STATUS_BAD_JOB;
    }
}

} // namespace

extern "C" int vortex_cxl_submit_job_mmio(void *control_window, uint32_t job_id, const void *arg_data, uint64_t arg_len,
                                          uint64_t seq, uint32_t flags) {
    if (!control_window)
        return VORTEX_ERROR_INVALID;
    if (job_id == VORTEX_CXL_JOB_NOP || job_id > VORTEX_CXL_JOB_CALL)
        return VORTEX_ERROR_INVALID;
    if (arg_len > VORTEX_CXL_ARG_PAYLOAD_BYTES)
        return VORTEX_ERROR_INVALID;
    if (arg_len != 0 && !arg_data)
        return VORTEX_ERROR_INVALID;

    static std::atomic<uint64_t> next_seq{1};
    uint64_t commit_seq = seq ? seq : next_seq.fetch_add(1, std::memory_order_relaxed);
    if (commit_seq == 0) {
        commit_seq = next_seq.fetch_add(1, std::memory_order_relaxed);
    }

    auto *control = reinterpret_cast<volatile uint8_t *>(control_window);
    auto *slot = vortex_cxl_arg_slot(control, job_id);
    if (!slot)
        return VORTEX_ERROR_INVALID;

    if (arg_len != 0) {
        vortex_cxl_store_bytes(vortex_cxl_mutable_arg_payload(slot), arg_data, arg_len);
    }

    VortexCxlArgSlotHeader header = {
        VORTEX_CXL_JOB_MAGIC, VORTEX_CXL_JOB_VERSION, job_id, commit_seq, arg_len,
    };
    vortex_cxl_store_bytes(slot, &header, sizeof(header));
    vortex_cxl_fence();

    VortexCxlDoorbell doorbell = {
        VORTEX_CXL_JOB_MAGIC, VORTEX_CXL_JOB_VERSION, job_id, flags, 0, commit_seq,
    };
    vortex_cxl_store_bytes(control + VORTEX_CXL_DOORBELL_OFF, &doorbell, sizeof(doorbell));
    vortex_cxl_fence();

    return VORTEX_SUCCESS;
}

extern "C" int vortex_cxl_submit_call_mmio(void *control_window, uint64_t seq, void *func_ptr, void **operands,
                                           uint32_t num_operands, void *completion_ptr) {
    if (!func_ptr)
        return VORTEX_ERROR_INVALID;
    if (num_operands != 0 && !operands)
        return VORTEX_ERROR_INVALID;

    VortexCxlCallJob job = {
        reinterpret_cast<uint64_t>(func_ptr),
        reinterpret_cast<uint64_t>(operands),
        reinterpret_cast<uint64_t>(completion_ptr),
        num_operands,
        0,
    };

    return vortex_cxl_submit_job_mmio(control_window, VORTEX_CXL_JOB_CALL, &job, sizeof(job), seq, 0);
}

extern "C" void __vortex_install_cacheline(void *addr, uint64_t size, int cache_level) {
    (void)cache_level;
    vortex_cxl_install_range(addr, size);
}

extern "C" void __vortex_prefetch_chain(void *start_node, uint64_t offset, uint64_t depth) {
    vortex_cxl_prefetch_chain_impl(start_node, offset, depth);
}

extern "C" void __vortex_prefetch_chain_kernel(void *stream, uint64_t depth) {
    if (!stream || depth == 0)
        return;

    // Fast path for a descriptor-style stream:
    //   [0] magic, [8] current node, [16] next pointer offset.
    // If no descriptor magic is present, treat stream as the starting node and
    // assume the next pointer is the first word of each node.
    struct StreamDesc {
        uint64_t magic;
        uint64_t current;
        uint64_t next_ptr_offset;
    };

    auto *desc = reinterpret_cast<volatile StreamDesc *>(stream);
    if (vortex_cxl_valid_magic(desc->magic)) {
        vortex_cxl_prefetch_chain_impl(reinterpret_cast<void *>(desc->current), desc->next_ptr_offset, depth);
        return;
    }

    vortex_cxl_prefetch_chain_impl(stream, 0, depth);
}

extern "C" uint32_t vortex_cxl_firmware_service_once(void *control_window) {
    if (!control_window)
        return VORTEX_CXL_STATUS_BAD_ARGS;

    auto *control = reinterpret_cast<volatile uint8_t *>(control_window);
    auto *doorbell = reinterpret_cast<volatile VortexCxlDoorbell *>(control + VORTEX_CXL_DOORBELL_OFF);

    vortex_cxl_fence();
    if (!vortex_cxl_valid_magic(doorbell->magic)) {
        return VORTEX_CXL_STATUS_BAD_ARGS;
    }

    VortexCxlDoorbell local = {
        doorbell->magic, doorbell->version, doorbell->job_id, doorbell->flags, doorbell->status, doorbell->seq,
    };

    doorbell->status = VORTEX_CXL_STATUS_RUNNING;
    vortex_cxl_mirror_status(control, local.job_id, local.seq, VORTEX_CXL_STATUS_RUNNING);

    uint32_t status = vortex_cxl_run_job(control, local);

    doorbell->status = status;
    vortex_cxl_mirror_status(control, local.job_id, local.seq, status);
    return status;
}

extern "C" void vortex_cxl_firmware_loop(void *control_window) {
    if (!control_window)
        return;

    auto *control = reinterpret_cast<volatile uint8_t *>(control_window);
    auto *doorbell = reinterpret_cast<volatile VortexCxlDoorbell *>(control + VORTEX_CXL_DOORBELL_OFF);
    uint64_t last_seq = 0;

    for (;;) {
        vortex_cxl_fence();
        if (vortex_cxl_valid_magic(doorbell->magic) && doorbell->seq != last_seq) {
            last_seq = doorbell->seq;
            (void)vortex_cxl_firmware_service_once(
                const_cast<uint8_t *>(reinterpret_cast<const volatile uint8_t *>(control)));
        } else {
            vortex_cxl_wait_for_interrupt();
        }
    }
}
