// CIRA CXL Type-2 MMIO window — see cira_mmio.h.

#include "cira_mmio.h"
#include "cira_mwait.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <mutex>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

struct cira_mmio_window {
    volatile uint8_t *base = nullptr;
    uint64_t size = 0;
    int fd = -1;
    bool owns_mapping = false;
    bool emulated = false;
    bool read_only = false;
    std::atomic<uint64_t> next_seq{1};
};

namespace {

std::mutex g_default_mutex;
cira_mmio_window_t *g_default_window = nullptr;
bool g_default_probed = false;
std::atomic<void *> g_device_func{nullptr};

bool parse_env_u64(const char *primary, const char *legacy, uint64_t &out) {
    for (const char *name : {primary, legacy}) {
        if (!name)
            continue;
        const char *text = std::getenv(name);
        if (!text || !*text)
            continue;
        char *end = nullptr;
        errno = 0;
        unsigned long long parsed = std::strtoull(text, &end, 0);
        if (errno != 0 || end == text || (end && *end != '\0'))
            continue;
        out = (uint64_t)parsed;
        return true;
    }
    return false;
}

const char *env_str(const char *primary, const char *legacy) {
    const char *v = std::getenv(primary);
    if (v && *v)
        return v;
    v = legacy ? std::getenv(legacy) : nullptr;
    return (v && *v) ? v : nullptr;
}

bool env_flag(const char *name) {
    const char *v = std::getenv(name);
    return v && *v && std::strcmp(v, "0") != 0;
}

bool in_range(const cira_mmio_window_t *w, uint64_t offset, uint64_t len) {
    return w && w->base && len <= w->size && offset <= w->size - len;
}

// Byte-granular copy through volatile pointers: MMIO must not be memcpy'd,
// the compiler is free to split or reorder those.
void store_bytes(volatile uint8_t *dst, const void *src, uint64_t len) {
    auto *in = static_cast<const uint8_t *>(src);
    uint64_t off = 0;
    for (; off + sizeof(uint64_t) <= len; off += sizeof(uint64_t)) {
        uint64_t word;
        std::memcpy(&word, in + off, sizeof(word));
        *reinterpret_cast<volatile uint64_t *>(dst + off) = word;
    }
    for (; off < len; ++off)
        dst[off] = in[off];
}

void load_bytes(void *dst, const volatile uint8_t *src, uint64_t len) {
    auto *out = static_cast<uint8_t *>(dst);
    uint64_t off = 0;
    for (; off + sizeof(uint64_t) <= len; off += sizeof(uint64_t)) {
        uint64_t word = *reinterpret_cast<const volatile uint64_t *>(src + off);
        std::memcpy(out + off, &word, sizeof(word));
    }
    for (; off < len; ++off)
        out[off] = src[off];
}

} // namespace

extern "C" {

// ---------------------------------------------------------------------------
// Window lifetime
// ---------------------------------------------------------------------------

int cira_mmio_open(cira_mmio_window_t **out, const cira_mmio_config_t *cfg) {
    if (!out || !cfg)
        return CIRA_MMIO_EINVAL;
    *out = nullptr;

    uint64_t size = cfg->size ? cfg->size : CIRA_CXL_CONTROL_BYTES;
    if (size < CIRA_CXL_CONTROL_BYTES)
        return CIRA_MMIO_ERANGE;

    auto *w = new (std::nothrow) cira_mmio_window();
    if (!w)
        return CIRA_MMIO_EIO;
    w->size = size;
    w->read_only = cfg->read_only;

    if (cfg->fixed_addr) {
        w->base = reinterpret_cast<volatile uint8_t *>(cfg->fixed_addr);
    } else if (cfg->emulate || !cfg->path) {
        if (!cfg->emulate) {
            delete w;
            return CIRA_MMIO_ENODEV;
        }
        void *map = mmap(nullptr, (size_t)size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (map == MAP_FAILED) {
            delete w;
            return CIRA_MMIO_EIO;
        }
        std::memset(map, 0, (size_t)size);
        w->base = static_cast<volatile uint8_t *>(map);
        w->owns_mapping = true;
        w->emulated = true;
    } else {
        int flags = cfg->read_only ? O_RDONLY : O_RDWR;
        int fd = open(cfg->path, flags | O_SYNC | O_CLOEXEC);
        if (fd < 0) {
            std::fprintf(stderr, "[cira-mmio] open %s failed: %s\n", cfg->path, std::strerror(errno));
            delete w;
            return CIRA_MMIO_ENODEV;
        }
        int prot = cfg->read_only ? PROT_READ : (PROT_READ | PROT_WRITE);
        void *map = mmap(nullptr, (size_t)size, prot, MAP_SHARED, fd, (off_t)cfg->offset);
        if (map == MAP_FAILED) {
            std::fprintf(stderr, "[cira-mmio] mmap %s offset=0x%llx size=0x%llx failed: %s\n", cfg->path,
                         (unsigned long long)cfg->offset, (unsigned long long)size, std::strerror(errno));
            close(fd);
            delete w;
            return CIRA_MMIO_EIO;
        }
        w->fd = fd;
        w->base = static_cast<volatile uint8_t *>(map);
        w->owns_mapping = true;
    }

    *out = w;
    return CIRA_MMIO_OK;
}

int cira_mmio_open_env(cira_mmio_window_t **out) {
    if (!out)
        return CIRA_MMIO_EINVAL;

    cira_mmio_config_t cfg = {};
    uint64_t size = CIRA_CXL_CONTROL_BYTES;
    parse_env_u64("CIRA_CXL_MMIO_SIZE", "CIRA_TYPE2_MMIO_SIZE", size);
    cfg.size = size;
    parse_env_u64("CIRA_CXL_MMIO_OFFSET", "CIRA_TYPE2_MMIO_OFFSET", cfg.offset);

    uint64_t addr = 0;
    if (parse_env_u64("CIRA_CXL_MMIO_ADDR", "CIRA_TYPE2_MMIO_ADDR", addr) && addr) {
        cfg.fixed_addr = (uintptr_t)addr;
        return cira_mmio_open(out, &cfg);
    }

    cfg.path = env_str("CIRA_CXL_MMIO_PATH", "CIRA_TYPE2_MMIO_PATH");
    cfg.emulate = env_flag("CIRA_CXL_MMIO_EMULATE");
    if (!cfg.path && !cfg.emulate)
        return CIRA_MMIO_ENODEV;
    if (cfg.emulate)
        cfg.path = nullptr;
    return cira_mmio_open(out, &cfg);
}

void cira_mmio_close(cira_mmio_window_t *window) {
    if (!window)
        return;
    if (window->owns_mapping && window->base) {
        munmap(const_cast<uint8_t *>(window->base), (size_t)window->size);
    }
    if (window->fd >= 0)
        close(window->fd);
    delete window;
}

cira_mmio_window_t *cira_mmio_default(void) {
    std::lock_guard<std::mutex> lock(g_default_mutex);
    if (g_default_window || g_default_probed)
        return g_default_window;
    g_default_probed = true;
    cira_mmio_window_t *w = nullptr;
    if (cira_mmio_open_env(&w) == CIRA_MMIO_OK)
        g_default_window = w;
    return g_default_window;
}

void cira_mmio_set_default(cira_mmio_window_t *window) {
    std::lock_guard<std::mutex> lock(g_default_mutex);
    if (g_default_window && g_default_window != window) {
        cira_mmio_close(g_default_window);
    }
    g_default_window = window;
    g_default_probed = window != nullptr;
}

void *cira_mmio_base(const cira_mmio_window_t *w) { return w ? const_cast<uint8_t *>(w->base) : nullptr; }
uint64_t cira_mmio_size(const cira_mmio_window_t *w) { return w ? w->size : 0; }
bool cira_mmio_is_emulated(const cira_mmio_window_t *w) { return w && w->emulated; }

// ---------------------------------------------------------------------------
// Register access
// ---------------------------------------------------------------------------

void cira_mmio_fence(void) { __atomic_thread_fence(__ATOMIC_SEQ_CST); }

uint32_t cira_mmio_read32(const cira_mmio_window_t *w, uint64_t offset) {
    if (!in_range(w, offset, sizeof(uint32_t)))
        return 0;
    return *reinterpret_cast<const volatile uint32_t *>(w->base + offset);
}

uint64_t cira_mmio_read64(const cira_mmio_window_t *w, uint64_t offset) {
    if (!in_range(w, offset, sizeof(uint64_t)))
        return 0;
    return *reinterpret_cast<const volatile uint64_t *>(w->base + offset);
}

int cira_mmio_write32(cira_mmio_window_t *w, uint64_t offset, uint32_t value) {
    if (w && w->read_only)
        return CIRA_MMIO_EINVAL;
    if (!in_range(w, offset, sizeof(value)))
        return CIRA_MMIO_ERANGE;
    *reinterpret_cast<volatile uint32_t *>(w->base + offset) = value;
    return CIRA_MMIO_OK;
}

int cira_mmio_write64(cira_mmio_window_t *w, uint64_t offset, uint64_t value) {
    if (w && w->read_only)
        return CIRA_MMIO_EINVAL;
    if (!in_range(w, offset, sizeof(value)))
        return CIRA_MMIO_ERANGE;
    *reinterpret_cast<volatile uint64_t *>(w->base + offset) = value;
    return CIRA_MMIO_OK;
}

int cira_mmio_write_block(cira_mmio_window_t *w, uint64_t offset, const void *src, uint64_t len) {
    if (w && w->read_only)
        return CIRA_MMIO_EINVAL;
    if (!src && len)
        return CIRA_MMIO_EINVAL;
    if (!in_range(w, offset, len))
        return CIRA_MMIO_ERANGE;
    store_bytes(w->base + offset, src, len);
    return CIRA_MMIO_OK;
}

int cira_mmio_read_block(const cira_mmio_window_t *w, uint64_t offset, void *dst, uint64_t len) {
    if (!dst && len)
        return CIRA_MMIO_EINVAL;
    if (!in_range(w, offset, len))
        return CIRA_MMIO_ERANGE;
    load_bytes(dst, w->base + offset, len);
    return CIRA_MMIO_OK;
}

// ---------------------------------------------------------------------------
// Completion lines
// ---------------------------------------------------------------------------

void cira_mmio_arm_completion(void *completion) {
    if (!completion)
        return;
    std::memset(completion, 0, CIRA_CXL_CACHELINE_SIZE);
    __atomic_thread_fence(__ATOMIC_RELEASE);
}

void cira_mmio_signal_completion(void *completion, uint32_t status, uint64_t result) {
    if (!completion)
        return;
    auto *line = static_cast<volatile cira_cxl_completion_t *>(completion);
    line->status = status;
    line->result = result;
    line->cycles = 0;
    line->timestamp = 0;
    // magic is published last; the acquire side of the wait pairs with this.
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_store_n(&line->magic, CIRA_CXL_COMPLETION_MAGIC, __ATOMIC_RELAXED);
}

int cira_mmio_wait_completion(void *completion, uint64_t timeout_ns) {
    if (!completion)
        return CIRA_MMIO_EINVAL;
    auto *magic = reinterpret_cast<volatile uint32_t *>(completion);
    return cira_wait_u32(magic, CIRA_CXL_COMPLETION_MAGIC, timeout_ns) ? CIRA_MMIO_OK : CIRA_MMIO_ETIMEDOUT;
}

uint64_t cira_mmio_default_timeout_ns(void) {
    uint64_t ns = 0;
    parse_env_u64("CIRA_CXL_MMIO_TIMEOUT_NS", nullptr, ns);
    return ns;
}

void *cira_mmio_device_func(void) {
    void *forced = g_device_func.load(std::memory_order_acquire);
    if (forced)
        return forced;
    uint64_t addr = 0;
    if (parse_env_u64("CIRA_CXL_DEVICE_FUNC_ADDR", "CIRA_TYPE2_DEVICE_FUNC_ADDR", addr) && addr) {
        return reinterpret_cast<void *>((uintptr_t)addr);
    }
    return nullptr;
}

void cira_mmio_set_device_func(void *func) { g_device_func.store(func, std::memory_order_release); }

// ---------------------------------------------------------------------------
// Job submission
// ---------------------------------------------------------------------------

int cira_mmio_submit_job(cira_mmio_window_t *w, uint32_t job_id, const void *args, uint64_t arg_len, uint32_t flags,
                         uint64_t *out_seq) {
    if (!w || !w->base)
        return CIRA_MMIO_ENODEV;
    if (w->read_only)
        return CIRA_MMIO_EINVAL;
    if (job_id == CIRA_CXL_JOB_NOP || job_id > CIRA_CXL_JOB_MAX)
        return CIRA_MMIO_EINVAL;
    if (arg_len > CIRA_CXL_ARG_PAYLOAD_BYTES)
        return CIRA_MMIO_ERANGE;
    if (arg_len && !args)
        return CIRA_MMIO_EINVAL;

    const uint64_t slot_off = cira_cxl_arg_slot_off(job_id);
    if (!in_range(w, slot_off, CIRA_CXL_ARG_SLOT_BYTES) ||
        !in_range(w, CIRA_CXL_DOORBELL_OFF, sizeof(cira_cxl_doorbell_t))) {
        return CIRA_MMIO_ERANGE;
    }

    uint64_t seq = w->next_seq.fetch_add(1, std::memory_order_relaxed);
    if (seq == 0)
        seq = w->next_seq.fetch_add(1, std::memory_order_relaxed);

    // 1. payload, 2. slot header, 3. doorbell — each step fenced so the device
    //    can never observe a doorbell that points at a half-written argument.
    if (arg_len) {
        store_bytes(w->base + slot_off + sizeof(cira_cxl_arg_slot_t), args, arg_len);
        cira_mmio_fence();
    }

    cira_cxl_arg_slot_t header = {CIRA_CXL_JOB_MAGIC, CIRA_CXL_JOB_VERSION, job_id, seq, arg_len};
    store_bytes(w->base + slot_off, &header, sizeof(header));
    cira_mmio_fence();

    cira_cxl_doorbell_t doorbell = {
        CIRA_CXL_JOB_MAGIC, CIRA_CXL_JOB_VERSION, job_id, flags, CIRA_CXL_STATUS_RUNNING, seq};
    store_bytes(w->base + CIRA_CXL_DOORBELL_OFF, &doorbell, sizeof(doorbell));
    cira_mmio_fence();

    if (out_seq)
        *out_seq = seq;
    return CIRA_MMIO_OK;
}

int cira_mmio_submit_call(cira_mmio_window_t *w, void *func, void **operands, uint32_t num_operands, void *completion,
                          uint64_t *out_seq) {
    if (num_operands && !operands)
        return CIRA_MMIO_EINVAL;

    void *target = func ? func : cira_mmio_device_func();
    if (!target)
        return CIRA_MMIO_ENODEV;

    cira_cxl_call_job_t job = {
        (uint64_t)(uintptr_t)target, (uint64_t)(uintptr_t)operands, (uint64_t)(uintptr_t)completion, num_operands, 0,
    };
    return cira_mmio_submit_job(w, CIRA_CXL_JOB_CALL, &job, sizeof(job), 0, out_seq);
}

int cira_mmio_wait_seq(const cira_mmio_window_t *w, uint64_t seq, uint64_t timeout_ns) {
    if (!w || !w->base)
        return CIRA_MMIO_ENODEV;
    if (!in_range(w, CIRA_CXL_STATUS_OFF, sizeof(cira_cxl_host_status_t)))
        return CIRA_MMIO_ERANGE;

    auto *status = reinterpret_cast<const volatile cira_cxl_host_status_t *>(w->base + CIRA_CXL_STATUS_OFF);

    // The device publishes seq last, so polling it is enough to order the read
    // of `status`. There is no 64-bit monitor primitive, so watch the low half
    // and confirm the full value.
    auto *seq_lo = reinterpret_cast<const volatile uint32_t *>(&status->seq);
    for (;;) {
        uint64_t observed = status->seq;
        if (observed >= seq) {
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            return status->status == CIRA_CXL_STATUS_SUCCESS ? CIRA_MMIO_OK : CIRA_MMIO_EIO;
        }
        if (!cira_wait_u32_ne(seq_lo, (uint32_t)observed, timeout_ns)) {
            return CIRA_MMIO_ETIMEDOUT;
        }
    }
}

} // extern "C"
