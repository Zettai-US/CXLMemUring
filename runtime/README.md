# Cira Runtime System Design

## Overview
The Cira runtime provides the execution engine for offloaded graph processing operations on CXL memory systems.

## Architecture Components

### 1. Offload Engine
Manages hardware offload operations for remote memory access.

```cpp
class OffloadEngine {
public:
    // Prefetch edge data from remote memory
    void* loadEdge(RemoteMemRef* edge_ptr, size_t index, size_t prefetch_distance);

    // Prefetch node data based on edge element
    void* loadNode(void* edge_element, const char* field_name, size_t prefetch_distance);

    // Get physical address of offloaded data
    uintptr_t getPhysicalAddr(const char* field_name, void* node_data);

    // Evict edge from cache
    void evictEdge(RemoteMemRef* edge_ptr, size_t index);
};
```

### 2. Memory Management
Handles allocation and management of remote memory regions.

```cpp
class RemoteMemoryManager {
    // Allocate remote memory region
    RemoteMemRef* allocateRemote(size_t size, MemoryTier tier);

    // Map remote memory to local address space
    void* mapRemote(RemoteMemRef* ref);

    // Manage memory tiers (local DRAM, CXL, far memory)
    void setMemoryTier(RemoteMemRef* ref, MemoryTier tier);
};
```

### 3. Prefetch Controller
Optimizes data prefetching based on access patterns.

```cpp
class PrefetchController {
    // Adaptive prefetch based on access pattern
    void adaptivePrefetch(void* base_addr, size_t stride, size_t distance);

    // Batch prefetch for multiple elements
    void batchPrefetch(void** addresses, size_t count);

    // Profile-guided prefetch optimization
    void profileGuidedPrefetch(AccessProfile* profile);
};
```

### 4. Graph Processing Runtime
Specialized runtime for graph algorithms.

```cpp
class GraphRuntime {
    // Initialize graph structure in remote memory
    void initializeGraph(Graph* graph, RemoteMemoryManager* mem_mgr);

    // Execute vertex-centric computation
    void executeVertexProgram(VertexProgram* program, Graph* graph);

    // Execute edge-centric computation
    void executeEdgeProgram(EdgeProgram* program, Graph* graph);
};
```

## Lowering Strategy

### MLIR to Runtime Mapping

1. **cira.offload.load_edge** → `OffloadEngine::loadEdge()`
2. **cira.offload.load_node** → `OffloadEngine::loadNode()`
3. **cira.offload.get_paddr** → `OffloadEngine::getPhysicalAddr()`
4. **cira.offload.evict_edge** → `OffloadEngine::evictEdge()`
5. **cira.call** → Direct function call with physical addresses

### Optimization Passes

1. **Prefetch Distance Analysis**: Determine optimal prefetch distance
2. **Access Pattern Recognition**: Identify strided vs random access
3. **Memory Tier Placement**: Decide data placement across memory tiers
4. **Batch Coalescing**: Combine multiple prefetch operations

## Implementation Phases

### Phase 1: Basic Runtime (Current)
- Simple offload operations
- Manual prefetch control
- Direct memory mapping

### Phase 2: Optimization Layer
- Adaptive prefetching
- Access pattern analysis
- Dynamic tier management

### Phase 3: Hardware Integration
- CXL controller integration
- DMA engine support
- Hardware prefetcher coordination

### Phase 4: Advanced Features
- Multi-node support
- Coherence management
- Fault tolerance

## Usage Example

```cpp
// Runtime initialization
auto runtime = CiraRuntime::create();
auto offload_engine = runtime->getOffloadEngine();
auto mem_manager = runtime->getMemoryManager();

// Allocate graph in remote memory
auto edge_data = mem_manager->allocateRemote(
    sizeof(Edge) * num_edges,
    MemoryTier::CXL_MEMORY
);

// Execute graph traversal
for (size_t i = 0; i < num_edges; i += cache_line_size) {
    // Prefetch next cache line
    offload_engine->loadEdge(edge_data, i, prefetch_distance);

    for (size_t j = 0; j < cache_line_size; j++) {
        auto edge = offload_engine->loadEdge(edge_data, i + j, 0);

        // Prefetch node data
        auto node_from = offload_engine->loadNode(edge, "from", 1);
        auto node_to = offload_engine->loadNode(edge, "to", 1);

        // Get physical addresses for computation
        auto paddr_from = offload_engine->getPhysicalAddr("from", node_from);
        auto paddr_to = offload_engine->getPhysicalAddr("to", node_to);

        // Execute update function
        update_node(edge, paddr_from, paddr_to);
    }

    // Evict processed cache line
    offload_engine->evictEdge(edge_data, i);
}
```

## Offload Dispatch Stack

Everything below `cira.offload` funnels through four files. They are layered,
and each one is the single implementation of its concern.

| Layer | File | Responsibility |
| --- | --- | --- |
| Wire protocol | `include/cira_cxl_job.h` | Control-window layout: doorbell, arg slots, host-status mirror, 64-byte completion line. Shared verbatim by host and device firmware. |
| Device access | `include/cira_mmio.h` | Open/mmap the CXL Type-2 control window, bounds-checked register access, doorbell submission, completion waiting. |
| Completion wait | `include/cira_mwait.h` | Cache-resident polling: bounded PAUSE spin, then `UMONITOR`/`UMWAIT`, `TPAUSE`, `MONITORX`/`MWAITX`, or `WFE`. Backend is chosen at run time from CPUID, never at compile time. |
| Specialization | `include/cira_jit_engine.h` | Folds runtime-chosen knobs into the compiler-emitted IR template and JITs it with ORC. |

### Submission path

`cira_offload_submit()` stages a `cira_cxl_call_job_t` in the arg slot for
`CIRA_CXL_JOB_CALL`, fences, then commits a doorbell carrying a monotonic
sequence number. The device firmware consumes the doorbell, runs the job, and
publishes the completion cache line; the host observes it with
`cira_future_await()` — no interrupt, no syscall. With no control window or no
device entry point configured the region stays on the host and the completion
is published in software, so callers see identical semantics either way.

### Configuration

| Variable | Meaning |
| --- | --- |
| `CIRA_CXL_MMIO_PATH` | sysfs `resource` file, `/dev/mem`, or any mappable file |
| `CIRA_CXL_MMIO_OFFSET` / `_SIZE` | mmap offset and window size |
| `CIRA_CXL_MMIO_ADDR` | use an already-mapped virtual address instead of a path |
| `CIRA_CXL_MMIO_EMULATE` | back the window with anonymous memory (no FPGA needed) |
| `CIRA_CXL_MMIO_WAIT` | block on the completion line inside `cira_offload_submit` |
| `CIRA_CXL_MMIO_TIMEOUT_NS` | completion timeout; `0` waits forever |
| `CIRA_CXL_DEVICE_FUNC_ADDR` | device-side entry point for `cira.offload` |
| `CIRA_WAIT_BACKEND` | force `spin`/`umwait`/`tpause`/`mwaitx`/`wfe` |

### JIT specialization

The compiler emits each offload region once as a *template* whose
schedule-dependent quantities are external sentinel globals. At run time
`cira_jit_decide()` picks values and `CiraJitEngine` folds them in:

```cpp
cira_jit_decision_t d;
cira_jit_decide(&workload, &limits, &d);

auto fn = reinterpret_cast<RegionFn>(
    cira::CiraJitEngine::shared().specialize("region.bc", "cira_region", d));
```

The knob set is not fixed to the four in the paper. Any named global of a
supported type can be folded, which is how a template becomes a kernel that
writes the doorbell itself:

```cpp
cira::CiraJitSpec spec;
spec.kernel = "cira_region";
spec.knobs  = cira::ciraKnobsFromDecision(d);
// control-window base, size, device entry point, wait backend
for (auto& k : cira::ciraKnobsFromDevice()) spec.knobs.push_back(k);

auto fn = engine.specializeFile("region.bc", spec);
```

Specialized kernels may call the whole CIRA runtime ABI
(`cira_offload_submit`, `cira_future_await`, `cira_mmio_submit_call`,
`cira_wait_u32`, ...); the engine binds those symbols and exposes the host
process, so a template only has to declare them. Each specialization lands in
its own `JITDylib`, so several schedules of the same region coexist.

## Building and Testing

```sh
cmake -S runtime -B build/runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/runtime -j
ctest --test-dir build/runtime --output-on-failure
```

LLVM and the Vortex SDK are both optional. Without LLVM the ORC specializer
(`cira_jit_engine.cpp` and `test_cira_jit_engine`) is skipped and the rest of
the runtime still builds; without the Vortex SDK the device side falls back to
simulation stubs. `test_cira_offload_path` exercises the full
submit/doorbell/completion protocol against an emulated window, so the suite
runs on any machine.