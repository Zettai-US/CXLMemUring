#ifndef TRANSFORM_PASSES
#define TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def RMEMSearchRemote : Pass<"rmem-search-remote", "::mlir::ModuleOp"> {
  let summary = "Search for loops that can be offloaded to remote processing";
  let description = [{
    This pass searches for loops that can be offloaded to remote execution.
    It extracts the loop body into a separate function and replaces the loop
    with a call to that function.
  }];
}

def CiraRegionFormation : Pass<"cira-region-formation", "::mlir::ModuleOp"> {
  let summary = "Form device offload regions by backward slicing from remote accesses";
  let description = [{
    Implements static region formation (Algorithms 1 and 2 of the CIRA paper).

    For every memory operation that touches CXL-resident data, the pass builds
    a backward slice from the address operand, recursively including the
    operations needed to compute that address. Constants and loop-invariant
    bases terminate the walk without being pulled in; host-only, side-effecting
    or unsupported operations terminate it and become slice boundaries.

    Slices that are device-legal and actually chase remote memory are cloned
    into a `cira.offload` region parameterized by the boundary values, and a
    `cira.barrier` is inserted before the original access.
  }];
  let options = [
    Option<"maxSliceOps", "max-slice-ops", "unsigned",
           /*default=*/"256",
           "Maximum number of operations pulled into a single slice">,
    Option<"followDependentLoads", "follow-loads", "bool",
           /*default=*/"true",
           "Follow dependent loads (pointer chasing) while slicing">,
    Option<"requireProfitable", "require-profitable", "bool",
           /*default=*/"true",
           "Only form regions whose dependence chain clears the cost model">
  ];
  let dependentDialects = ["::mlir::cira::RemoteMemDialect"];
}

def GpuOffloadDecision : Pass<"gpu-offload-decision", "::mlir::ModuleOp"> {
  let summary = "Analyze operations and decide which should be offloaded to GPU";
  let description = [{
    This pass analyzes matrix operations (GEMM, Conv2D, etc.) and automatically
    decides which ones should be offloaded to GPU based on:
    - Compute intensity (FLOPs per byte)
    - Expected speedup (CPU time vs GPU time)
    - Memory footprint
    - Data reuse patterns

    Similar to CXLMemUring's profile-guided offloading but for GPU targets.
  }];
  let options = [
    Option<"minComputeIntensity", "min-compute-intensity", "double",
           /*default=*/"10.0",
           "Minimum compute intensity (FLOPs/byte) to offload">,
    Option<"minSpeedup", "min-speedup", "double",
           /*default=*/"1.1",
           "Minimum expected speedup to offload">,
    Option<"maxMemoryBytes", "max-memory-bytes", "uint64_t",
           /*default=*/"536870912",  // 512MB
           "Maximum memory footprint for GPU operations">,
    Option<"useCoherentMemory", "use-coherent-memory", "bool",
           /*default=*/"true",
           "Use CXL coherent memory (no explicit data movement)">
  ];
}

def GpuKernelGeneration : Pass<"gpu-kernel-gen", "::mlir::ModuleOp"> {
  let summary = "Generate GPU kernel launch code for offloaded operations";
  let description = [{
    Transforms offloaded operations (marked by gpu-offload-decision) into
    explicit GPU kernel launch calls with proper memory management.

    Inserts:
    - Memory allocation for GPU buffers
    - Data copy operations to GPU memory
    - Kernel launch with proper CSR setup
    - Synchronization and result retrieval
  }];
  let options = [
    Option<"gpuMemoryBudget", "gpu-memory-budget", "uint64_t",
           /*default=*/"268435456",  // 256MB
           "Total GPU memory budget for allocation">
  ];
}

def GpuMemoryOptimization : Pass<"gpu-memory-opt", "::mlir::ModuleOp"> {
  let summary = "Optimize GPU memory usage for offloaded kernels";
  let description = [{
    Performs GPU-specific memory optimizations:
    - Reuse allocations across multiple kernels
    - Pipeline data movement with computation
    - Coalesce memory accesses
    - Minimize coherent memory transfers
  }];
}

def GpuRuntimeLowering : Pass<"gpu-runtime-lowering", "::mlir::ModuleOp"> {
  let summary = "Lower GPU-annotated MLIR to concrete GPU runtime calls";
  let description = [{
    Converts MLIR operations marked for GPU execution into
    concrete GPU runtime function calls. Prepares for code generation
    targeting specific GPU architectures (Vortex, Type2, etc).
  }];
}

def VortexKernelGen : Pass<"vortex-kernel-gen", "::mlir::ModuleOp"> {
  let summary = "Generate Vortex GPU kernel code from offloaded operations";
  let description = [{
    Generates C++ kernel source code for Vortex GPU architecture
    from MLIR operations marked for GPU offloading. Produces
    compilable kernel source that can be converted to .vxbin format.
  }];
}

def Type2GpuCodeGen : Pass<"type2-gpu-codegen", "::mlir::ModuleOp"> {
  let summary = "Generate Type2 GPU host code with CXL Type2 data sharing";
  let description = [{
    Generates host-side C++ code for Intel IA-780i Type2 GPU execution
    using CXL Type2 data sharing (DCOH-coherent memory). Integrates
    CSR-based kernel control and CompletionData synchronization.

    Features:
    - BAR0 DCOH-coherent memory allocation
    - Type2KernelRequest submission
    - CompletionData writeback synchronization
    - Direct CPU-GPU data access (zero-copy via CXL)
  }];
}

#endif
