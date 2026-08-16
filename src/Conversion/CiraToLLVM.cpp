#include "Conversion/CiraToLLVM.h"
#include "Dialect/CiraOps.h"
#include "Dialect/RemoteMem.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"
// Allow marking CIR dialect ops as legal when running CIRA-only lowering
#include <clang/CIR/Dialect/IR/CIRDialect.h>

using namespace mlir;
using namespace mlir::cira;

enum class TargetArchitecture {
    X86,
    ARM,
    RISCV_VORTEX, // Vortex RISC-V SIMT (generates NVVM IR)
    Heterogeneous
};

namespace {

// Command-line options for target architecture
static llvm::cl::opt<std::string> targetArch("target-arch", llvm::cl::desc("Target architecture for code generation"),
                                             llvm::cl::init("auto"));

// Helper function to determine target architecture
TargetArchitecture getTargetArchitecture(ModuleOp module) {
    if (targetArch != "auto") {
        if (targetArch == "x86")
            return TargetArchitecture::X86;
        else if (targetArch == "arm")
            return TargetArchitecture::ARM;
        else if (targetArch == "vortex" || targetArch == "riscv-vortex")
            return TargetArchitecture::RISCV_VORTEX;
        else if (targetArch == "hetero")
            return TargetArchitecture::Heterogeneous;
    }

    // Check module's target triple attribute
    if (auto tripleAttr = module->getAttrOfType<StringAttr>("llvm.target_triple")) {
        llvm::Triple triple(tripleAttr.getValue());
        if (triple.isX86())
            return TargetArchitecture::X86;
        else if (triple.isARM() || triple.isAArch64())
            return TargetArchitecture::ARM;
        else if (triple.isRISCV()) {
            // Check for Vortex-specific markers
            if (triple.getVendorName() == "vortex" || module->hasAttr("vortex.target"))
                return TargetArchitecture::RISCV_VORTEX;
        }
    }

    // Default to heterogeneous for CXLMemUring
    return TargetArchitecture::Heterogeneous;
}

//===----------------------------------------------------------------------===//
// NVVM Intrinsic Generation for Vortex
//===----------------------------------------------------------------------===//

// Generate NVVM thread indexing intrinsics (map to Vortex vx_thread_id via CuPBoP)
static Value generateNVVMThreadIdx(ConversionPatternRewriter &rewriter, Location loc) {
    auto context = rewriter.getContext();
    auto i32Type = IntegerType::get(context, 32);
    auto tidFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.read.ptx.sreg.tid.x");
    return rewriter.create<LLVM::CallOp>(loc, i32Type, tidFunc).getResult();
}

static Value generateNVVMBlockIdx(ConversionPatternRewriter &rewriter, Location loc) {
    auto context = rewriter.getContext();
    auto i32Type = IntegerType::get(context, 32);
    auto bidFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.read.ptx.sreg.ctaid.x");
    return rewriter.create<LLVM::CallOp>(loc, i32Type, bidFunc).getResult();
}

static Value generateNVVMBlockDim(ConversionPatternRewriter &rewriter, Location loc) {
    auto context = rewriter.getContext();
    auto i32Type = IntegerType::get(context, 32);
    auto bdimFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.read.ptx.sreg.ntid.x");
    return rewriter.create<LLVM::CallOp>(loc, i32Type, bdimFunc).getResult();
}

// Generate global thread ID: blockIdx.x * blockDim.x + threadIdx.x
static Value generateGlobalThreadId(ConversionPatternRewriter &rewriter, Location loc) {
    auto tid = generateNVVMThreadIdx(rewriter, loc);
    auto bid = generateNVVMBlockIdx(rewriter, loc);
    auto bdim = generateNVVMBlockDim(rewriter, loc);

    auto mul = rewriter.create<LLVM::MulOp>(loc, bid, bdim);
    return rewriter.create<LLVM::AddOp>(loc, tid, mul);
}

// Generate NVVM warp-level intrinsics (map to Vortex vx_* intrinsics)
static void generateNVVMBarrier(ConversionPatternRewriter &rewriter, Location loc) {
    auto context = rewriter.getContext();
    auto barrierFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.barrier0");
    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, barrierFunc, ValueRange{});
}

static Value generateNVVMBallot(ConversionPatternRewriter &rewriter, Location loc, Value predicate) {
    auto context = rewriter.getContext();
    auto i32Type = IntegerType::get(context, 32);
    auto ballotFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.vote.ballot.sync");

    // All-active mask for warp (0xFFFFFFFF)
    auto mask = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(-1));

    return rewriter.create<LLVM::CallOp>(loc, i32Type, ballotFunc, ValueRange{mask, predicate}).getResult();
}

static Value generateNVVMShfl(ConversionPatternRewriter &rewriter, Location loc, Value var, Value lane) {
    auto context = rewriter.getContext();
    auto i32Type = IntegerType::get(context, 32);
    auto shflFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.shfl.sync.idx.i32");

    auto mask = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(-1));
    auto warpSize = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(32));

    return rewriter.create<LLVM::CallOp>(loc, i32Type, shflFunc, ValueRange{mask, var, lane, warpSize}).getResult();
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Convert cira.offload.load_edge to LLVM operations
struct LoadEdgeOpLowering : public ConversionPattern {
    TargetArchitecture targetArch;

    LoadEdgeOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
        : ConversionPattern(converter, LoadEdgeOp::getOperationName(), 1, &converter.getContext()), targetArch(arch) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loadOp = cast<LoadEdgeOp>(op);
        Location loc = op->getLoc();

        // Get the type converter
        auto *converter = const_cast<LLVMTypeConverter *>(static_cast<const LLVMTypeConverter *>(getTypeConverter()));

        // Convert result type
        Type resultType = converter->convertType(loadOp.getType());
        if (!resultType)
            return failure();

        // Create LLVM operations to implement load_edge
        // This is a simplified version - you would implement the actual logic here
        // For now, we'll create a placeholder that loads from a pointer

        // Calculate the address: base_ptr + index * element_size
        auto indexType = converter->getIndexType();
        auto index = operands[1];

        // Get element size (placeholder - you'd calculate this based on the edge struct)
        auto elementSize = rewriter.create<LLVM::ConstantOp>(
            loc, indexType, rewriter.getIntegerAttr(indexType, 32)); // Assuming 32-byte edges

        // Calculate offset
        auto offset = rewriter.create<LLVM::MulOp>(loc, index, elementSize);

        // GEP to get the element address
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto gep = rewriter.create<LLVM::GEPOp>(loc, ptrType, resultType, operands[0], ValueRange(offset));

        // Load the value
        auto loadedValue = rewriter.create<LLVM::LoadOp>(loc, resultType, gep);

        // Handle prefetch based on target architecture
        if (operands.size() > 2 && operands[2]) {
            // Prefetch will be handled by target-specific lowering
            // For now, emit a generic prefetch intrinsic
            auto context = rewriter.getContext();
            auto prefetchFunc = FlatSymbolRefAttr::get(context, "llvm.prefetch.p0");
            auto i32Type = IntegerType::get(context, 32);
            auto rw = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));
            auto locality = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(3));
            auto cacheType = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));
            rewriter.create<LLVM::CallOp>(loc, TypeRange{}, prefetchFunc, ValueRange{gep, rw, locality, cacheType});
        }

        rewriter.replaceOp(op, loadedValue.getResult());
        return success();
    }
};

/// Convert cira.offload.load_node to LLVM operations
struct LoadNodeOpLowering : public ConversionPattern {
    LoadNodeOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, LoadNodeOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loadOp = cast<LoadNodeOp>(op);
        Location loc = op->getLoc();

        auto *converter = const_cast<LLVMTypeConverter *>(static_cast<const LLVMTypeConverter *>(getTypeConverter()));
        Type resultType = converter->convertType(loadOp.getType());
        if (!resultType)
            return failure();

        // Extract field based on field_name attribute
        StringRef fieldName = loadOp.getFieldName();

        // This is a placeholder implementation
        // In a real implementation, you would:
        // 1. Extract the appropriate field from the edge element
        // 2. Load the node data from that address
        // 3. Handle prefetching if specified

        // For now, just extract a field from the struct
        // You would implement proper struct field extraction here
        Value nodePtr;
        if (fieldName == "from") {
            // Extract 'from' field (assuming it's at offset 0)
            nodePtr = rewriter.create<LLVM::ExtractValueOp>(loc, operands[0], ArrayRef<int64_t>{0});
        } else if (fieldName == "to") {
            // Extract 'to' field (assuming it's at offset 1)
            nodePtr = rewriter.create<LLVM::ExtractValueOp>(loc, operands[0], ArrayRef<int64_t>{1});
        } else {
            return failure();
        }

        rewriter.replaceOp(op, nodePtr);
        return success();
    }
};

/// Convert cira.offload.get_paddr to LLVM operations
struct GetPaddrOpLowering : public ConversionPattern {
    GetPaddrOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, GetPaddrOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op->getLoc();
        auto *ctx = rewriter.getContext();
        auto *converter = static_cast<const LLVMTypeConverter *>(getTypeConverter());

        Type resultType = converter->convertType(op->getResult(0).getType());
        if (!resultType)
            return failure();

        auto ptrType = LLVM::LLVMPointerType::get(ctx);
        auto nullFieldName = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
        auto translateFunc = FlatSymbolRefAttr::get(ctx, "cira_translate_paddr");

        auto translated = rewriter.create<LLVM::CallOp>(loc, resultType, translateFunc,
                                                        ValueRange{nullFieldName.getResult(), operands[0]});
        rewriter.replaceOp(op, translated.getResults());
        return success();
    }
};

// Helper function to emit architecture-specific prefetch
static void emitPrefetchForTarget(ConversionPatternRewriter &rewriter, Location loc, Value addr,
                                  TargetArchitecture arch) {
    auto context = rewriter.getContext();

    if (arch == TargetArchitecture::X86 || arch == TargetArchitecture::Heterogeneous) {
        // x86 prefetch intrinsic: llvm.prefetch(ptr, rw, locality, cache_type)
        auto prefetchFunc = FlatSymbolRefAttr::get(context, "llvm.prefetch.p0");
        auto i32Type = IntegerType::get(context, 32);

        // rw=0 (read), locality=3 (high), cache_type=1 (data)
        auto rw = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));
        auto locality = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(3));
        auto cacheType = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));

        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, prefetchFunc, ValueRange{addr, rw, locality, cacheType});
    }

    if (arch == TargetArchitecture::ARM) {
        // ARM/AArch64 prefetch intrinsic: llvm.aarch64.prefetch(ptr, rw, target, stream, keep)
        auto prefetchFunc = FlatSymbolRefAttr::get(context, "llvm.aarch64.prefetch.p0");
        auto i32Type = IntegerType::get(context, 32);

        // rw=0 (read), target=0 (L1), stream=3 (all levels), keep=0 (temporal), stream=1
        auto rw = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));
        auto target = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(3));
        auto stream = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));
        auto keep = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));

        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, prefetchFunc, ValueRange{addr, rw, target, stream, keep});
    }

    if (arch == TargetArchitecture::RISCV_VORTEX) {
        // NVVM prefetch intrinsic (maps to Vortex vx_prefetch_l1 via CuPBoP)
        // Use NVVM's prefetch.global intrinsic which CuPBoP translates
        auto prefetchFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.prefetch.global");
        auto i32Type = IntegerType::get(context, 32);

        // level=1 (L1 cache), hint=0 (load)
        auto level = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(1));
        auto hint = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));

        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, prefetchFunc, ValueRange{addr, level, hint});
    }
}

/// Convert cira.offload.evict_edge to LLVM operations
struct EvictEdgeOpLowering : public ConversionPattern {
    TargetArchitecture targetArch;

    EvictEdgeOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
        : ConversionPattern(converter, EvictEdgeOp::getOperationName(), 1, &converter.getContext()), targetArch(arch) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op->getLoc();
        auto context = rewriter.getContext();

        // Emit architecture-specific cache eviction
        if (targetArch == TargetArchitecture::X86 || targetArch == TargetArchitecture::Heterogeneous) {
            // x86: Use clflush instruction
            auto clflushFunc = FlatSymbolRefAttr::get(context, "llvm.x86.sse2.clflush");
            rewriter.create<LLVM::CallOp>(loc, TypeRange{}, clflushFunc, operands[0]);
        }

        if (targetArch == TargetArchitecture::ARM) {
            // ARM: Use DC CIVAC (Clean and Invalidate by VA to PoC)
            auto dcFunc = FlatSymbolRefAttr::get(context, "llvm.aarch64.dc.civac");
            rewriter.create<LLVM::CallOp>(loc, TypeRange{}, dcFunc, operands[0]);
        }

        if (targetArch == TargetArchitecture::RISCV_VORTEX) {
            // Vortex: Use memory fence to flush cache
            // NVVM doesn't have a direct evict, so use a fence to ensure coherency
            auto fenceFunc = FlatSymbolRefAttr::get(context, "llvm.nvvm.membar.gl");
            rewriter.create<LLVM::CallOp>(loc, TypeRange{}, fenceFunc, ValueRange{});
        }

        rewriter.eraseOp(op);
        return success();
    }
};

/// Convert cira.call to LLVM call
struct CallOpLowering : public ConversionPattern {
    CallOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, CallOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto callOp = cast<CallOp>(op);
        auto *converter = const_cast<LLVMTypeConverter *>(static_cast<const LLVMTypeConverter *>(getTypeConverter()));

        // Convert result types
        SmallVector<Type> resultTypes;
        if (failed(converter->convertTypes(callOp.getResultTypes(), resultTypes)))
            return failure();

        // Create LLVM call
        auto llvmCallOp = rewriter.create<LLVM::CallOp>(op->getLoc(), resultTypes, callOp.getCalleeAttr(), operands);

        rewriter.replaceOp(op, llvmCallOp.getResults());
        return success();
    }
};

//===----------------------------------------------------------------------===//
// New CIRA Operation Lowerings (Memory, Cache, Sync, Control, Stream)
//===----------------------------------------------------------------------===//

/// Lower cira.alloc_cxl to a runtime-managed LLC tile allocation.
struct AllocCxlOpLowering : public ConversionPattern {
    AllocCxlOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, AllocCxlOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op->getLoc();
        auto *ctx = rewriter.getContext();

        auto allocFunc = FlatSymbolRefAttr::get(ctx, "cira_llc_tile_alloc");
        auto ptrType = LLVM::LLVMPointerType::get(ctx);

        auto result = rewriter.create<LLVM::CallOp>(loc, ptrType, allocFunc, operands[0]);

        rewriter.replaceOp(op, result.getResult());
        return success();
    }
};

/// Lower cira.load_async to non-blocking load + prefetch hint
struct LoadAsyncOpLowering : public ConversionPattern {
    TargetArchitecture targetArch;

    LoadAsyncOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
        : ConversionPattern(converter, LoadAsyncOp::getOperationName(), 1, &converter.getContext()), targetArch(arch) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loadOp = cast<LoadAsyncOp>(op);
        Location loc = op->getLoc();
        auto *ctx = rewriter.getContext();
        auto *converter = static_cast<const LLVMTypeConverter *>(getTypeConverter());

        // Get the element type
        Type resultType = converter->convertType(loadOp.getType());
        auto ptrType = LLVM::LLVMPointerType::get(ctx);

        // Issue prefetch for the address
        emitPrefetchForTarget(rewriter, loc, operands[0], targetArch);

        // Create the load (will complete when data arrives)
        auto loadResult = rewriter.create<LLVM::LoadOp>(loc, resultType, operands[0]);

        // For now, the "future" is just the loaded value
        // Real implementation would use io_uring or similar async mechanism
        rewriter.replaceOp(op, loadResult.getResult());
        return success();
    }
};

/// Lower cira.store_async to non-blocking store
struct StoreAsyncOpLowering : public ConversionPattern {
    StoreAsyncOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, StoreAsyncOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op->getLoc();

        // Create non-temporal store for CXL memory
        rewriter.create<LLVM::StoreOp>(loc, operands[0], operands[1],
                                       /*alignment=*/0, /*isVolatile=*/false,
                                       /*isNonTemporal=*/true);

        // Return a "completed" future (null for now)
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
        auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
        rewriter.replaceOp(op, nullPtr.getResult());
        return success();
    }
};

/// Lower cira.flush to memory fence
struct FlushOpLowering : public ConversionPattern {
    FlushOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, FlushOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op->getLoc();

        // Issue memory fence
        rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::seq_cst);
        rewriter.eraseOp(op);
        return success();
    }
};

/// Lower cira.install_cacheline to architecture-specific cache install
struct InstallCachelineOpLowering : public ConversionPattern {
    TargetArchitecture targetArch;

    InstallCachelineOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
        : ConversionPattern(converter, InstallCachelineOp::getOperationName(), 1, &converter.getContext()),
          targetArch(arch) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto installOp = cast<InstallCachelineOp>(op);
        Location loc = op->getLoc();
        auto *ctx = rewriter.getContext();
        auto i32Type = IntegerType::get(ctx, 32);
        auto i64Type = IntegerType::get(ctx, 64);
        auto ptrType = LLVM::LLVMPointerType::get(ctx);

        // Get cache level (default L3 = 3)
        int64_t cacheLevel = 3;
        if (auto levelAttr = installOp.getCacheLevelAttr())
            cacheLevel = levelAttr.getInt();

        Value ptr = operands[0];
        Value size = operands[1];

        if (targetArch == TargetArchitecture::X86 || targetArch == TargetArchitecture::Heterogeneous) {
            // x86: Use prefetchT0/T1/T2 based on cache level, then MWAIT-style polling.
            // For LLC install: emit a loop that prefetches each cache line in the range.
            //
            // For each 64-byte cache line in [ptr, ptr+size):
            //   _mm_prefetch(addr, _MM_HINT_T0)   // install into L1 (propagates to all levels)
            //
            // Cache level mapping:
            //   L1 -> PREFETCHT0 (locality=3)
            //   L2 -> PREFETCHT1 (locality=2)
            //   L3 -> PREFETCHT2 (locality=1)
            int locality = (cacheLevel == 1) ? 3 : (cacheLevel == 2) ? 2 : 1;

            auto prefetchFunc = FlatSymbolRefAttr::get(ctx, "llvm.prefetch.p0");
            auto rw = rewriter.create<LLVM::ConstantOp>(loc, i32Type,
                                                        rewriter.getI32IntegerAttr(0)); // read
            auto localityVal = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(locality));
            auto cacheType = rewriter.create<LLVM::ConstantOp>(loc, i32Type,
                                                               rewriter.getI32IntegerAttr(1)); // data cache

            // Cache line size = 64 bytes
            auto cacheLineSize = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(64));
            auto zero64 = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(0));

            // Create prefetch loop: for (offset = 0; offset < size; offset += 64)
            auto i8Type = IntegerType::get(ctx, 8);

            // Emit loop: we unroll up to 16 cache lines, or emit a counted loop
            // For simplicity, call runtime helper that loops internally
            auto installFunc = FlatSymbolRefAttr::get(ctx, "cira_install_cacheline_x86");
            auto levelArg = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(cacheLevel));
            rewriter.create<LLVM::CallOp>(loc, TypeRange{}, installFunc, ValueRange{ptr, size, levelArg});
        }

        if (targetArch == TargetArchitecture::RISCV_VORTEX) {
            // Vortex device-side: take ownership of cacheline via CXL.cache protocol.
            // This is the device writing to the host-visible address, which triggers
            // DCOH writeback and installs the line in the host LLC.
            auto installFunc = FlatSymbolRefAttr::get(ctx, "__vortex_install_cacheline");
            auto levelArg = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(cacheLevel));
            rewriter.create<LLVM::CallOp>(loc, TypeRange{}, installFunc, ValueRange{ptr, size, levelArg});
        }

        rewriter.eraseOp(op);
        return success();
    }
};

/// Lower cira.evict_hint to architecture-specific cache eviction
struct EvictHintOpLowering : public ConversionPattern {
    TargetArchitecture targetArch;

    EvictHintOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
        : ConversionPattern(converter, EvictHintOp::getOperationName(), 1, &converter.getContext()), targetArch(arch) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op->getLoc();
        auto *ctx = rewriter.getContext();
        auto ptrType = LLVM::LLVMPointerType::get(ctx);

        Value ptr = operands[0];

        if (targetArch == TargetArchitecture::X86 || targetArch == TargetArchitecture::Heterogeneous) {
            // x86: Use CLDEMOTE to move cache line from L1 to LLC (non-destructive),
            // or CLFLUSHOPT for full eviction. CLDEMOTE is preferred because it
            // keeps the line in LLC for potential reuse by other cores.
            //
            // CLDEMOTE is available on Granite Rapids (ICX+). If not available,
            // falls back to CLFLUSHOPT.
            auto evictFunc = FlatSymbolRefAttr::get(ctx, "cira_evict_hint_x86");
            if (operands.size() > 1 && operands[1]) {
                // Have size: evict range
                rewriter.create<LLVM::CallOp>(loc, TypeRange{}, evictFunc, ValueRange{ptr, operands[1]});
            } else {
                // Single cache line
                auto i64Type = IntegerType::get(ctx, 64);
                auto defaultSize = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(64));
                rewriter.create<LLVM::CallOp>(loc, TypeRange{}, evictFunc, ValueRange{ptr, defaultSize});
            }
        }

        if (targetArch == TargetArchitecture::RISCV_VORTEX) {
            // Vortex: issue CXL.cache back-invalidation hint
            auto fenceFunc = FlatSymbolRefAttr::get(ctx, "llvm.nvvm.membar.gl");
            rewriter.create<LLVM::CallOp>(loc, TypeRange{}, fenceFunc, ValueRange{});
        }

        rewriter.eraseOp(op);
        return success();
    }
};

/// Lower cira.prefetch_stream to prefetch intrinsics
struct PrefetchStreamOpLowering : public ConversionPattern {
    TargetArchitecture targetArch;

    PrefetchStreamOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
        : ConversionPattern(converter, PrefetchStreamOp::getOperationName(), 1, &converter.getContext()),
          targetArch(arch) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto prefetchOp = cast<PrefetchStreamOp>(op);
        Location loc = op->getLoc();

        // Issue multiple prefetch hints for the stream
        // For simplicity, prefetch the first few cache lines
        emitPrefetchForTarget(rewriter, loc, operands[0], targetArch);

        rewriter.eraseOp(op);
        return success();
    }
};

/// Lower cira.prefetch_indirect to pointer-chasing prefetch
struct PrefetchIndirectOpLowering : public ConversionPattern {
    TargetArchitecture targetArch;

    PrefetchIndirectOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
        : ConversionPattern(converter, PrefetchIndirectOp::getOperationName(), 1, &converter.getContext()),
          targetArch(arch) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto prefetchOp = cast<PrefetchIndirectOp>(op);
        Location loc = op->getLoc();
        auto *ctx = rewriter.getContext();

        // For Vortex target, generate microkernel for pointer chasing
        if (targetArch == TargetArchitecture::RISCV_VORTEX) {
            // Generate NVVM-style kernel call for Vortex
            auto launchFunc = FlatSymbolRefAttr::get(ctx, "__vortex_prefetch_chain");
            auto i64Type = IntegerType::get(ctx, 64);

            auto offset = rewriter.create<LLVM::ConstantOp>(loc, i64Type, prefetchOp.getNextPtrOffsetAttr());
            auto depth = rewriter.create<LLVM::ConstantOp>(loc, i64Type, prefetchOp.getDepthAttr());

            rewriter.create<LLVM::CallOp>(loc, TypeRange{}, launchFunc, ValueRange{operands[0], offset, depth});
        } else {
            // For x86/ARM, just prefetch the first node
            emitPrefetchForTarget(rewriter, loc, operands[0], targetArch);
        }

        rewriter.eraseOp(op);
        return success();
    }
};

/// Lower cira.future_create to allocate cache-line-aligned completion struct
struct FutureCreateOpLowering : public ConversionPattern {
    FutureCreateOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, FutureCreateOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op->getLoc();
        auto *ctx = rewriter.getContext();
        auto ptrType = LLVM::LLVMPointerType::get(ctx);
        auto i64Type = IntegerType::get(ctx, 64);

        // Allocate a 64-byte-aligned completion structure (matches Type2KernelCompletion)
        // Layout: [magic:u32, status:u32, result:u64, cycles:u64, timestamp:u64, pad:32]
        // Total: 64 bytes = 1 cache line
        auto allocFunc = FlatSymbolRefAttr::get(ctx, "cira_future_alloc");
        auto result = rewriter.create<LLVM::CallOp>(loc, ptrType, allocFunc, ValueRange{});

        // Initialize magic to 0 (not ready; device writes 0xDEADBEEF when done)
        auto i32Type = IntegerType::get(ctx, 32);
        auto zeroMagic = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));
        rewriter.create<LLVM::StoreOp>(loc, zeroMagic, result.getResult(),
                                       /*alignment=*/64);

        rewriter.replaceOp(op, result.getResult());
        return success();
    }
};

/// Lower cira.future_await to mwait-style polling on completion cacheline
struct FutureAwaitOpLowering : public ConversionPattern {
    TargetArchitecture targetArch;

    FutureAwaitOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
        : ConversionPattern(converter, FutureAwaitOp::getOperationName(), 1, &converter.getContext()),
          targetArch(arch) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto awaitOp = cast<FutureAwaitOp>(op);
        Location loc = op->getLoc();
        auto *ctx = rewriter.getContext();
        auto *converter = static_cast<const LLVMTypeConverter *>(getTypeConverter());

        Type resultType = converter->convertType(awaitOp.getType());
        if (!resultType)
            return failure();

        // Call cira_future_await which implements:
        //   1. MONITOR the completion cacheline address
        //   2. Check if magic == 0xDEADBEEF (DCOH-delivered)
        //   3. If not ready: MWAIT (on Granite Rapids, this doesn't alter cstate)
        //   4. When ready: load result value from completion struct
        auto awaitFunc = FlatSymbolRefAttr::get(ctx, "cira_future_await");
        auto ptrType = LLVM::LLVMPointerType::get(ctx);

        auto result = rewriter.create<LLVM::CallOp>(loc, ptrType, awaitFunc, ValueRange{operands[0]});

        // Load the actual result from the returned pointer
        auto loadResult = rewriter.create<LLVM::LoadOp>(loc, resultType, result.getResult());

        rewriter.replaceOp(op, loadResult.getResult());
        return success();
    }
};

/// Lower cira.barrier to memory fence
struct BarrierOpLowering : public ConversionPattern {
    BarrierOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, BarrierOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op->getLoc();
        rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::seq_cst);
        rewriter.eraseOp(op);
        return success();
    }
};

/// Lower cira.release to the runtime tile/free path.
struct ReleaseOpLowering : public ConversionPattern {
    ReleaseOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, ReleaseOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op->getLoc();
        auto *ctx = rewriter.getContext();

        auto freeFunc = FlatSymbolRefAttr::get(ctx, "cira_llc_tile_free");
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, freeFunc, operands[0]);
        rewriter.eraseOp(op);
        return success();
    }
};

/// Lower cira.phase_boundary to barrier + runtime phase notification
struct PhaseBoundaryOpLowering : public ConversionPattern {
    PhaseBoundaryOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, PhaseBoundaryOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto phaseOp = cast<PhaseBoundaryOp>(op);
        Location loc = op->getLoc();
        auto *ctx = rewriter.getContext();

        // 1. Issue full memory fence to ensure all prior operations complete
        rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::seq_cst);

        // 2. Wait for device to finish all current offload tasks
        //    (checked through control bits in completion cachelines)
        auto barrierFunc = FlatSymbolRefAttr::get(ctx, "cira_phase_barrier");
        auto ptrType = LLVM::LLVMPointerType::get(ctx);

        if (auto phaseName = phaseOp.getPhaseNameAttr()) {
            // Pass phase name to runtime for profiling/debugging
            auto phaseNameFunc = FlatSymbolRefAttr::get(ctx, "cira_phase_boundary_named");
            // Create global string for phase name
            auto i8Type = IntegerType::get(ctx, 8);
            auto strType = LLVM::LLVMArrayType::get(i8Type, phaseName.getValue().size() + 1);
            auto globalName = std::string("__cira_phase_") + phaseName.getValue().str();

            // Emit runtime call with phase name
            rewriter.create<LLVM::CallOp>(loc, TypeRange{}, barrierFunc, ValueRange{});
        } else {
            rewriter.create<LLVM::CallOp>(loc, TypeRange{}, barrierFunc, ValueRange{});
        }

        // 3. Second fence after barrier to ensure ordering for next phase
        rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::seq_cst);

        rewriter.eraseOp(op);
        return success();
    }
};

/// Lower cira.speculate to conditional execution region
struct SpeculateOpLowering : public ConversionPattern {
    SpeculateOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, SpeculateOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto specOp = cast<SpeculateOp>(op);
        Location loc = op->getLoc();

        // Inline the speculative body - at LLVM level, speculation is handled
        // by the runtime; here we just emit the operations.
        // The runtime can later decide to discard results if speculation fails.
        Block &body = specOp.getBody().front();
        SmallVector<Value> results;

        for (auto &bodyOp : llvm::make_early_inc_range(body.getOperations())) {
            if (auto yieldOp = dyn_cast<YieldOp>(&bodyOp)) {
                for (auto val : yieldOp.getValues())
                    results.push_back(val);
            } else {
                rewriter.clone(bodyOp);
            }
        }

        rewriter.replaceOp(op, results);
        return success();
    }
};

/// Lower cira.stream_create_indirect - create stream state structure
struct StreamCreateIndirectOpLowering : public ConversionPattern {
    StreamCreateIndirectOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, StreamCreateIndirectOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        // For now, the stream is just the starting pointer
        // Real impl would allocate a stream descriptor struct
        rewriter.replaceOp(op, operands[0]);
        return success();
    }
};

/// Lower cira.prefetch_chain - generate Vortex microkernel
struct PrefetchChainOpLowering : public ConversionPattern {
    TargetArchitecture targetArch;

    PrefetchChainOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
        : ConversionPattern(converter, PrefetchChainOp::getOperationName(), 1, &converter.getContext()),
          targetArch(arch) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto chainOp = cast<PrefetchChainOp>(op);
        Location loc = op->getLoc();
        auto *ctx = rewriter.getContext();

        if (targetArch == TargetArchitecture::RISCV_VORTEX) {
            // Generate NVVM kernel call
            auto launchFunc = FlatSymbolRefAttr::get(ctx, "__vortex_prefetch_chain_kernel");
            auto i64Type = IntegerType::get(ctx, 64);
            auto depth = rewriter.create<LLVM::ConstantOp>(loc, i64Type, chainOp.getDepthAttr());

            rewriter.create<LLVM::CallOp>(loc, TypeRange{}, launchFunc, ValueRange{operands[0], depth});
        }

        rewriter.eraseOp(op);
        return success();
    }
};

/// Lower cira.peek_stream - return current element
struct PeekStreamOpLowering : public ConversionPattern {
    PeekStreamOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, PeekStreamOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto peekOp = cast<PeekStreamOp>(op);
        Location loc = op->getLoc();
        auto *converter = static_cast<const LLVMTypeConverter *>(getTypeConverter());

        Type resultType = converter->convertType(peekOp.getType());

        // Load from current stream position
        auto loadResult = rewriter.create<LLVM::LoadOp>(loc, resultType, operands[0]);
        rewriter.replaceOp(op, loadResult.getResult());
        return success();
    }
};

/// Lower cira.advance_stream - update stream pointer
struct AdvanceStreamOpLowering : public ConversionPattern {
    AdvanceStreamOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, AdvanceStreamOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        // Stream advancement is handled by the transformed loop structure
        rewriter.eraseOp(op);
        return success();
    }
};

/// Lower cira.offload_start - emit Vortex kernel launch
struct OffloadStartOpLowering : public ConversionPattern {
    TargetArchitecture targetArch;

    OffloadStartOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
        : ConversionPattern(converter, OffloadStartOp::getOperationName(), 1, &converter.getContext()),
          targetArch(arch) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto offloadOp = cast<OffloadStartOp>(op);
        Location loc = op->getLoc();

        if (targetArch == TargetArchitecture::RISCV_VORTEX) {
            // Emit kernel launch for Vortex
            // The body operations will be lowered separately
        }

        // Inline the body operations
        Block &body = offloadOp.getBody().front();
        for (auto &bodyOp : llvm::make_early_inc_range(body.getOperations())) {
            rewriter.clone(bodyOp);
        }

        rewriter.eraseOp(op);
        return success();
    }
};

//===----------------------------------------------------------------------===//
// CXL Type 2 Operation Lowerings (.cache and .mem protocols)
//===----------------------------------------------------------------------===//

/// Lower cira.type2.cache_load to runtime call cira_type2_cache_load
struct Type2CacheLoadOpLowering : public ConversionPattern {
    TargetArchitecture targetArch;

    Type2CacheLoadOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
        : ConversionPattern(converter, Type2CacheLoadOp::getOperationName(), 1, &converter.getContext()),
          targetArch(arch) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loadOp = cast<Type2CacheLoadOp>(op);
        Location loc = op->getLoc();
        auto *ctx = rewriter.getContext();
        auto *converter = static_cast<const LLVMTypeConverter *>(getTypeConverter());

        Type resultType = converter->convertType(loadOp.getType());
        auto ptrType = LLVM::LLVMPointerType::get(ctx);
        auto i64Type = IntegerType::get(ctx, 64);
        auto i8Type = IntegerType::get(ctx, 8);

        // Allocate stack slot for coherency state output
        auto one = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(1));
        auto stateAlloca = rewriter.create<LLVM::AllocaOp>(loc, ptrType, i8Type, one);

        // Call cira_type2_cache_load(controller, host_addr, size, &state)
        auto cacheLoadFunc = FlatSymbolRefAttr::get(ctx, "cira_type2_cache_load");
        auto result = rewriter.create<LLVM::CallOp>(loc, ptrType, cacheLoadFunc,
                                                    ValueRange{operands[0], operands[0], operands[1], stateAlloca});

        // For RISCV_VORTEX target, also emit speculative prefetch chain
        if (targetArch == TargetArchitecture::RISCV_VORTEX) {
            auto prefetchFunc = FlatSymbolRefAttr::get(ctx, "__vortex_prefetch_chain");
            auto offset = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(0));
            auto depth = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(4));
            rewriter.create<LLVM::CallOp>(loc, TypeRange{}, prefetchFunc, ValueRange{operands[0], offset, depth});
        }

        rewriter.replaceOp(op, result.getResult());
        return success();
    }
};

/// Lower cira.type2.cache_store to runtime call cira_type2_cache_store
struct Type2CacheStoreOpLowering : public ConversionPattern {
    Type2CacheStoreOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, Type2CacheStoreOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op->getLoc();
        auto *ctx = rewriter.getContext();
        auto ptrType = LLVM::LLVMPointerType::get(ctx);
        auto i64Type = IntegerType::get(ctx, 64);

        // Store value to a temporary, then pass pointer to runtime
        auto one = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(1));
        auto valAlloca = rewriter.create<LLVM::AllocaOp>(loc, ptrType, operands[0].getType(), one);
        rewriter.create<LLVM::StoreOp>(loc, operands[0], valAlloca);

        // size = sizeof(element) — use constant matching the type
        auto size = rewriter.create<LLVM::ConstantOp>(loc, i64Type,
                                                      rewriter.getI64IntegerAttr(8)); // default 8 bytes

        // Call cira_type2_cache_store(controller, host_addr, data, size)
        auto cacheStoreFunc = FlatSymbolRefAttr::get(ctx, "cira_type2_cache_store");
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, cacheStoreFunc,
                                      ValueRange{operands[1], operands[1], valAlloca, size});

        // Return null future
        auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
        rewriter.replaceOp(op, nullPtr.getResult());
        return success();
    }
};

/// Lower cira.type2.mem_load to GEP + LLVM LoadOp with non-temporal hint
struct Type2MemLoadOpLowering : public ConversionPattern {
    Type2MemLoadOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, Type2MemLoadOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto memLoadOp = cast<Type2MemLoadOp>(op);
        Location loc = op->getLoc();
        auto *converter = static_cast<const LLVMTypeConverter *>(getTypeConverter());

        Type resultType = converter->convertType(memLoadOp.getType());
        if (!resultType)
            return failure();

        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

        // GEP: dev_ptr + offset
        auto gep = rewriter.create<LLVM::GEPOp>(loc, ptrType, resultType, operands[0], ValueRange{operands[1]});

        // Non-temporal load for CXL.mem (device BAR4 mapped memory)
        auto loadResult = rewriter.create<LLVM::LoadOp>(loc, resultType, gep, /*alignment=*/0,
                                                        /*isVolatile=*/false, /*isNonTemporal=*/true);

        rewriter.replaceOp(op, loadResult.getResult());
        return success();
    }
};

/// Lower cira.type2.mem_store to GEP + LLVM StoreOp with non-temporal hint
struct Type2MemStoreOpLowering : public ConversionPattern {
    Type2MemStoreOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, Type2MemStoreOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op->getLoc();
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

        // GEP: dev_ptr + offset
        auto gep =
            rewriter.create<LLVM::GEPOp>(loc, ptrType, operands[0].getType(), operands[1], ValueRange{operands[2]});

        // Non-temporal store for CXL.mem (device BAR4 mapped memory)
        rewriter.create<LLVM::StoreOp>(loc, operands[0], gep,
                                       /*alignment=*/0,
                                       /*isVolatile=*/false,
                                       /*isNonTemporal=*/true);

        rewriter.eraseOp(op);
        return success();
    }
};

/// Lower cira.type2.drain to runtime call + memory fence
struct Type2DrainOpLowering : public ConversionPattern {
    Type2DrainOpLowering(LLVMTypeConverter &converter)
        : ConversionPattern(converter, Type2DrainOp::getOperationName(), 1, &converter.getContext()) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op->getLoc();
        auto *ctx = rewriter.getContext();

        // Call cira_type2_drain_delay_buffer(controller)
        // Controller is obtained from global state; emit call with null (runtime resolves)
        auto drainFunc = FlatSymbolRefAttr::get(ctx, "cira_type2_drain_delay_buffer");
        auto ptrType = LLVM::LLVMPointerType::get(ctx);
        auto nullCtrl = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
        rewriter.create<LLVM::CallOp>(loc, TypeRange{}, drainFunc, ValueRange{nullCtrl});

        // Full memory fence to ensure ordering
        rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::seq_cst);

        rewriter.eraseOp(op);
        return success();
    }
};

/// Lower cira.offload region - emit Vortex kernel launch or inline for x86
struct OffloadRegionOpLowering : public ConversionPattern {
    TargetArchitecture targetArch;

    OffloadRegionOpLowering(LLVMTypeConverter &converter, TargetArchitecture arch)
        : ConversionPattern(converter, OffloadRegionOp::getOperationName(), 1, &converter.getContext()),
          targetArch(arch) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        auto offloadOp = cast<OffloadRegionOp>(op);
        Location loc = op->getLoc();
        auto *ctx = rewriter.getContext();
        auto ptrType = LLVM::LLVMPointerType::get(ctx);
        auto i32Type = IntegerType::get(ctx, 32);
        auto i64Type = IntegerType::get(ctx, 64);

        if (targetArch == TargetArchitecture::RISCV_VORTEX || targetArch == TargetArchitecture::Heterogeneous) {
            // ---- Heterogeneous offload path ----
            // 1. Pack operands into MMIO task struct (matches Type2KernelRequest)
            // 2. Submit to ring buffer via BAR0 CSR writes
            // 3. Allocate completion cacheline (DCOH-coherent)
            // 4. Inline the body operations for x86 fallback / host-side stub
            // 5. On return, host polls completion cacheline

            // Allocate CompletionData (64-byte aligned, DCOH-delivered by device)
            auto completionAllocFunc = FlatSymbolRefAttr::get(ctx, "cira_future_alloc");
            auto completionPtr = rewriter.create<LLVM::CallOp>(loc, ptrType, completionAllocFunc, ValueRange{});

            // Initialize completion magic to 0 (not ready)
            auto zeroMagic = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));
            rewriter.create<LLVM::StoreOp>(loc, zeroMagic, completionPtr.getResult(), /*alignment=*/64);

            // Submit offload task to device via MMIO queue
            // cira_offload_submit(target_func_ptr, operands[], num_operands, completion_ptr)
            auto submitFunc = FlatSymbolRefAttr::get(ctx, "cira_offload_submit");

            // Pack operands into a stack-allocated array
            if (!operands.empty()) {
                auto numOps =
                    rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(operands.size()));
                auto one64 =
                    rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(operands.size()));

                // Allocate array of pointers for operands
                auto opsArray = rewriter.create<LLVM::AllocaOp>(loc, ptrType, ptrType, one64);

                // Store each operand pointer
                for (size_t i = 0; i < operands.size(); ++i) {
                    auto idx = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(i));
                    auto slot = rewriter.create<LLVM::GEPOp>(loc, ptrType, ptrType, opsArray, ValueRange{idx});
                    rewriter.create<LLVM::StoreOp>(loc, operands[i], slot);
                }

                // Get target function reference (if specified)
                Value targetFuncPtr;
                if (auto targetAttr = offloadOp.getTargetAttr()) {
                    auto funcRef = FlatSymbolRefAttr::get(ctx, targetAttr.getRootReference());
                    auto addrOfFunc = FlatSymbolRefAttr::get(ctx, "cira_get_device_func_addr");
                    targetFuncPtr = rewriter.create<LLVM::CallOp>(loc, ptrType, addrOfFunc, ValueRange{}).getResult();
                } else {
                    targetFuncPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
                }

                rewriter.create<LLVM::CallOp>(loc, TypeRange{}, submitFunc,
                                              ValueRange{targetFuncPtr, opsArray, numOps, completionPtr.getResult()});
            } else {
                // No operands - simple offload
                auto nullOps = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
                auto zeroOps = rewriter.create<LLVM::ConstantOp>(loc, i32Type, rewriter.getI32IntegerAttr(0));
                auto nullFunc = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
                rewriter.create<LLVM::CallOp>(loc, TypeRange{}, submitFunc,
                                              ValueRange{nullFunc, nullOps, zeroOps, completionPtr.getResult()});
            }

            // Also inline the body operations for the x86 host-side path
            // (host work that runs concurrently with device prefetching)
            Block &body = offloadOp.getBody().front();
            SmallVector<Value> results;

            IRMapping mapping;
            for (auto [arg, operand] : llvm::zip(body.getArguments(), operands))
                mapping.map(arg, operand);

            for (auto &bodyOp : llvm::make_early_inc_range(body.getOperations())) {
                if (auto yieldOp = dyn_cast<YieldOp>(&bodyOp)) {
                    for (auto val : yieldOp.getValues())
                        results.push_back(mapping.lookupOrDefault(val));
                } else {
                    rewriter.clone(bodyOp, mapping);
                }
            }

            rewriter.replaceOp(op, results);
        } else {
            // ---- x86-only fallback: inline everything ----
            Block &body = offloadOp.getBody().front();
            SmallVector<Value> results;

            IRMapping mapping;
            for (auto [arg, operand] : llvm::zip(body.getArguments(), operands))
                mapping.map(arg, operand);

            for (auto &bodyOp : llvm::make_early_inc_range(body.getOperations())) {
                if (auto yieldOp = dyn_cast<YieldOp>(&bodyOp)) {
                    for (auto val : yieldOp.getValues())
                        results.push_back(mapping.lookupOrDefault(val));
                } else {
                    rewriter.clone(bodyOp, mapping);
                }
            }

            rewriter.replaceOp(op, results);
        }

        return success();
    }
};

//===----------------------------------------------------------------------===//
// Helper function for populating conversion patterns
//===----------------------------------------------------------------------===//

static void addCiraLLVMTypeConversions(LLVMTypeConverter &converter) {
    converter.addConversion([](HandleType type) -> Type { return LLVM::LLVMPointerType::get(type.getContext()); });
    converter.addConversion([](FutureType type) -> Type { return LLVM::LLVMPointerType::get(type.getContext()); });
    converter.addConversion([](StreamType type) -> Type { return LLVM::LLVMPointerType::get(type.getContext()); });
}

static void populateCiraToLLVMConversionPatternsImpl(LLVMTypeConverter &converter, RewritePatternSet &patterns,
                                                     TargetArchitecture arch) {
    addCiraLLVMTypeConversions(converter);

    // Legacy patterns
    patterns.add<LoadEdgeOpLowering, EvictEdgeOpLowering>(converter, arch);
    patterns.add<LoadNodeOpLowering, GetPaddrOpLowering, CallOpLowering>(converter);

    // New CIRA operation patterns
    patterns.add<AllocCxlOpLowering>(converter);
    patterns.add<LoadAsyncOpLowering>(converter, arch);
    patterns.add<StoreAsyncOpLowering>(converter);
    patterns.add<FlushOpLowering>(converter);
    patterns.add<InstallCachelineOpLowering>(converter, arch);
    patterns.add<EvictHintOpLowering>(converter, arch);
    patterns.add<PrefetchStreamOpLowering>(converter, arch);
    patterns.add<PrefetchIndirectOpLowering>(converter, arch);
    patterns.add<FutureCreateOpLowering>(converter);
    patterns.add<FutureAwaitOpLowering>(converter, arch);
    patterns.add<BarrierOpLowering>(converter);
    patterns.add<ReleaseOpLowering>(converter);
    patterns.add<PhaseBoundaryOpLowering>(converter);
    patterns.add<SpeculateOpLowering>(converter);

    // Stream operation patterns
    patterns.add<StreamCreateIndirectOpLowering>(converter);
    patterns.add<PrefetchChainOpLowering>(converter, arch);
    patterns.add<PeekStreamOpLowering>(converter);
    patterns.add<AdvanceStreamOpLowering>(converter);

    // Control operation patterns
    patterns.add<OffloadStartOpLowering>(converter, arch);
    patterns.add<OffloadRegionOpLowering>(converter, arch);

    // CXL Type 2 operation patterns
    patterns.add<Type2CacheLoadOpLowering>(converter, arch);
    patterns.add<Type2CacheStoreOpLowering>(converter);
    patterns.add<Type2MemLoadOpLowering>(converter);
    patterns.add<Type2MemStoreOpLowering>(converter);
    patterns.add<Type2DrainOpLowering>(converter);

    // Legacy generic offload (renamed to offload_legacy)
    struct GenericOffloadOpLowering : public ConversionPattern {
        GenericOffloadOpLowering(LLVMTypeConverter &converter)
            : ConversionPattern(converter, OffloadOp::getOperationName(), 1, &converter.getContext()) {}
        LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                      ConversionPatternRewriter &rewriter) const override {
            rewriter.eraseOp(op);
            return success();
        }
    };
    patterns.add<GenericOffloadOpLowering>(converter);
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

// Specialized pass for x86 target
struct ConvertCiraToLLVMX86Pass : public PassWrapper<ConvertCiraToLLVMX86Pass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCiraToLLVMX86Pass)

    void getDependentDialects(DialectRegistry &registry) const override { registry.insert<LLVM::LLVMDialect>(); }

    StringRef getArgument() const final { return "convert-cira-to-llvm-x86"; }
    StringRef getDescription() const final { return "Convert Cira dialect to LLVM dialect for x86 target"; }

    void runOnOperation() override {
        auto module = getOperation();
        // Set target triple for x86
        module->setAttr("llvm.target_triple", StringAttr::get(&getContext(), "x86_64-unknown-linux-gnu"));

        LLVMTypeConverter converter(&getContext());
        RewritePatternSet patterns(&getContext());
        LLVMConversionTarget target(getContext());

        target.addIllegalDialect<RemoteMemDialect>();
        target.addLegalDialect<LLVM::LLVMDialect>();
        // Permit CIR ops to remain when only lowering CIRA pieces.
        target.addLegalDialect<cir::CIRDialect>();
        target.addLegalDialect<scf::SCFDialect>();
        target.addLegalDialect<cf::ControlFlowDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();
        target.addLegalDialect<memref::MemRefDialect>();

        populateCiraToLLVMConversionPatternsImpl(converter, patterns, TargetArchitecture::X86);

        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};

// Specialized pass for ARM target
struct ConvertCiraToLLVMARMPass : public PassWrapper<ConvertCiraToLLVMARMPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCiraToLLVMARMPass)

    void getDependentDialects(DialectRegistry &registry) const override { registry.insert<LLVM::LLVMDialect>(); }

    StringRef getArgument() const final { return "convert-cira-to-llvm-arm"; }
    StringRef getDescription() const final { return "Convert Cira dialect to LLVM dialect for ARM target"; }

    void runOnOperation() override {
        auto module = getOperation();
        // Set target triple for ARM
        module->setAttr("llvm.target_triple", StringAttr::get(&getContext(), "aarch64-unknown-linux-gnu"));

        LLVMTypeConverter converter(&getContext());
        RewritePatternSet patterns(&getContext());
        LLVMConversionTarget target(getContext());

        target.addIllegalDialect<RemoteMemDialect>();
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addLegalDialect<cir::CIRDialect>();
        target.addLegalDialect<scf::SCFDialect>();
        target.addLegalDialect<cf::ControlFlowDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();
        target.addLegalDialect<memref::MemRefDialect>();

        populateCiraToLLVMConversionPatternsImpl(converter, patterns, TargetArchitecture::ARM);

        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};

// Specialized pass for Vortex RISC-V SIMT target
struct ConvertCiraToLLVMVortexPass : public PassWrapper<ConvertCiraToLLVMVortexPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCiraToLLVMVortexPass)

    void getDependentDialects(DialectRegistry &registry) const override { registry.insert<LLVM::LLVMDialect>(); }

    StringRef getArgument() const final { return "convert-cira-to-llvm-vortex"; }
    StringRef getDescription() const final { return "Convert Cira dialect to NVVM IR for Vortex RISC-V SIMT target"; }

    void runOnOperation() override {
        auto module = getOperation();

        // Set target triple for NVVM (will be translated to RISC-V via CuPBoP)
        module->setAttr("llvm.target_triple", StringAttr::get(&getContext(), "nvptx64-nvidia-cuda"));

        // Mark module as Vortex target for backend processing
        module->setAttr("vortex.target", UnitAttr::get(&getContext()));

        // Set data layout for NVVM
        module->setAttr("llvm.data_layout",
                        StringAttr::get(&getContext(), "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                                                       "i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-"
                                                       "v64:64:64-v128:128:128-n16:32:64"));

        // Mark functions as CUDA kernels — detect both legacy and new CIRA ops
        module.walk([&](func::FuncOp func) {
            bool hasOffload = false;
            func.walk([&](Operation *op) {
                if (isa<LoadEdgeOp, LoadNodeOp, EvictEdgeOp, InstallCachelineOp, PrefetchChainOp, PrefetchIndirectOp,
                        PrefetchStreamOp, OffloadRegionOp>(op) ||
                    func->hasAttr("vortex.kernel")) {
                    hasOffload = true;
                    return WalkResult::interrupt();
                }
                return WalkResult::advance();
            });

            if (hasOffload) {
                func->setAttr("nvvm.kernel", UnitAttr::get(&getContext()));
                func->setAttr("nvvm.maxntid", DenseI32ArrayAttr::get(&getContext(), {256, 1, 1}));
            }
        });

        LLVMTypeConverter converter(&getContext());
        RewritePatternSet patterns(&getContext());
        LLVMConversionTarget target(getContext());

        target.addIllegalDialect<RemoteMemDialect>();
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addLegalDialect<cir::CIRDialect>();
        target.addLegalDialect<scf::SCFDialect>();
        target.addLegalDialect<cf::ControlFlowDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();
        target.addLegalDialect<memref::MemRefDialect>();

        populateCiraToLLVMConversionPatternsImpl(converter, patterns, TargetArchitecture::RISCV_VORTEX);

        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};

// Heterogeneous pass that partitions code between x86 host and Vortex device.
//
// For each cira.offload region:
//   1. Extract device-side operations (install_cacheline, prefetch_chain, etc.)
//      into a new func.func marked with "vortex.kernel" attribute
//   2. Replace the offload region in the host function with:
//      a. cira_future_alloc() for DCOH completion tracking
//      b. cira_offload_submit() to push task to MMIO ring buffer
//   3. Lower remaining host-side CIRA ops to x86 LLVM intrinsics
//   4. Lower device-side CIRA ops with RISCV_VORTEX target
struct ConvertCiraToLLVMHeteroPass : public PassWrapper<ConvertCiraToLLVMHeteroPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCiraToLLVMHeteroPass)

    void getDependentDialects(DialectRegistry &registry) const override { registry.insert<LLVM::LLVMDialect>(); }

    StringRef getArgument() const final { return "convert-cira-to-llvm-hetero"; }
    StringRef getDescription() const final {
        return "Convert Cira dialect to LLVM dialect for heterogeneous x86+Vortex execution";
    }

    void runOnOperation() override {
        auto module = getOperation();
        auto *ctx = &getContext();

        // ---- Phase 1: Extract device kernels from cira.offload regions ----
        int kernelId = 0;
        SmallVector<OffloadRegionOp> offloadOps;
        module.walk([&](OffloadRegionOp op) { offloadOps.push_back(op); });

        for (auto offloadOp : offloadOps) {
            // Generate device kernel function name
            std::string kernelName = "__vortex_kernel_" + std::to_string(kernelId++);
            if (auto regionName = offloadOp->getAttrOfType<StringAttr>("region_name"))
                kernelName = "__vortex_" + regionName.getValue().str();

            // Create a new function for the device kernel (will be compiled
            // separately with the Vortex toolchain to produce .vxbin)
            OpBuilder moduleBuilder(module.getBody(), module.getBody()->end());
            auto kernelFunc = moduleBuilder.create<func::FuncOp>(offloadOp.getLoc(), kernelName,
                                                                 moduleBuilder.getFunctionType({}, {}));

            // Mark as Vortex kernel
            kernelFunc->setAttr("vortex.kernel", UnitAttr::get(ctx));
            kernelFunc->setAttr("nvvm.kernel", UnitAttr::get(ctx));
            kernelFunc->setAttr("target-cpu", StringAttr::get(ctx, "riscv64-vortex"));

            // Copy access pattern metadata
            if (auto pattern = offloadOp->getAttr("cira.access_pattern"))
                kernelFunc->setAttr("cira.access_pattern", pattern);
            if (auto depth = offloadOp->getAttr("cira.chain_depth_estimate"))
                kernelFunc->setAttr("cira.chain_depth_estimate", depth);

            // Create entry block and clone offload body into it
            Block *entryBlock = kernelFunc.addEntryBlock();
            OpBuilder kernelBuilder(entryBlock, entryBlock->begin());

            // Clone operations from the offload body into the kernel
            Block &offloadBody = offloadOp.getBody().front();
            IRMapping mapping;
            for (auto &bodyOp : offloadBody.getOperations()) {
                if (!isa<YieldOp>(&bodyOp)) {
                    kernelBuilder.clone(bodyOp, mapping);
                }
            }
            kernelBuilder.create<func::ReturnOp>(offloadOp.getLoc());
        }

        // ---- Phase 2: Mark host functions for x86 ----
        module.walk([&](func::FuncOp func) {
            if (!func->hasAttr("vortex.kernel")) {
                func->setAttr("target-cpu", StringAttr::get(ctx, "x86-64"));
                func->setAttr("target-features", StringAttr::get(ctx, "+sse4.2,+avx2,+avx512f"));
            }
        });

        // ---- Phase 3: Lower CIRA operations ----
        // Host-side operations lower with Heterogeneous target (x86 prefetch +
        // MMIO offload submission). Device kernel functions lower separately
        // via vortex-kernel-gen pass.

        LLVMTypeConverter converter(ctx);
        RewritePatternSet patterns(ctx);
        LLVMConversionTarget target(*ctx);

        target.addIllegalDialect<RemoteMemDialect>();
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addLegalDialect<cir::CIRDialect>();
        target.addLegalDialect<scf::SCFDialect>();
        target.addLegalDialect<cf::ControlFlowDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();
        target.addLegalDialect<memref::MemRefDialect>();

        populateCiraToLLVMConversionPatternsImpl(converter, patterns, TargetArchitecture::Heterogeneous);

        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};

// Original pass with auto-detection
struct ConvertCiraToLLVMPass : public PassWrapper<ConvertCiraToLLVMPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCiraToLLVMPass)

    void getDependentDialects(DialectRegistry &registry) const override { registry.insert<LLVM::LLVMDialect>(); }

    StringRef getArgument() const final { return "convert-cira-to-llvm"; }
    StringRef getDescription() const final { return "Convert Cira dialect to LLVM dialect (auto-detect target)"; }

    void runOnOperation() override {
        auto module = getOperation();
        auto arch = getTargetArchitecture(module);

        LLVMTypeConverter converter(&getContext());
        RewritePatternSet patterns(&getContext());
        LLVMConversionTarget target(getContext());

        target.addIllegalDialect<RemoteMemDialect>();
        target.addLegalDialect<LLVM::LLVMDialect>();
        target.addLegalDialect<cir::CIRDialect>();
        target.addLegalDialect<scf::SCFDialect>();
        target.addLegalDialect<cf::ControlFlowDialect>();
        target.addLegalDialect<arith::ArithDialect>();
        target.addLegalDialect<func::FuncDialect>();
        target.addLegalDialect<memref::MemRefDialect>();

        populateCiraToLLVMConversionPatternsImpl(converter, patterns, arch);

        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};

} // namespace

void mlir::cira::populateCiraToLLVMConversionPatterns(LLVMTypeConverter &converter, RewritePatternSet &patterns) {
    // Default to heterogeneous if not specified
    populateCiraToLLVMConversionPatternsImpl(converter, patterns, TargetArchitecture::Heterogeneous);
}

std::unique_ptr<Pass> mlir::cira::createConvertCiraToLLVMPass() { return std::make_unique<ConvertCiraToLLVMPass>(); }

std::unique_ptr<Pass> mlir::cira::createConvertCiraToLLVMX86Pass() {
    return std::make_unique<ConvertCiraToLLVMX86Pass>();
}

std::unique_ptr<Pass> mlir::cira::createConvertCiraToLLVMARMPass() {
    return std::make_unique<ConvertCiraToLLVMARMPass>();
}

std::unique_ptr<Pass> mlir::cira::createConvertCiraToLLVMVortexPass() {
    return std::make_unique<ConvertCiraToLLVMVortexPass>();
}

std::unique_ptr<Pass> mlir::cira::createConvertCiraToLLVMHeteroPass() {
    return std::make_unique<ConvertCiraToLLVMHeteroPass>();
}
