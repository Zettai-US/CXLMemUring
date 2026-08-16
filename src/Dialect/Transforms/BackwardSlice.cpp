//===- BackwardSlice.cpp - Static region formation for CIRA ---------------===//
//
// Implementation of Algorithm 2 (BackwardSlice) and its supporting predicates.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Transforms/BackwardSlice.h"
#include "Dialect/CiraOps.h"
#include "Dialect/RemoteMemRef.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/TypeSwitch.h"

#include "clang/CIR/Dialect/IR/CIRDialect.h"

using namespace mlir;
using namespace mlir::cira;

//===----------------------------------------------------------------------===//
// ClangIR (CIR) support
//===----------------------------------------------------------------------===//

/// Locals are `cir.alloca` slots. An access whose address does not come from a
/// stack slot goes to the heap or to a global, i.e. it is a candidate for
/// CXL-resident data.
static bool isCirStackSlot(Value addr) {
    Operation *def = addr.getDefiningOp();
    if (!def)
        return false;
    if (isa<::cir::AllocaOp>(def))
        return true;
    // Member/element addresses inherit the storage class of their base.
    if (auto member = dyn_cast<::cir::GetMemberOp>(def))
        return isCirStackSlot(member.getAddr());
    if (auto stride = dyn_cast<::cir::PtrStrideOp>(def))
        return isCirStackSlot(stride.getBase());
    if (auto base = dyn_cast<::cir::BaseClassAddrOp>(def))
        return isCirStackSlot(base.getDerivedAddr());
    if (auto cast = dyn_cast<::cir::CastOp>(def))
        return isCirStackSlot(cast.getSrc());
    return false;
}

//===----------------------------------------------------------------------===//
// Classification
//===----------------------------------------------------------------------===//

/// A memref/LLVM pointer in a non-default address space is treated as
/// CXL-resident; address space 0 is host DRAM.
static bool isRemoteAddressSpace(Type type) {
    if (auto memref = dyn_cast<MemRefType>(type)) {
        Attribute space = memref.getMemorySpace();
        if (!space)
            return false;
        if (auto intSpace = dyn_cast<IntegerAttr>(space))
            return intSpace.getInt() != 0;
        return true; // symbolic space, e.g. #cira.remote
    }
    if (auto ptr = dyn_cast<LLVM::LLVMPointerType>(type))
        return ptr.getAddressSpace() != 0;
    return false;
}

static bool isRemoteHandleType(Type type) { return isa<HandleType>(type) || isa<RemoteMemRefType>(type); }

bool mlir::cira::isRemoteMemoryAccess(Operation *op) {
    if (!op)
        return false;

    // Explicit annotation wins: lets PGO / frontend mark known-remote accesses.
    if (op->hasAttr("cira.remote"))
        return true;

    if (isa<LoadAsyncOp, StoreAsyncOp, InstallCachelineOp>(op))
        return true;

    for (Value operand : op->getOperands())
        if (isRemoteHandleType(operand.getType()))
            return isMemoryRead(op) || !isMemoryEffectFree(op);

    if (isa<memref::LoadOp, memref::StoreOp, LLVM::LoadOp, LLVM::StoreOp, affine::AffineLoadOp, affine::AffineStoreOp>(
            op)) {
        for (Value operand : op->getOperands())
            if (isRemoteAddressSpace(operand.getType()))
                return true;
    }

    if (auto load = dyn_cast<::cir::LoadOp>(op))
        return !isCirStackSlot(load.getAddr());
    if (auto store = dyn_cast<::cir::StoreOp>(op))
        return !isCirStackSlot(store.getAddr());
    return false;
}

SmallVector<Value> mlir::cira::getAddressOperands(Operation *op) {
    SmallVector<Value> addresses;
    if (!op)
        return addresses;

    if (auto load = dyn_cast<LoadAsyncOp>(op)) {
        addresses.push_back(load.getPtr());
        if (load.getIndex())
            addresses.push_back(load.getIndex());
        return addresses;
    }
    if (auto store = dyn_cast<StoreAsyncOp>(op)) {
        addresses.push_back(store.getPtr());
        if (store.getIndex())
            addresses.push_back(store.getIndex());
        return addresses;
    }
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
        addresses.push_back(load.getMemRef());
        llvm::append_range(addresses, load.getIndices());
        return addresses;
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
        addresses.push_back(store.getMemRef());
        llvm::append_range(addresses, store.getIndices());
        return addresses;
    }
    if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
        addresses.push_back(load.getMemRef());
        llvm::append_range(addresses, load.getIndices());
        return addresses;
    }
    if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
        addresses.push_back(store.getMemRef());
        llvm::append_range(addresses, store.getIndices());
        return addresses;
    }
    if (auto load = dyn_cast<LLVM::LoadOp>(op)) {
        addresses.push_back(load.getAddr());
        return addresses;
    }
    if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
        addresses.push_back(store.getAddr());
        return addresses;
    }
    if (auto load = dyn_cast<::cir::LoadOp>(op)) {
        addresses.push_back(load.getAddr());
        return addresses;
    }
    if (auto store = dyn_cast<::cir::StoreOp>(op)) {
        addresses.push_back(store.getAddr());
        return addresses;
    }

    // Generic fallback: any pointer-like operand.
    for (Value operand : op->getOperands())
        if (isRemoteHandleType(operand.getType()) || isRemoteAddressSpace(operand.getType()))
            addresses.push_back(operand);
    return addresses;
}

bool mlir::cira::isMemoryRead(Operation *op) {
    if (!op)
        return false;
    if (isa<LoadAsyncOp, memref::LoadOp, affine::AffineLoadOp, LLVM::LoadOp, ::cir::LoadOp>(op))
        return true;
    if (auto effects = dyn_cast<MemoryEffectOpInterface>(op))
        return effects.hasEffect<MemoryEffects::Read>();
    return false;
}

bool mlir::cira::isSupportedSliceOp(Operation *op) {
    if (!op)
        return false;

    // Pure arithmetic, comparison, selection and casts.
    if (isa<arith::ArithDialect>(op->getDialect()))
        return isMemoryEffectFree(op);
    if (isa<affine::AffineDialect>(op->getDialect()))
        return isa<affine::AffineApplyOp, affine::AffineMinOp, affine::AffineMaxOp, affine::AffineLoadOp>(op);
    if (isa<LLVM::GEPOp, LLVM::PtrToIntOp, LLVM::IntToPtrOp, LLVM::BitcastOp, LLVM::AddrSpaceCastOp, LLVM::ZExtOp,
            LLVM::SExtOp, LLVM::TruncOp, LLVM::LoadOp>(op))
        return true;
    if (isa<memref::LoadOp, memref::SubViewOp, memref::ReinterpretCastOp, memref::CastOp, memref::DimOp>(op))
        return true;
    if (isa<LoadAsyncOp, FutureAwaitOp, GetPaddrOp>(op))
        return true;
    // CIR address arithmetic and pure value computation.
    if (isa<::cir::LoadOp, ::cir::PtrStrideOp, ::cir::GetMemberOp, ::cir::BaseClassAddrOp, ::cir::CastOp,
            ::cir::ConstantOp, ::cir::BinOp, ::cir::UnaryOp, ::cir::CmpOp, ::cir::SelectOp, ::cir::GetGlobalOp>(op))
        return true;
    return false;
}

bool mlir::cira::isHostOnlyOp(Operation *op) {
    if (!op)
        return false;
    if (op->hasAttr("cira.host_only"))
        return true;
    if (isa<func::CallOp, LLVM::CallOp, func::CallIndirectOp, ::cir::CallOp>(op))
        return true;
    if (isa<memref::AllocOp, memref::AllocaOp, memref::DeallocOp, LLVM::AllocaOp, ::cir::AllocaOp>(op))
        return true;
    // Anything that writes or has unmodelled effects stays on the host, unless
    // it is one of the CIRA ops we explicitly support.
    if (isSupportedSliceOp(op))
        return false;
    return !isMemoryEffectFree(op);
}

bool mlir::cira::isDeviceLegalOp(Operation *op) { return isSupportedSliceOp(op) && !isHostOnlyOp(op); }

StringRef mlir::cira::getBoundaryReasonName(SliceBoundaryReason reason) {
    switch (reason) {
    case SliceBoundaryReason::Constant:
        return "constant";
    case SliceBoundaryReason::LoopInvariantBase:
        return "loop-invariant-base";
    case SliceBoundaryReason::BlockArgument:
        return "block-argument";
    case SliceBoundaryReason::HostOnly:
        return "host-only";
    case SliceBoundaryReason::SideEffecting:
        return "side-effecting";
    case SliceBoundaryReason::Unsupported:
        return "unsupported";
    case SliceBoundaryReason::DepthLimit:
        return "depth-limit";
    }
    return "unknown";
}

bool BackwardSliceResult::contains(Operation *op) const { return llvm::is_contained(ops, op); }

//===----------------------------------------------------------------------===//
// Algorithm 2: BackwardSlice(m)
//===----------------------------------------------------------------------===//

namespace {

/// Worklist-driven backward walk over the def-use graph.
class SliceBuilder {
public:
    SliceBuilder(Operation *root, const SliceOptions &options) : options(options) {
        result.root = root;
        scope = options.scope;
        if (!scope) {
            if (auto *parent = root->getParentOp())
                scope = parent->getParentRegion() ? root->getParentRegion() : nullptr;
            else
                scope = root->getParentRegion();
        }
    }

    BackwardSliceResult build() {
        for (Value address : getAddressOperands(result.root))
            worklist.push_back(address);

        while (!worklist.empty()) {
            Value v = worklist.pop_back_val();
            if (!visited.insert(v).second)
                continue;
            visit(v);
        }

        orderSlice();
        result.chainDepth = computeChainDepth();
        return std::move(result);
    }

private:
    /// A value defined outside the slicing scope is a loop-invariant base.
    bool isOutOfScope(Value v) const {
        if (!scope)
            return false;
        Region *definingRegion = v.getParentRegion();
        return definingRegion && !scope->isAncestor(definingRegion);
    }

    void addBoundary(Value v, Operation *def, SliceBoundaryReason reason) {
        result.boundaries.push_back({v, def, reason});
        // Constants are rematerialized on the device, everything else has to be
        // shipped across the interface.
        if (reason != SliceBoundaryReason::Constant)
            result.liveIns.insert(v);
    }

    void visit(Value v) {
        if (matchPattern(v, m_Constant())) {
            addBoundary(v, v.getDefiningOp(), SliceBoundaryReason::Constant);
            return;
        }
        if (isOutOfScope(v)) {
            addBoundary(v, v.getDefiningOp(), SliceBoundaryReason::LoopInvariantBase);
            return;
        }

        Operation *def = v.getDefiningOp();
        if (!def) {
            visitBlockArgument(cast<BlockArgument>(v));
            return;
        }

        if (isHostOnlyOp(def)) {
            addBoundary(v, def,
                        isMemoryEffectFree(def) ? SliceBoundaryReason::HostOnly : SliceBoundaryReason::SideEffecting);
            return;
        }
        if (!isSupportedSliceOp(def)) {
            addBoundary(v, def, SliceBoundaryReason::Unsupported);
            return;
        }
        if (isMemoryRead(def) && !options.followLoads) {
            addBoundary(v, def, SliceBoundaryReason::DepthLimit);
            return;
        }
        if (sliceOps.size() >= options.maxOps) {
            result.truncated = true;
            addBoundary(v, def, SliceBoundaryReason::DepthLimit);
            return;
        }

        sliceOps.insert(def);
        for (Value operand : def->getOperands())
            worklist.push_back(operand);
    }

    /// Loop-carried values behave like phi nodes: follow both the initial value
    /// and the value yielded by the loop body.
    void visitBlockArgument(BlockArgument arg) {
        Operation *owner = arg.getOwner()->getParentOp();
        if (!options.followLoopCarried || !owner) {
            addBoundary(arg, nullptr, SliceBoundaryReason::BlockArgument);
            return;
        }

        if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
            if (arg == forOp.getInductionVar()) {
                // The induction variable is regenerated on the device from the loop
                // bounds; the bounds themselves become live-ins.
                worklist.push_back(forOp.getLowerBound());
                worklist.push_back(forOp.getUpperBound());
                worklist.push_back(forOp.getStep());
                return;
            }
            if (auto init = forOp.getTiedLoopInit(arg)) {
                worklist.push_back(init->get());
                if (auto yielded = forOp.getTiedLoopYieldedValue(arg))
                    worklist.push_back(yielded->get());
                return;
            }
        }

        if (auto whileOp = dyn_cast<scf::WhileOp>(owner)) {
            unsigned idx = arg.getArgNumber();
            if (arg.getOwner() == whileOp.getBeforeBody()) {
                if (idx < whileOp.getInits().size())
                    worklist.push_back(whileOp.getInits()[idx]);
                auto yieldOp = whileOp.getYieldOp();
                if (idx < yieldOp.getResults().size())
                    worklist.push_back(yieldOp.getResults()[idx]);
                return;
            }
            auto condOp = whileOp.getConditionOp();
            if (idx < condOp.getArgs().size()) {
                worklist.push_back(condOp.getArgs()[idx]);
                return;
            }
        }

        addBoundary(arg, nullptr, SliceBoundaryReason::BlockArgument);
    }

    /// Emit the slice in program order so it can be cloned without forward
    /// references.
    void orderSlice() {
        if (sliceOps.empty())
            return;
        Operation *top = result.root;
        while (top->getParentOp() && !isa<FunctionOpInterface>(top))
            top = top->getParentOp();
        top->walk<WalkOrder::PreOrder>([&](Operation *op) {
            if (sliceOps.contains(op))
                result.ops.push_back(op);
        });
        // Fallback for slices rooted outside a function body.
        if (result.ops.size() != sliceOps.size()) {
            result.ops.assign(sliceOps.begin(), sliceOps.end());
        }
    }

    /// Longest path of dependent loads, i.e. the pointer-chasing depth.
    unsigned computeChainDepth() {
        DenseMap<Operation *, unsigned> depth;
        unsigned maxDepth = 0;
        for (Operation *op : result.ops) {
            unsigned incoming = 0;
            for (Value operand : op->getOperands())
                if (Operation *def = operand.getDefiningOp())
                    incoming = std::max(incoming, depth.lookup(def));
            unsigned current = incoming + (isMemoryRead(op) ? 1 : 0);
            depth[op] = current;
            maxDepth = std::max(maxDepth, current);
        }
        // The rooting access itself closes the chain.
        return maxDepth + 1;
    }

    SliceOptions options;
    Region *scope = nullptr;
    BackwardSliceResult result;
    SmallVector<Value> worklist;
    DenseSet<Value> visited;
    SetVector<Operation *> sliceOps;
};

} // namespace

BackwardSliceResult mlir::cira::computeBackwardSlice(Operation *m, const SliceOptions &opts) {
    BackwardSliceResult empty;
    if (!m)
        return empty;

    SliceOptions options = opts;
    if (!options.scope) {
        if (auto func = m->getParentOfType<FunctionOpInterface>())
            options.scope = &func.getFunctionBody();
        else
            options.scope = m->getParentRegion();
    }
    return SliceBuilder(m, options).build();
}

void mlir::cira::pruneUnsupported(BackwardSliceResult &slice) {
    bool changed = true;
    while (changed) {
        changed = false;
        SmallVector<Operation *> kept;
        DenseSet<Operation *> keptSet;
        for (Operation *op : slice.ops)
            if (isDeviceLegalOp(op))
                keptSet.insert(op);

        for (Operation *op : slice.ops) {
            if (!keptSet.contains(op)) {
                for (Value result : op->getResults())
                    slice.liveIns.insert(result);
                slice.boundaries.push_back(
                    {op->getNumResults() ? op->getResult(0) : Value(), op, SliceBoundaryReason::Unsupported});
                changed = true;
                continue;
            }
            kept.push_back(op);
        }
        slice.ops = std::move(kept);
    }

    // Live-ins are exactly the non-constant operands the surviving slice and the
    // rooting access consume from outside the region. Constants are
    // rematerialized by the outliner rather than shipped across the interface.
    slice.liveIns.clear();
    DenseSet<Operation *> sliceSet(slice.ops.begin(), slice.ops.end());
    auto addExternalOperands = [&](ValueRange values) {
        for (Value operand : values) {
            Operation *def = operand.getDefiningOp();
            if (def && sliceSet.contains(def))
                continue;
            if (matchPattern(operand, m_Constant()))
                continue;
            slice.liveIns.insert(operand);
        }
    };
    for (Operation *op : slice.ops)
        addExternalOperands(op->getOperands());
    if (slice.root) {
        // Stores are not replayed on the device, only their address chain.
        if (isMemoryRead(slice.root))
            addExternalOperands(slice.root->getOperands());
        else
            addExternalOperands(getAddressOperands(slice.root));
    }
}

bool mlir::cira::isDeviceLegal(const BackwardSliceResult &slice) {
    if (slice.ops.empty())
        return false;
    return llvm::all_of(slice.ops, isDeviceLegalOp);
}

bool mlir::cira::hasRemoteDependence(const BackwardSliceResult &slice) { return llvm::any_of(slice.ops, isMemoryRead); }
