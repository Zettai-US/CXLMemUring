//===- RegionFormation.cpp - Algorithm 1: static region formation ---------===//
//
// Walks every memory operation that touches CXL-resident data, forms a
// backward slice rooted at it, prunes what the device cannot run, and outlines
// the survivor into a `cira.offload` region followed by a `cira.barrier`.
//
//===----------------------------------------------------------------------===//

#include "Dialect/CiraOps.h"
#include "Dialect/RemoteMem.h"
#include "Dialect/Transforms/BackwardSlice.h"
#include "Dialect/Transforms/Passes.h"
#include "Dialect/TwoPassTimingAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "cira-region-formation"

namespace mlir {
#define GEN_PASS_DEF_CIRAREGIONFORMATION
#include "Dialect/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::cira;

namespace {

class CiraRegionFormationPass : public impl::CiraRegionFormationBase<CiraRegionFormationPass> {
public:
    using impl::CiraRegionFormationBase<CiraRegionFormationPass>::CiraRegionFormationBase;

    void runOnOperation() override {
        ModuleOp module = getOperation();
        unsigned regionId = 0;

        for (auto func : module.getOps<func::FuncOp>()) {
            if (func.isExternal())
                continue;

            // Collect roots first: the walk mutates the IR.
            SmallVector<Operation *> roots;
            func.walk([&](Operation *op) {
                if (op->getParentOfType<OffloadRegionOp>())
                    return;
                if (isRemoteMemoryAccess(op))
                    roots.push_back(op);
            });

            for (Operation *m : roots)
                if (formRegion(m, regionId))
                    ++regionId;
        }

        if (regionId == 0)
            LLVM_DEBUG(llvm::dbgs() << "no offloadable region found\n");
    }

private:
    /// One iteration of Algorithm 1 for a single remote access `m`.
    bool formRegion(Operation *m, unsigned regionId) {
        SliceOptions options;
        options.maxOps = maxSliceOps;
        options.followLoads = followDependentLoads;

        BackwardSliceResult slice = computeBackwardSlice(m, options);
        pruneUnsupported(slice);

        if (!isDeviceLegal(slice)) {
            LLVM_DEBUG(llvm::dbgs() << "slice not device-legal at " << *m << "\n");
            return false;
        }
        if (!hasRemoteDependence(slice)) {
            LLVM_DEBUG(llvm::dbgs() << "slice has no remote dependence at " << *m << "\n");
            return false;
        }
        if (requireProfitable && !OffloadCostModelParams::shouldOffload(slice.chainDepth)) {
            LLVM_DEBUG(llvm::dbgs() << "slice not profitable (depth " << slice.chainDepth << ") at " << *m << "\n");
            return false;
        }

        outlineAsDeviceKernel(slice, regionId);
        return true;
    }

    /// OutlineAsDeviceKernel + InsertOffload + InsertSync.
    ///
    /// The slice is cloned (not moved) into the offload body: the device runs
    /// the address-production chain ahead of the host, which still executes its
    /// own copy and consumes the warmed cache lines.
    void outlineAsDeviceKernel(const BackwardSliceResult &slice, unsigned regionId) {
        Operation *m = slice.root;
        OpBuilder builder(m);
        Location loc = m->getLoc();

        SmallVector<Value> liveIns(slice.liveIns.begin(), slice.liveIns.end());
        SmallVector<Type> liveInTypes;
        for (Value v : liveIns)
            liveInTypes.push_back(v.getType());

        // Insert before the first operation of the slice so the offload starts as
        // early as its inputs allow.
        Operation *insertionPoint = slice.ops.empty() ? m : slice.ops.front();
        if (insertionPoint->getBlock() != m->getBlock())
            insertionPoint = m;
        builder.setInsertionPoint(insertionPoint);

        auto offloadOp =
            builder.create<OffloadRegionOp>(loc, /*results=*/TypeRange{}, /*target=*/SymbolRefAttr(), liveIns);
        offloadOp->setAttr("cira.region_id", builder.getI32IntegerAttr(regionId));
        offloadOp->setAttr("cira.chain_depth", builder.getI32IntegerAttr(slice.chainDepth));
        offloadOp->setAttr("cira.slice_size", builder.getI32IntegerAttr(slice.ops.size()));

        SmallVector<Location> argLocs(liveIns.size(), loc);
        Block *body = builder.createBlock(&offloadOp.getBody(), {}, liveInTypes, argLocs);
        builder.setInsertionPointToStart(body);

        IRMapping mapping;
        for (auto [outer, arg] : llvm::zip(liveIns, body->getArguments()))
            mapping.map(outer, arg);
        rematerializeConstants(slice, builder, mapping);
        for (Operation *op : slice.ops)
            builder.clone(*op, mapping);
        // Replaying a store would duplicate the write; the device only needs the
        // address chain in that case.
        if (isMemoryRead(m))
            builder.clone(*m, mapping);
        builder.setInsertionPoint(m);
        builder.create<BarrierOp>(loc);
    }

    /// Constants are boundaries of the slice but are cheaper to recreate on the
    /// device than to ship as operands.
    void rematerializeConstants(const BackwardSliceResult &slice, OpBuilder &builder, IRMapping &mapping) {
        SetVector<Operation *> constants;
        auto collect = [&](ValueRange values) {
            for (Value operand : values) {
                Operation *def = operand.getDefiningOp();
                if (def && !mapping.contains(operand) && matchPattern(operand, m_Constant()))
                    constants.insert(def);
            }
        };
        for (Operation *op : slice.ops)
            collect(op->getOperands());
        collect(slice.root->getOperands());

        for (Operation *constant : constants)
            builder.clone(*constant, mapping);
    }
};

} // namespace
