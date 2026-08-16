//===- BackwardSlice.h - Static region formation for CIRA -----------------===//
//
// Implements the static region-formation analysis described in the CIRA paper
// (Algorithms 1 and 2): starting from a memory operation that touches
// CXL-resident data, walk backward through the def-use graph collecting the
// operations that produce the address, stopping at constants, loop-invariant
// bases, and host-only / side-effecting operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRA_BACKWARD_SLICE_H
#define CIRA_BACKWARD_SLICE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace cira {

/// Why the backward traversal stopped at a particular value.
enum class SliceBoundaryReason {
    Constant, // literal, folds on the device
    LoopInvariantBase, // defined outside the slicing scope
    BlockArgument, // region/loop argument that is not a tracked phi
    HostOnly, // op the device cannot execute
    SideEffecting, // op whose effects must stay on the host
    Unsupported, // op kind not modelled by the device ISA
    DepthLimit // slice grew past the configured budget
};

struct SliceBoundary {
    Value value;
    Operation *definingOp = nullptr; // null for block arguments / constants
    SliceBoundaryReason reason = SliceBoundaryReason::Constant;
};

struct SliceOptions {
    /// Restrict the slice to this region. Values defined outside it are treated
    /// as loop-invariant bases. Defaults to the enclosing function body.
    Region *scope = nullptr;
    /// Follow loop-carried values (scf iter_args) as phi nodes.
    bool followLoopCarried = true;
    /// Include dependent loads in the slice (pointer chasing). When false the
    /// slice stops at the first indirection.
    bool followLoads = true;
    /// Upper bound on the number of operations pulled into a slice.
    unsigned maxOps = 256;
};

struct BackwardSliceResult {
    /// The rooting memory operation `m`.
    Operation *root = nullptr;
    /// Operations forming the slice, in program (topological) order.
    SmallVector<Operation *> ops;
    /// Values that must be supplied from the host to run the slice.
    SetVector<Value> liveIns;
    /// Where the traversal stopped.
    SmallVector<SliceBoundary> boundaries;
    /// Longest chain of dependent memory reads in the slice. This is the
    /// quantity the cost model compares against MIN_CHAIN_DEPTH.
    unsigned chainDepth = 0;
    /// Set to true when the slice was truncated by `maxOps`.
    bool truncated = false;

    bool contains(Operation *op) const;
    bool empty() const { return ops.empty(); }
};

//===----------------------------------------------------------------------===//
// Operation classification
//===----------------------------------------------------------------------===//

/// True if `op` reads or writes data that lives in CXL-attached memory: a
/// `cira` operation on a `!cira.handle`, a memref/LLVM access through a
/// non-default (remote) address space, or an op explicitly tagged
/// `cira.remote`.
bool isRemoteMemoryAccess(Operation *op);

/// The operands of `op` that participate in address computation. Empty when
/// `op` is not a memory operation.
SmallVector<Value> getAddressOperands(Operation *op);

/// True if `op` is a memory read (used for dependence-chain depth).
bool isMemoryRead(Operation *op);

/// True if `op` is pure arithmetic / compare / GEP / select / cast, i.e. the
/// "supported" class in Algorithm 2.
bool isSupportedSliceOp(Operation *op);

/// True if the near-memory device can execute `op`.
bool isDeviceLegalOp(Operation *op);

/// True if `op` must remain on the host (unmodelled calls, allocation,
/// I/O, or arbitrary side effects).
bool isHostOnlyOp(Operation *op);

//===----------------------------------------------------------------------===//
// Algorithm 2: BackwardSlice(m)
//===----------------------------------------------------------------------===//

BackwardSliceResult computeBackwardSlice(Operation *m, const SliceOptions &options = {});

/// Drop operations the device cannot run, promoting their results to live-ins.
/// Iterates to a fixed point because pruning can orphan further operations.
void pruneUnsupported(BackwardSliceResult &slice);

/// A slice is device-legal when it is non-empty and every remaining operation
/// can run on the device.
bool isDeviceLegal(const BackwardSliceResult &slice);

/// A slice is worth offloading only if it actually chases memory: it must
/// contain at least one load feeding the root address.
bool hasRemoteDependence(const BackwardSliceResult &slice);

StringRef getBoundaryReasonName(SliceBoundaryReason reason);

} // namespace cira
} // namespace mlir

#endif // CIRA_BACKWARD_SLICE_H
