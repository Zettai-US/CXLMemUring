// RUN: cira %s --cira-region-formation="require-profitable=false" | FileCheck %s

// Indirect (gather) access: the load of the index feeds the address of the
// remote data load, so the backward slice must capture it.
// CHECK-LABEL: func.func @gather
// CHECK: cira.offload
// CHECK: cira.barrier
func.func @gather(%idx: memref<?xi64, 1>, %data: memref<?xf32, 1>, %n: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (f32) {
    %e = memref.load %idx[%i] : memref<?xi64, 1>
    %ei = arith.index_cast %e : i64 to index
    %v = memref.load %data[%ei] : memref<?xf32, 1>
    %s = arith.addf %acc, %v : f32
    scf.yield %s : f32
  }
  return %r : f32
}

// A direct, non-indirect access has no remote dependence in its slice and must
// not be offloaded.
// CHECK-LABEL: func.func @direct
// CHECK-NOT: cira.offload
func.func @direct(%data: memref<?xf32, 1>, %n: index) -> f32 {
  %c0 = arith.constant 0 : index
  %v = memref.load %data[%c0] : memref<?xf32, 1>
  return %v : f32
}
