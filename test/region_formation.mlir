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

// RUN: cira %s --cira-region-formation | FileCheck %s --check-prefix=PGO
// A three-deep chase clears the cost model's MIN_CHAIN_DEPTH, and the constant
// offset is rematerialized inside the region instead of being passed in.
// PGO-LABEL: func.func @chase
// PGO: cira.offload{{.*}}cira.chain_depth = 4
// PGO: arith.constant 4 : index
func.func @chase(%idx: memref<?xi64, 1>, %data: memref<?xf32, 1>, %i: index) -> f32 {
  %c4 = arith.constant 4 : index
  %a = memref.load %idx[%i] : memref<?xi64, 1>
  %ai = arith.index_cast %a : i64 to index
  %b = memref.load %idx[%ai] : memref<?xi64, 1>
  %bi = arith.index_cast %b : i64 to index
  %c = memref.load %idx[%bi] : memref<?xi64, 1>
  %ci = arith.index_cast %c : i64 to index
  %off = arith.addi %ci, %c4 : index
  %v = memref.load %data[%off] : memref<?xf32, 1>
  return %v : f32
}

