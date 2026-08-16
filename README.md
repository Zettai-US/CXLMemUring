# CIRA / CXLMemUring

Compiler-driven heterogeneous execution for CXL-attached memory.

CIRA takes a program whose bottleneck is remote-memory stalls, finds the regions
where a dependent load chain stalls the host, and moves that chain onto a
near-memory RISC-V SIMT core on a CXL Type-2 device. The host keeps doing
independent work while the device walks the pointers and publishes results back
into a host-visible cache line.

This repository holds the software half: the MLIR compiler that forms and lowers
offload regions, and the host runtime that dispatches them. The FPGA design that
executes them is a separate project.

```
   program
      |  MLIR: cira dialect -- region formation, offload lowering   (src/, include/)
      v
   offload template (LLVM IR, knobs left as sentinel globals)
      |  runtime: cost model picks knobs, ORC folds them in         (runtime/)
      v
   doorbell in the CXL Type-2 control window
      |  device: validate, launch, republish status                 (separate FPGA project)
      v
   Vortex RV64 SIMT core, near the memory
```

## Layout

| Path | What it is |
| --- | --- |
| `include/`, `src/` | The `cira` MLIR dialect, analyses, and lowering passes. `src/driver.cpp` registers everything. |
| `runtime/` | Host runtime: cost model, ORC specializer, CXL MMIO window, cache-resident wait, Vortex device interface. |
| `driver/` | Linux kernel bits: CXL Type-2 NUMA node setup, RCRB access. |
| `bench/` | Benchmarks (GAPBS, MCF, MonetDB, DataFrame, llama.cpp, NPB, Spatter). |
| `test/`, `tests/` | Dialect and lowering tests. |
| `scripts/` | Experiment and plotting helpers. |

## The offload contract

One header defines the host/device wire format, and both sides include it:
[runtime/include/cira_cxl_job.h](runtime/include/cira_cxl_job.h).

```
0x0000  doorbell     magic / version / job_id / flags / status / seq
0x0100  arg slots    5 x 1 KB, one per job id: header then payload
0x1E00  kernel table device entry point per job id (RTL extension)
0x1F20  status       magic / version / job_id / status / seq
0x2000  end of the control window
```

The host stages the payload, then the slot header, then commits the doorbell
with `seq` written last. The device polls `seq`, validates magic/version and
that the slot matches the doorbell, launches the core, and republishes the
record with `seq` last so the host never reads a torn result.

Job ids are `NOP`, `INSTALL_CACHELINE`, `PREFETCH_CHAIN`, `STREAM_PREFETCH` and
`CALL`, mirroring the `cira.*` operations the compiler emits.

Completion reaches the host on two channels, both already supported by the
runtime:

- the status line at `0x1F20`, polled with `cira_mmio_wait_seq()`; and
- a 64-byte completion line written by the offloaded kernel to an address named
  in the job, polled with `cira_mmio_wait_completion()`. That poll is the
  cache-resident wait: a bounded `PAUSE` spin, then `UMONITOR`/`UMWAIT`,
  `TPAUSE`, `MONITORX`/`MWAITX` or `WFE`, chosen at run time from CPUID.

## Building

### Compiler

Needs LLVM/MLIR. See the root `CMakeLists.txt` for the exact version pin.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces the `cira` driver, which runs the dialect passes:

```sh
./build/bin/cira test/region_formation.mlir --cira-region-formation
```

### Runtime

LLVM and the Vortex SDK are both optional. Without LLVM the ORC specializer is
skipped and the rest still builds; without the Vortex SDK the device side falls
back to simulation stubs.

```sh
cmake -S runtime -B build/runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build/runtime -j
ctest --test-dir build/runtime --output-on-failure
```

The suite runs anywhere: `test_cira_offload_path` exercises the whole
submit/doorbell/completion protocol against a control window backed by
anonymous memory (`CIRA_CXL_MMIO_EMULATE=1`), so no FPGA is required.

See [runtime/README.md](runtime/README.md) for the runtime's internals and the
`CIRA_CXL_MMIO_*` / `CIRA_WAIT_BACKEND` environment variables.

## Status

Working: dialect and lowering passes; the host runtime including the cost model,
ORC specializer, MMIO window and wait backends. The runtime suite runs anywhere,
against an emulated control window.

The device side lives in a separate FPGA project (Agilex 7 R-tile CXL Type-2
with a Vortex RV64 core). It implements the same `cira_cxl_job.h` contract, so
the runtime talks to it unchanged once `CIRA_CXL_MMIO_PATH` points at the
device's control window.
