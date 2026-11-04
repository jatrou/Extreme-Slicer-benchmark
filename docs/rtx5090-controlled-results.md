# RTX 5090 controlled benchmark results (2026-11-04)

These measurements exercise Extreme Slicer's CUDA vertical-infill-intersection
path on a synthetic, reproducible model. They show correct results for that
path and **no measured whole-process acceleration** on this fixture at batch
size 64. They do not establish performance or correctness for other models,
printer profiles, GPU paths, or a complete print. The benchmark method and
runner are described in [rtx5090-benchmark.md](rtx5090-benchmark.md); the
implementation scope is documented in
[gpu-benchmark-code-review.md](gpu-benchmark-code-review.md).

## Provenance and setup

- Source base: `mk0000001/Extreme-Slicer` commit
  `0bf0c192484aa0c83c1824f72b32a4c8edd99b0b`. The measured source also
  includes `b8d3cda91f927cc0eadb4755c9aff142b26bb176` (CUDA 13 Windows
  compiler flag fix) and `657a167741144b11cd31f2bb33d088cbe343cbd6`
  (CLI benchmark instrumentation, console shim, runner, standalone harness).
  The later `bc446e8e6afe779db77aa7180848f682e0ed168f` adds source review
  documentation only. The build fix changes CMake flags; these patches do not
  change the CUDA intersection kernel.
- Measured Orca engine DLL SHA-256:
  `cfc30370e2052e4bc7487746de8cd12bda4ff2cfb35990822aced89682257899`.
  Console shim SHA-256:
  `b374eb22f528637d97e4bd206ff7f2cc621df6f099b2db4c495bbff5e14969fc`.
  The strict and throughput manifests record the same hashes. The standalone
  microbenchmark results do not contain an executable hash.
- Windows 11 Pro build 26200; Intel Core i9-14900K (24 cores, 32 logical
  processors); NVIDIA GeForce RTX 5090 (32,607 MiB reported), driver 617.42;
  CUDA toolkit 13.1.115; MSVC 19.44.35229; CMake 3.31.6. The hardware
  inventory was captured later on the same date, so it is a system record,
  not a simultaneous telemetry trace. The recorded power scheme was Ultimate
  Performance.
- Out-of-source, Release x64 Visual Studio 2022 build with
  `SLIC3R_ENABLE_CUDA_SLICER=ON`, `SLIC3R_ENABLE_VULKAN_SLICER=OFF`,
  `CMAKE_CUDA_ARCHITECTURES=120`, `BUILD_TESTS=ON`, and `ORCA_TOOLS=OFF`.
  CPU and CUDA CLI runs used the **same** engine DLL and fresh data/output
  directories. The environment selected `ORCA_BENCH_BACKEND=cpu|cuda`,
  `ORCA_BENCH_STRICT=0|1`, and batch size 64. No preset overrides were applied
  to the controlled fixture.

## Controlled full-slice fixture

The runner generated a closed, corrugated cylinder: 124,800 STL triangles,
up to 90.4 mm across and 20 mm tall. It resolved bundled X1C machine and PLA
filament presets, set 0.2 mm layers, 100% density, and rectilinear sparse,
internal solid, top, and bottom fill patterns. The fixture STL SHA-256 was
`64a1fe815714ae2a4e6b32e26ea532bd1be246b651f3937b97d3c8ffe5064c99`.
This is a controlled geometry and profile, not a printer-scale workload
representative of every full-density model.

`process_wall_ms` measures `Print::process()` only. Total wall time includes
startup, loading, processing, export, and process shutdown. The runner's wall
clock stops when the slicer process exits; resource sampling does not add a
one-second polling interval. The table below uses **five measured runs per
backend after one warmup per backend**, in alternating CPU/CUDA order.

| Metric | CPU, median (range) | CUDA, median (range) |
| --- | ---: | ---: |
| `Print::process()` | 680.31 ms (660.34–772.18) | 680.76 ms (659.93–689.26) |
| Total CLI wall | 1,423.00 ms (1,392.48–1,568.71) | 1,530.25 ms (1,452.90–1,541.41) |

The process median ratio is CPU/CUDA = **0.9993×**. It divides the two
backend medians; it is not the median of five paired ratios. Those paired
ratios range from 0.977× to 1.134×, so small apparent advantages within an
individual pair are not stable. The total-wall medians favor CPU on this
fixture. CUDA executed 100 dispatches and accepted all 106,562 submitted
intersections in each run; 399 sampled CPU checks per CUDA run found zero
failures. The CUDA `total_gpu_ms` median was 24.01 ms and `total_host_ms`
median 149.86 ms. The former is host-clock launch/synchronization time, not
pure GPU kernel time; neither subtotal measures a full slice.

For a separate **strict correctness pass**, one CPU and one CUDA run both
completed. CUDA accepted 106,562 intersections and recomputed every one on
CPU, with zero validation failures. Their process times were 662.56 and
672.67 ms respectively. Strict validation adds CPU work, so those times are
not a throughput comparison.

An independent, streaming rehash checked every saved G-code output. The 12
throughput outputs had 12 distinct raw SHA-256 values and one identical
normalized SHA-256 value. The strict CPU/CUDA pair had two distinct raw hashes
and the same normalized value:
`2b46fc577eead14d4fb09a63e0d5053bc3a57749aabc7f6cb2cdd398908ba70d`.
Normalization removes only the known generation timestamp and the varying
numeric object ID in `printing object` comments; it preserves generator and
version, object name and copy number, other comments, and all printer
commands. Two earlier CPU-only repeat outputs also match after applying this
current rule. Their originally recorded normalized hashes differ because that
older runner version had not yet normalized object IDs. Matching normalized
G-code is a useful output check, not proof that every possible GPU behavior
has been validated.

## Standalone printer-scale dispatch check

Two separate runs of `tools/benchmark_cuda_scanlines.cpp` used deterministic
vertical-intersection requests spanning a 300 mm plate. Each row below is a
**median of 21 repetitions within each run**; the paired ranges are the two
run medians, not confidence intervals over 42 raw samples. The harness checks
every CUDA result against the CPU reference after each timed dispatch, and
the two runs' checksums agree. A cold, first 64-request CUDA dispatch took
64.87 and 67.01 ms, including runtime/device initialization.

| Requests | CPU per pass, ms (two runs) | Full CUDA dispatch, ms (two runs) | CUDA/CPU latency (two runs) |
| ---: | ---: | ---: | ---: |
| 64 | 0.000286, 0.000286 | 0.743, 0.756 | 2,595×, 2,647× |
| 512 | 0.002306, 0.002315 | 0.771, 0.792 | 334×, 342× |
| 4,096 | 0.018387, 0.018463 | 0.886, 0.905 | 48.2×, 49.0× |
| 16,384 | 0.074352, 0.076065 | 1.222, 1.296 | 16.4×, 17.0× |
| 65,536 | 0.301026, 0.296967 | 2.901, 2.965 | 9.64×, 9.98× |
| 262,144 | 1.313593, 1.308053 | 9.191, 9.208 | 7.00×, 7.04× |

Here, a ratio above one means the complete CUDA dispatch was slower than the
standalone CPU arithmetic. The CPU path writes into a preallocated vector;
the CUDA number includes production request packing, allocation, copies,
launch/synchronization, free, and result materialization. These are synthetic
requests at realistic coordinate scale, **not** a full slicer or a replay of
captured production requests. The small-batch overhead helps explain why a
fast GPU kernel need not accelerate the controlled full slice.

## Scope and exclusions

The data support a correctness result for the exercised vertical-intersection
path and a near-tie in whole-process time for one controlled fixture. A
different object or infill workload can change dispatch sizes and the share
of CPU work. This note makes **no real-project generalization** and **no GPU
comparison against an unmodified stock slicer**. Incomplete or failed CLI
runs, including one that finished `Print::process()` but failed before usable
G-code export, are excluded from all timing medians and correctness claims.
The separate source review documents additional GPU paths and their open
correctness questions; the scanline validation here does not cover them.
