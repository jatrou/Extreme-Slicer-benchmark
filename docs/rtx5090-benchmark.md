# RTX 5090 CLI benchmark

The optional CLI benchmark uses the same locally built executable and project
for CPU and CUDA. It only selects a compute backend immediately before each
plate's `Print::process()` call. It does not change presets, project files,
toolpaths, or the normal CLI behavior when `ORCA_BENCH_BACKEND` is unset.

## Build

Build the pinned source checkout yourself. On Windows, the repository's
`build_release_vs.bat` builds dependencies from their source packages before
building the slicer. From a Visual Studio developer prompt in the checkout:

```bat
build_release_vs.bat deps x64
set ORCA_CMAKE_EXTRA_ARGS=-DSLIC3R_ENABLE_CUDA_SLICER=ON -DCMAKE_CUDA_ARCHITECTURES=120
build_release_vs.bat slicer x64 tests
ctest --test-dir build --build-config Release --output-on-failure
```

The RTX 5090 needs a CUDA toolkit and driver that support its architecture.
Check the configured `CMAKE_CUDA_ARCHITECTURES` and the compiler output; the
`SLIC3R_ENABLE_CUDA_SLICER` CMake option defaults to OFF. Vulkan is not needed
for this comparison and remains disabled, preventing a Vulkan fallback from
masquerading as a CUDA result. The top-level CMake configure may download a
pinned, SHA-256-checked `uv` tool unless `ORCA_BUNDLED_UV_EXECUTABLE` points to
a locally trusted copy. Dependencies are downloaded and compiled by the
`deps` CMake project, so review its source URLs before building.

The independent Windows build used VS 2022 Build Tools (MSVC 19.44), CMake
3.31.6, CUDA 13.1, and native Strawberry Perl for OpenSSL. Dependencies and
build products were kept outside the checkout. For an explicit out-of-source
configuration, use the dependency install prefix and selected VS instance:

```powershell
cmake -S . -B '<build-dir>' -G 'Visual Studio 17 2022' -A x64 `
  -T 'cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1' `
  '-DCMAKE_GENERATOR_INSTANCE=C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools' `
  '-DCMAKE_PREFIX_PATH=<dependency-install>/usr/local' `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DORCA_TOOLS=OFF `
  -DSLIC3R_ENABLE_CUDA_SLICER=ON -DSLIC3R_ENABLE_VULKAN_SLICER=OFF `
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build '<build-dir>' --config Release --target OrcaSlicer_app_gui COPY_DLLS libslic3r_tests -- /m:3 /p:CL_MPCount=12 /nr:false
ctest --test-dir '<build-dir>/tests/libslic3r' -C Release -L Gpu --output-on-failure
```

The CUDA build fixes retain CMake's default host flags and prevent an imported
IlmBase `/EHsc` option from reaching NVCC unwrapped. They do not modify a GPU
kernel. A reused cache from the original checkout needs `-U 'CMAKE_CUDA_FLAGS*'`
once when configuring to discard the original forced-empty values. Limiting
compiler workers avoids excessive committed memory while building the GUI PCH.
The focused CTest cases exercise host helpers; the standalone microbenchmark
and strict full-slice pass provide actual device validation.

For this local build, copy the toolkit's `bin/x64/cudart64_13.dll` beside the
application DLL as well. With the tested CMake version, upstream's install rule
resolves `CUDA::cudart` to its import library, so it does not stage that runtime
DLL correctly. This benchmark does not validate an installer package.

## Run

`tools/benchmark_slicer.py` automates isolated CLI runs, writes logs and a JSON
manifest outside the checkout, records executable and engine DLL hashes, and
compares G-code while preserving all machine commands. It normalizes the known
generation timestamp and the numeric ID in `printing object ... id:... copy ...`
comments; repeated CPU-only runs also vary that ID. Object names, copy indices,
unique object-label comments, and printer commands remain part of the comparison.
Raw hashes are retained as well.
With no project argument it creates a deterministic corrugated cylinder and
resolved X1C settings at 0.2 mm layers and 100% rectilinear infill:

```powershell
python -B tools/benchmark_slicer.py --exe '<console-shim.exe>' --strict
python -B tools/benchmark_slicer.py --exe '<console-shim.exe>' --runs 5 --warmups 1
```

Use `--project '<project.3mf>'` for a real project, `--backends cuda` for a
separate CUDA pass, or `--backends stock --stock-exe '<stock-console-shim.exe>'`
for an unmodified Bambu baseline. `--allow-newer-file` permits the import attempt
without overriding embedded settings. A single-backend run cannot establish
CPU/CUDA output equality by itself. GPU validation failures, absent accepted
GPU work, and failed G-code output are reported as failures; a speedup ratio
is emitted only for matching CPU/CUDA output. `--sample-gpu` optionally records
whole-device utilization, which can include other applications and does not
replace the backend's accepted-work counters.
`--set KEY=VALUE` records explicit compatibility overrides and applies them to
every selected backend; none are applied automatically. Such a run must be
described with its overrides, rather than as the untouched project settings.

For manual runs:

Use a single plate and a fresh output directory for each run. In PowerShell,
replace the executable and project placeholders with absolute paths to your
self-built executable and the same `.3mf` file:

```powershell
$env:ORCA_BENCH_BACKEND = 'cpu'
$env:ORCA_BENCH_STRICT = '0'
$env:ORCA_BENCH_BATCH_SIZE = '64'
& '<self-built-OrcaSlicer.exe>' --slice=1 --outputdir='<cpu-output-dir>' '<project.3mf>'

$env:ORCA_BENCH_BACKEND = 'cuda'
& '<self-built-OrcaSlicer.exe>' --slice=1 --outputdir='<cuda-output-dir>' '<project.3mf>'
```

The CLI prints one line per processed plate beginning `ORCA_BENCH_RESULT `,
followed by a JSON object. `process_wall_ms` measures only `Print::process()`,
not project loading or G-code export. Compare it across the two runs, and
check `active_backend`, `device`, `dispatch_calls`,
`accepted_gpu_intersections`, `outcome`, `gpu_performance_valid`, and
`validation_failures`. The CPU run reports `active_backend=CPU` and skips
CUDA initialization. `gpu_performance_valid=true` requires CUDA to remain
active, at least one accepted GPU intersection, and zero validation failures.
`outcome=no_gpu_work` means the model did not exercise the eligible GPU path;
`outcome=validation_failed` means GPU results failed the CPU check. Neither is
a valid GPU performance measurement. For CUDA, `total_gpu_ms` is host-clock
kernel-launch-plus-synchronization time, not an isolated GPU event measurement;
`total_host_ms` includes host dispatch overhead. Neither is whole-slice time. The existing
`result.json` still reports broader CLI elapsed times, including work outside
the benchmarked process section.

For a correctness pass, set `ORCA_BENCH_STRICT=1` for CUDA. This recomputes
every GPU intersection on the CPU and adds validation cost, so use
`ORCA_BENCH_STRICT=0` for the throughput comparison. The backend variables
accept exactly `cpu` or `cuda`, `0` or `1` for strict validation, and a decimal
batch size from 16 to 4096, matching the backend clamp. Invalid values fail
before slicing. An unavailable CUDA device fails with a clear error. Check
the reported `validation_mode`; the existing
`ORCA_VULKAN_SLICER_VALIDATION=strict` environment override can still force
strict checks. `--load-slicedata` is rejected in benchmark mode to prevent a
cached slice from being compared with a full slice.

The CLI's existing G-code export path is retained. It passes no thumbnail
generator to `export_gcode`, and this benchmark does not add or bypass any
printer upload step. For comparable results, keep the same project and
settings, avoid other heavy GPU or CPU workloads, and repeat runs after the
first warm-up. Compare emitted G-code/toolpaths as well as elapsed time before
using the output to print; a matching benchmark time alone says nothing about
print correctness.

### Capturing Windows CLI output

The release GUI launcher may detach its console. Compile
`tools/benchmark_slicer_console.cpp` with MSVC (`cl /O2 /MD /EHsc /std:c++17
benchmark_slicer_console.cpp /link /SUBSYSTEM:CONSOLE`) and place the executable
beside the slicer DLL and `resources` directory. It forwards command-line
arguments to `orcaslicer_main` in `OrcaSlicer.dll` or `bambustu_main` in
`BambuStudio.dll`, preserving the engine. It does not initialize a viewport.
Use a separate copied application directory for a stock Bambu baseline and
pass a separate `--datadir`; do not replace the installed application's launcher.
Keep the compiler's object and executable output outside the source checkout.

The supplied `fox.3mf` is an H3C project made with Bambu Studio 02.09.00.79.
This fork's bundled profiles do not include an H3C preset. It also maps the
project's `zig-zag` infill value to `rectilinear` on import, so a successful
slice here would not be a like-for-like H3C toolpath comparison. First prove
the benchmark path with a supported test project. Treat the fox project as a
separate compatibility investigation, and do not print its output without a
full preset and toolpath review.
