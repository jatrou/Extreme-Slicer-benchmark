# CUDA vertical scanline microbenchmark

This standalone harness compiles the production `Gpu/CudaSlicer.cu` and
`Gpu/CudaSlicer.cpp` unchanged. It calls `CudaSlicerBackend` directly; it does
not measure the full slicing pipeline, geometry tiling, Vulkan dispatch,
toolpath construction, or G-code generation.

## Build on Windows

From the repository root in PowerShell, with CUDA 13.1 and VS 2022 Build
Tools installed:

```powershell
$repo = (Get-Location).Path
$out = Join-Path (Split-Path $repo -Parent) 'cuda-microbench'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$boost = Join-Path (Split-Path $repo -Parent) 'extreme-deps\dep_Boost-prefix\src\dep_Boost'
$nvcc = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe'
$vsdev = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
$includes = @(Get-ChildItem (Join-Path $boost 'libs') -Directory -Recurse -Filter include |
  ForEach-Object { $_.FullName } |
  ForEach-Object { '-I"' + ($_.Replace('\', '/')) + '"' })
$argsFile = Join-Path $out 'nvcc.rsp'
@('-std=c++17', '-O3', '-arch=sm_120', '--cudart=shared',
  '-Xcompiler=/MD', '-DSLIC3R_ENABLE_CUDA_SLICER=1') + $includes +
  @('"' + (Join-Path $repo 'tools\benchmark_cuda_scanlines.cpp') + '"',
    '"' + (Join-Path $repo 'src\libslic3r\Gpu\CudaSlicer.cpp') + '"',
    '"' + (Join-Path $repo 'src\libslic3r\Gpu\CudaSlicer.cu') + '"',
    '-o', '"' + (Join-Path $out 'benchmark_cuda_scanlines.exe') + '"') |
  Set-Content -Encoding ascii $argsFile
$cmdFile = Join-Path $out 'build.cmd'
$cmdText = ('call "' + $vsdev + '" -arch=amd64' + "`r`n" +
  '"' + $nvcc + '" --options-file "' + $argsFile + '"' + "`r`n")
[IO.File]::WriteAllText($cmdFile, $cmdText, [Text.Encoding]::ASCII)
cmd /d /c $cmdFile
if ($LASTEXITCODE -ne 0) { throw 'CUDA microbenchmark build failed' }
```

The response file and executable live in the sibling `work/cuda-microbench`
directory, outside the Git repository. If Boost's consolidated `boost/`
headers have been generated, their parent can replace the modular include
list.

## Run after the main build is idle

```powershell
$out = Join-Path (Split-Path (Get-Location).Path -Parent) 'cuda-microbench'
& (Join-Path $out 'benchmark_cuda_scanlines.exe') --repetitions 11 --format csv |
  Tee-Object (Join-Path $out 'results.csv')
```

Use `--format json` to emit a single JSON object. The program exits nonzero
on a CUDA error or any CPU/GPU result mismatch. Every result is checked on
every dispatch, outside the timed interval. Timed data are deterministic,
cover a 300 mm plate at 1,000,000 fixed-point units per mm, include both
segment directions,
and consist only of strict-interior intersections like the production caller.
A separate untimed dispatch checks vertical, endpoint, and outside-segment
misses at base coordinates and after translating both axes by -300 mm and
+300 mm. The largest possible absolute numerator term is bounded by
`120,000,000 * 300,000,000 + 600,000,000 * 120,000,000 = 108,000,000,000,000,000`,
well below signed 64-bit maximum. The CPU output
checksum is consumed after each timed sample to keep the CPU work observable.

The first call at `n=64` is reported as `cold` and includes CUDA runtime and
device initialization. All other rows are warmed medians across the requested
number of repetitions at `n=64, 512, 4096, 16384, 65536, 262144`.

`cpu_ms` is a median per pass of exact 64-bit CPU rational arithmetic with
results written to a preallocated vector. The CPU pass count targets roughly
four million requests per sample to make small-batch timing useful.
`cuda_full_ms` is wall time around the production wrapper. It includes request
packing, allocations, host-to-device copy, launch and synchronization,
device-to-host copy, frees, and output materialization.
`cuda_launch_sync_ms` is the timing recorded by the production CUDA runtime
around kernel launch and `cudaDeviceSynchronize()`. It is **not** a pure
GPU-event kernel duration. `cuda_over_cpu` divides full CUDA latency by CPU
latency; greater than one means the standalone CPU arithmetic was faster.

The current CUDA dispatcher copies the caller's global coordinates directly
into device requests. These 300 mm coordinates exercise that same coordinate
scale, though the deterministic vectors are synthetic rather than captured
production requests. Earlier `results.json` and `results-repeat.json` were
measured with the former sub-millimetre dataset and should be identified as
such if retained for comparison.

Avoid running performance trials while compiling, slicing, or otherwise
loading the CPU or GPU. Record the exact executable/source revision, GPU
driver, CUDA toolkit, power settings, and whether this machine was idle when
publishing results.
