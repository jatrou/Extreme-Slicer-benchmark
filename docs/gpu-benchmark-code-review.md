# GPU source review for the independent benchmark

Reviewed source: `mk0000001/Extreme-Slicer`, commit
`0bf0c192484aa0c83c1824f72b32a4c8edd99b0b`.
All source paths and line numbers below refer to that pinned revision, not a
moving branch. Review was read-only apart from this document. This is a scoped
review of the added compute paths, not a security certification of the app.
It makes no measured performance claim.

## What is actually offloaded

CUDA computes the rational Y intersection of an existing polygon edge with a
vertical infill scanline. The caller has already enumerated eligible edge/scanline
pairs on the CPU. The CUDA kernel performs integer comparisons, subtraction,
multiplication and addition; it does not slice the triangle mesh or generate a
complete toolpath (`src/libslic3r/Gpu/CudaSlicer.cu:24-45`,
`src/libslic3r/Fill/FillRectilinear.cpp:775-846`).

Vulkan provides the same intersection operation and a separate tree-support
branch/contour-edge bounding-box filter. Contour-vertex rules, sorting, winding,
polygon clipping and final toolpath topology remain CPU work. The shader filename
`perimeter_infill_candidates.comp` should not be read as evidence that the full
perimeter generator runs on the GPU.

## Findings

### P1: tree-support filter tests a different predicate

**Proven predicate mismatch; full-print reachability not reproduced.**
`src/libslic3r/Gpu/shaders/tree_support_contour_candidates.comp:29-41` tests a
branch's axis-aligned bounding box against the boxes of polygon boundary edges.
`src/libslic3r/Support/TreeSupport.cpp:2874-2882` caches a negative answer as a
replacement for `intersection_ln(line, layer_contours)`.
That CPU operation clips a line against filled polygons, not only their boundary
edges (`src/libslic3r/ClipperUtils.hpp:540-544`; caller at
`src/libslic3r/Support/TreeSupport.cpp:2752-2769`).

A segment wholly inside a polygon can overlap none of its boundary-edge boxes.
This minimal Python example expresses the shader's predicate; it needs no GPU:

```python
def bbox(segment):
    (ax, ay), (bx, by) = segment
    return min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)

def overlaps(a, b):
    ax0, ay0, ax1, ay1 = bbox(a)
    bx0, by0, bx1, by1 = bbox(b)
    return ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1

square = [(0, 0), (10, 0), (10, 10), (0, 10)]
edges = list(zip(square, square[1:] + square[:1]))
branch = ((4, 4), (6, 6))
assert not any(overlaps(branch, edge) for edge in edges)
# The entire nonzero-length branch lies in the filled square, so its
# polygon-clipping intersection is the branch itself, not an empty result.
```

The dispatch size gate is immaterial to this geometric counterexample; enough
such branch/edge requests can exceed the minimum pair count. This document does
not claim that an actual model has demonstrated a changed print. The support
caller groups interior nodes and contains logic retaining some shallow interior
nodes (`TreeSupport.cpp:2773-2815`, `3013-3036`), so an outside-endpoint invariant
needs proof rather than assumption. Do not cache these negatives without a
correct containment check or a proven caller invariant. Strict scanline
validation does not validate this tree filter.

### P2: calibrated dispatch policy is bypassed

**Confirmed control-flow defect.**
`src/libslic3r/Gpu/VulkanSlicer.cpp:1540-1552` first rejects counts below
`batch_size()`, then returns `request_count >= batch_size() || calibrated_check`.
The first operand is necessarily true for every remaining Vulkan workload.
The calibrated crossover and slow-device rejection therefore cannot veto it.
The default configured minimum is 64 (`:42`). CUDA also uses the configured
minimum without a measured crossover. Require both the configured minimum and
the applicable calibrated decision; assess the CUDA policy independently.

### P2: sampled validation does not roll back an affected region

**Confirmed recovery limitation, conditional on a wrong result.**
`src/libslic3r/Fill/FillRectilinear.cpp:946-984` switches to CPU arithmetic after
a mismatch, but retains earlier unchecked GPU results already inserted into the
region. The diagnostic explicitly describes fallback for the remaining results
(`:996`). Sampling checks only the first, last and every 512th result
(`src/libslic3r/Gpu/VulkanSlicer.cpp:1685-1692`).
This does not demonstrate ordinary kernel errors, but the claimed fallback is
not whole-region recovery. Discard and recompute the affected region on a
detected mismatch, and disable or requalify the suspect backend. A benchmark
must reject any run reporting a validation failure.

### P2: CUDA qualification and test coverage do not match the labels

**Confirmed coverage gaps.**
CUDA preparation only queries the device and selects device zero
(`src/libslic3r/Gpu/CudaSlicer.cu:80-91`). The 2,048-vector qualification belongs
to the Vulkan context alone (`src/libslic3r/Gpu/VulkanSlicer.cpp:1102-1141`), while
CUDA proceeds under the shared sampled-validation policy. Add a CUDA-specific
qualification before allowing sampled acceptance.

`src/libslic3r/Gpu/VulkanComputeSelfTest.cpp:31-45` does not select Vulkan despite
its name and output: with CUDA compiled and available, the default preference
can route its intersection test to CUDA. Its tree test is Vulkan. Select and
assert each backend explicitly. The standalone selftest is an `ORCA_TOOLS`
target, not a registered CTest (`src/libslic3r/CMakeLists.txt:606-609`).
`tests/libslic3r/test_gpu_exact_geometry.cpp` checks host helpers and capability
flags, not GPU execution.

## Numerical and measurement boundaries

The host does **not** currently tile live geometry. `GpuExactGeometry.hpp`
defines wide orientation arithmetic and tile-range helpers, but production
dispatch directly copies global coordinates (`VulkanSlicer.cpp:1235-1239`,
`CudaSlicer.cu:108-113`). The CUDA comment that the host tiles requests (`:40`)
is inaccurate. No live range check or wide-reference acceptance gate protects
the multiply/add. This is a missing numerical precondition, not a demonstrated
overflow on an ordinary printer model; test large translated coordinates and
the actual coordinate limits before claiming general exactness.

CUDA allocates two device buffers, uploads, launches, synchronizes, downloads,
frees and materializes output on every batch (`CudaSlicer.cu:108-151`). CPU
candidate enumeration and result consumption remain. Vulkan serializes its
dispatches under a mutex and synchronously waits. These costs must be included
in measurements; their presence alone does not establish a slowdown.

CUDA's `gpu_elapsed_ms` is host-clock launch-plus-device-synchronization time,
excluding allocation and copies (`CudaSlicer.cu:132-137`). Vulkan uses device
timestamps (`VulkanSlicer.cpp:1275-1279`). They are different metrics. Report
end-to-end elapsed time and precisely named stage timings. Synthetic arithmetic
benchmarks are not a substitute for full slicing comparisons.

## Reuse recommendation

Retain the idea of narrow, explicit compute boundaries, stable result IDs and an
independent CPU reference. Do not transplant this implementation wholesale into
Bambu Studio. First profile real projects, fix predicate equivalence and full
fallback recovery, enforce numeric bounds, then evaluate persistent buffers,
larger batched work and an effective crossover policy. Bambu integration also
requires checking its own coordinate scale and caller invariants.

The independent CUDA-only benchmark disables Vulkan, which avoids the identified
tree-filter path. It does not prove correctness of the whole application, Vulkan
mode, all inputs, or a future Bambu port. No network or subprocess execution was
found in the reviewed GPU source files; this statement does not cover unrelated
application code or dependencies.
