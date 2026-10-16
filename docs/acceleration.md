# CPU and GPU acceleration

Every solver defaults to `backend="auto"`. It checks that CuPy, an NVIDIA GPU and
cuDSS can complete a small sparse solve, then selects GPU if available or SciPy CPU
otherwise. Automatic CPU fallback emits a `RuntimeWarning`. `backend="cpu"` forces
CPU execution without probing CUDA or warning; `backend="gpu"` raises
`BackendCapabilityError` if its required CUDA facilities are unavailable.
Numerical failures after selection propagate without silently retrying.

Periodic eigensolvers accept `eigensolver="auto"`, `"dense"` or `"sparse"`. Dense
eigensolving requires `backend="cpu"`; GPU uses sparse shift-invert. Mesh assembly
stays on the CPU. The shared implementation is in `fem.common.linalg`,
`fem.common.gpu` and `fem.common._cudss`.

By default, GPU eigenproblems use device-resident refined Arnoldi with custom complex FP64
projection and subtraction kernels, restored from commit `3402f85`. The large
Arnoldi basis, orthogonalization, sparse products, original-pencil residuals,
and restart vectors stay on the GPU. GPU eigenproblems require CUDA `geev` and
SVD for dense Ritz extraction. Before selection, a small readiness check compiles
the custom kernels and exercises CUDA eigensolving and SVD, in addition to the
cuDSS sparse readiness solve. This resident core serves generalized and quadratic
eigenproblems in 1D/2D waveguides, 2D/3D periodic modes and scattering port modes.
Only a two-byte control packet is read per restart cycle; converged eigenpairs
are downloaded once for family-specific result handling. If any required facility
is unavailable, `auto` warns and selects CPU, and explicit `gpu` raises an error.
There is no automatic hybrid CPU Ritz fallback. The historical hybrid extraction
is retained only for explicitly selected benchmark controls.
CPU eigenproblems continue to use SciPy ARPACK. Outside the complete 3D periodic
path below, physical filters and field reconstruction still run on CPU.

Quadratic problems retain block elimination: cuDSS factors the reduced
`Q(shift)` matrix, rather than a companion matrix twice its size. Singular
quadratic mass blocks remain supported. Each solve performs fresh analysis and
factorization and reuses that factor throughout Arnoldi and its restarts.

Resident GPU `ncv` defaults to `min(size, max(96, 6*count + 16))`. Here `count` includes
the extra candidates requested by the FEM solver's physical filters. An explicit
`ncv` is respected. GPU `max_iterations` limits operator applications; its default
is thirteen cycles of the selected subspace size. CPU `max_iterations` retains
SciPy's ARPACK iteration semantics. When a budget expires, only converged
original-pencil eigenpairs are returned; an empty converged set raises an error.

## Resident 3D periodic solve

For `PeriodicModeSolver3D`, CUDA Ritz availability also enables the complete
resident numerical solve automatically:

```python
modes = solver.solve(backend="gpu", num_modes=2,
                     ncv=96, eigensolver_tolerance=1e-8, max_refinements=0)
```

This path uses CUDA for the nonsymmetric Ritz eigenproblem (`cuSOLVER geev`),
batched refined-vector SVDs, Arnoldi normalization/orthogonalization, polynomial
and weak-Gauss checks, duplicate/direction/PML filtering, field reconstruction
and normalization, and the interface-jump adaptive estimator. Large numerical
arrays remain on device throughout these operations. No norm scalar is
downloaded inside each Arnoldi step; two status bytes are read per restart cycle.
The final accepted modes/fields are downloaded to preserve the existing NumPy,
plotting and HDF5 result APIs. Metadata `gpu_execution` records iteration work,
explicit application control-download bytes, setup time and final numerical
export size. These counts exclude vendor-library internal workspace/control.

Meshing, FE trace geometry and arbitrary Python material callbacks are prepared
on CPU. Static trace data/material samples and GPU operators are cached for the
current mesh and invalidated when geometry or mesh changes. Both interior and
periodic-seam traces are processed in batches. This caching/batching also removes
CPU work, so a comparison against the original hybrid path is not a measurement
of PCIe transfer cost alone. First-solve setup and warmed repeated solves should
be reported separately. Sparse analysis/factorization is fresh for every solve.
Changing the selected CUDA device also rebuilds its cached buffers. The resident
subspace default is `min(size, max(96, 6*count + 16))`; an explicit `ncv` is
respected. CPU refinement/remeshing remains outside the device numerical solve.

CuDSS still performs host-side symbolic analysis; CUDA `geev` uses library-owned
pinned host workspace. Thus this is not literally a CPU-free or universally
zero-transfer vendor-library execution. CuPy/cuSOLVER must support CUDA `geev`;
configurations without it warn and select CPU for `auto`, and raise an error
for explicit `gpu`. No additional public solver flag is required.

Run `benchmarks/benchmark_resident_periodic_3d.py` for complete-solve timings and
`benchmarks/benchmark_resident_cached_cpu.py` for a control using identical
prepared/batched trace arithmetic on CPU. Both compare validated physical modes
and include the adaptive estimate even with zero requested refinements.

## Direct scattering solves

`ScatteringSolver2D.solve(backend="gpu")` uses cuDSS for fresh sparse numeric
factorization and forward/back substitution of the PEC-reduced complex Maxwell
system. The default `backend="auto"` selects this same path when the GPU runtime
is available. This applies to longitudinal PML and matched-port boundaries,
including nonzero prescribed PEC traces. If lead modes must also be solved,
the selected backend additionally requires the resident eigenvalue facilities
described above; supplying existing modes requires only the direct CUDA/cuDSS
runtime. The result exposes
`result.solve_info["method"] == "cudss-direct"` and the checked linear residual.

Mesh/matrix/source assembly, PEC condensation, residual validation, field
reconstruction, monitor projection and the adaptive jump estimator currently run
on CPU. Matrix/RHS upload and coefficient download occur for each direct solve;
no factorization is cached between calls. Consequently GPU selection does not
promise a speedup for small or inexpensive 2D factorizations.

`benchmarks/benchmark_scattering_cudss.py` measures both the complete public solve
and its direct stage on identical lossy-dielectric scattering systems. The direct
stage includes PEC condensation, analysis/factorization, substitution, transfers
and residual validation. Meshing and incident-mode preparation are excluded.
CPU/GPU runs alternate after warmup, use fresh factors, and check identical
assembled matrices/sources and matching coefficients and S parameters.

## Install CUDA support

CUDA is optional and needs an NVIDIA GPU, a compatible driver and a CUDA 12 or 13
runtime. Install the extra matching your CUDA version; it pulls CuPy and the cuDSS
runtime wheel:

```sh
python -m pip install -e ".[cuda12]"   # or ".[cuda13]"
```

FEM finds the pip-installed cuDSS automatically. To use another installation, set
`CUDSS_LIBRARY` to `libcudss.so` or `cudss64_0.dll`.

## Architecture compatibility

FEM distributes Python GPU operators and CUDA kernel source in the wheel.
CuPy compiles the Arnoldi and elementwise kernels for the active device at
runtime and calls NVIDIA sparse libraries. cuDSS supplies sparse factorization and solves. The
same pure-Python wheel can therefore run on different supported NVIDIA architectures;
install the matching CuPy, CUDA toolkit and cuDSS build on each machine.

| GPU family | Compute capability | Runtime selection |
|---|---|---|
| Pascal | 6.x | CUDA 12 + `cuda12` extra + CUDA 12 cuDSS |
| Volta | 7.0 / 7.2 | CUDA 12 + `cuda12` extra + CUDA 12 cuDSS |
| Turing | 7.5 | CUDA 12 or 13, with matching CuPy and cuDSS |
| Ampere, Ada, Hopper | 8.x / 9.0 | CUDA 12 or 13, with matching CuPy and cuDSS |
| Blackwell | 10.x / 12.x | Prefer CUDA 13 and matching CuPy/cuDSS |
| Maxwell or older | Below 6.0 | SciPy CPU backend |

This table describes runtime requirements, not a hardware test matrix. A toolkit
must support the exact GPU model, and its driver and cuDSS build must be compatible.
FEM rejects pre-Pascal devices and Pascal/Volta with CUDA 13 before GPU allocation.
`backend="auto"` warns and falls back to SciPy CPU if the device/runtime probe
fails; `backend="gpu"` reports an error. Eigenvalue solves additionally require
the resident Ritz/kernel readiness check above. macOS uses CPU.

Install exactly one of the `cuda12` and `cuda13` extras; each includes CuPy and the
matching cuDSS runtime wheel (see above). Set `CUDSS_LIBRARY`
when several cuDSS/CUDA installations coexist so that the selected library matches CuPy.
Compiled GPU binaries and CuPy caches are not part of the package.

## Verify a colleague's GPU

Run the qualification script using the environment that contains the installed
FEM package. It tests every visible device, compares complex direct solves with
SciPy, and checks generalized and quadratic eigenproblems against known roots:

```sh
python /path/to/FEM/scripts/qualify_cuda.py --output cuda-qualification.json
```

Use a new `CUPY_CACHE_DIR` to exercise runtime compilation. The JSON report records
the actual GPU, compute capability, CuPy/CUDA versions and numerical residuals.
To select a GPU in application code, use `with cupy.cuda.Device(index):` around
the solver call. Qualification checks devices individually; a single FEM solve
does not distribute its work across multiple GPUs.

The local 2.0 qualification uses an RTX 4070 Laptop GPU (SM 8.9), CUDA runtime
13.2, CuPy 14.2 and cuDSS 0.8. Other architectures require execution testing on
those devices; architecture-policy unit tests do not replace that testing.

## Upstream support references

- [NVIDIA cuDSS architecture and operating-system support](https://docs.nvidia.com/cuda/cudss/index.html)
- [NVIDIA CUDA 13 architecture transition](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers)
- [CuPy runtime compilation and caching](https://docs.cupy.dev/en/stable/user_guide/performance.html)
