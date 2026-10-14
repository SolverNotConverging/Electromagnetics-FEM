# Periodic eigensolver CUDA benchmarks

`benchmark_cuda.py` compares SciPy ARPACK `eigs`, the existing CPU refined
Arnoldi backend, and CUDA refined Arnoldi with cuDSS. CUDA uses complex FP64
throughout. CPU `eigs` applies `(A - sigma B)^-1 B` explicitly, so the same
benchmark supports the singular mass matrices in the FEM linearization.

Numeric factorization, shifted solves, sparse products, Arnoldi basis storage,
custom CUDA orthogonalization, reconstruction, and residuals run on the GPU.
cuDSS symbolic analysis and the small Hessenberg eigendecomposition/refined
SVDs run on the CPU. GPU timings include upload, analysis, factorization,
iterations, extraction, and download. Assembly and one warmup per backend and
size are excluded. Three repetitions alternate backend order. The default CPU
BLAS thread count is one; existing thread environment settings are respected
and recorded. Do not run other compute jobs while measuring.

Each row records median/min/max seconds, original-pencil residuals, convergence,
operator applications/Arnoldi steps, and eigenvalue agreement with CPU ARPACK
using optimal matching. A speedup is reported only for converged results with
relative eigenvalue mismatch below `1e-5`. JSON records hardware, CUDA/CuPy
versions, thread settings, and command options. CSV/JSON are saved after each
size, and a timing plot is produced at the end. Failed sizes retain an error
record. GPU out-of-memory exceptions can be reported; a host OOM or process
termination cannot be recovered by the script.

For a stencil, `pencil_dofs = side**3`. For the antenna, `field_dofs` is the
reduced FEM unknown count and `pencil_dofs = 2 * field_dofs`. The antenna
benchmark operates on the real unit-cell pencil near `neff=0.5-0.02j` and
measures eigenpairs before the FEM direction/Gauss/PML filters. Increase
`--modes` to benchmark the larger candidate set used by a full FEM solve.

## CUDA setup on Windows

Install the project dependencies as described in the root README. Then install
one optional CuPy distribution and download the matching **cuDSS 0.8** archive
from [NVIDIA's Windows redistributables](https://developer.download.nvidia.com/compute/cudss/redist/libcudss/windows-x86_64/).
The laptop measurements use CuPy 14.2, CUDA 13, and cuDSS 0.8.0.10.

```powershell
uv pip install --python .venv/Scripts/python.exe cupy-cuda13x
$env:CUDSS_LIBRARY=(Resolve-Path build/cuda-deps/libcudss-windows-x86_64-0.8.0.10_cuda13-archive/bin/cudss64_0.dll).Path
.venv/Scripts/python.exe benchmarks/periodic_eigensolver/benchmark_cuda.py --output build/cuda-stencil
.venv/Scripts/python.exe benchmarks/periodic_eigensolver/benchmark_cuda.py --problem antenna --mesh-mm 4 3 2 --ncv 80 --output build/cuda-antenna
```

The archive is a separate runtime dependency; it is not committed or bundled
in the Python wheel. `CUDSS_LIBRARY` can point to an installation anywhere.
The backend handles CUDA DLL search paths, including CUDA 13's `bin/x64`.

## CUDA setup on Linux

Install one CuPy distribution matching the selected CUDA toolkit, and install
cuDSS >= 0.8 from [NVIDIA's Linux redistributables](https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/).
For a CUDA 12 installation:

```sh
python -m pip install cupy-cuda12x
export CUDSS_LIBRARY="/path/to/cudss/lib/libcudss.so"
export LD_LIBRARY_PATH="$(dirname "$CUDSS_LIBRARY"):${LD_LIBRARY_PATH:-}"
python benchmarks/periodic_eigensolver/benchmark_cuda.py --output build/cuda-benchmark
```

For CUDA 13, use `cupy-cuda13x` and the corresponding cuDSS runtime.

## Use in a FEM solve

```python
modes = solver.solve(
    neff_guess=0.5 - 0.02j,
    num_modes=2,
    eigensolver="refined",
    backend="cuda",
    ncv=80,
    max_restarts=24,
    eigensolver_tolerance=1e-8,
    max_refinements=0,
)
```

Run the optional GPU numerical tests with `CUDSS_LIBRARY` set:

```sh
python -m pytest tests/libraries/periodic_eigensolver/test_cuda.py -q
```

CPU-only installations do not import CuPy or load cuDSS unless CUDA is
requested. `backend="auto"` keeps its existing CPU selection.

Saved laptop CSV/JSON and plots are under
`benchmarks/reference_results/cuda_laptop_stencil/` and
`benchmarks/reference_results/cuda_laptop_antenna/`.
