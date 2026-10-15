# CPU and GPU acceleration

Every solver defaults to `backend="auto"`. It checks that CuPy, an NVIDIA GPU and
cuDSS can complete a small sparse solve, then selects GPU if available or SciPy CPU
otherwise. `backend="cpu"` forces CPU execution; `backend="gpu"` requires GPU support
and reports runtime failures. Numerical failures propagate without silently retrying.

Periodic eigensolvers accept `eigensolver="auto"`, `"dense"` or `"sparse"`. Dense
eigensolving requires `backend="cpu"`; GPU uses sparse shift-invert. Mesh assembly
stays on the CPU. The shared implementation is in `fem.common.linalg`,
`fem.common.gpu` and `fem.common._cudss`.

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

FEM distributes Python GPU operators, not a GPU-specific compiled kernel.
CuPy compiles its elementwise kernels for the active device at runtime and calls
NVIDIA sparse libraries. cuDSS supplies sparse factorization and solves. The
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
`backend="auto"` falls back to SciPy CPU if the device/runtime probe fails;
`backend="gpu"` reports the failure. macOS uses CPU.

Install exactly one of the `cuda12` and `cuda13` extras; each includes CuPy and the
matching cuDSS runtime wheel (see above). Set `CUDSS_LIBRARY`
when several cuDSS/CUDA installations coexist so that the selected library matches CuPy.
GPU kernels and CuPy caches are not part of the package.

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
