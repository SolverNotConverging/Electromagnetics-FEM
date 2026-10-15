# CUDA architecture compatibility

FEM 2.0 distributes Python GPU operators, not a GPU-specific compiled kernel.
CuPy compiles its elementwise kernels for the active device at runtime and calls
NVIDIA sparse libraries. cuDSS supplies sparse factorization and solves. The
same FEM wheel can therefore run on different supported NVIDIA architectures;
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

Install exactly one of the `cuda12` and `cuda13` extras. cuDSS >= 0.8 is installed
separately; see [installation](installation.md). Set `CUDSS_LIBRARY` when several
cuDSS/CUDA installations coexist so that the selected library matches CuPy.
GPU kernels and CuPy caches are not included in release wheels.

## Verify a colleague's GPU

Run the qualification script using the environment that contains the installed
FEM wheel. It tests every visible device, compares complex direct solves with
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
