# FEM benchmarks

Run analytical benchmarks after installing the project:

```sh
python benchmarks/analytical/coaxial_waveguide_adaptivity.py --check
python benchmarks/analytical/parallel_plate_electrostatics.py --check
python benchmarks/analytical/rectangular_waveguide_modes.py --check
python benchmarks/analytical/uniform_periodic_medium.py --check
```

Reports are written to ignored `outputs/benchmarks/analytical/`.

## Current CUDA versus SciPy modal solvers (FEM and FDFD)

The current [RTX 4070 Laptop / i9 CUDA versus SciPy benchmark](../docs/benchmarks/cuda_vs_scipy.rst)
contains project-specific results, plots, timing boundaries and downloadable data.

With the sibling FDFD checkout available and FEM's CUDA-enabled Python environment:

```powershell
uv pip install --python .venv/Scripts/python.exe threadpoolctl
python benchmarks/cuda_vs_scipy.py
python benchmarks/report_cuda_vs_scipy.py outputs/benchmarks/cuda_vs_scipy_20261016
```

Use the optional CUDA runtime described in the [acceleration guide](../docs/acceleration.md).
The driver sets
single-thread BLAS/OpenMP execution and verifies the loaded thread pools. It
compares each project's current public CUDA path with SciPy ARPACK/SuperLU on
identical meshes, checks mode agreement and residuals, and records medians of
three alternating paired repeats after per-case warmup. Every solve performs a
fresh factorization. Mesh time, first solve, eigenvalue-stage time and complete
public solve time are recorded separately; warm FEM 3D repeats reuse static
resident geometry. Waveguide Krylov settings retain production defaults;
FDFD periodic 2D and FEM periodic 3D use `ncv=96` on both backends. FDFD periodic
3D is excluded. Results checkpoint to JSON and completed cases are skipped on
resume. `--ids` selects individual configuration IDs from `configurations()`.

## Additional 2D waveguide runner

`gpu_waveguide_2d.py` compares explicit CPU and GPU backends on the same
assembled mesh, with no adaptive remeshing. It requires the `cuda12` or `cuda13`
extra and a working cuDSS installation; it never silently falls back to CPU.

PowerShell example:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python benchmarks/gpu_waveguide_2d.py --levels 12 24 48 80 160 --repeats 3
```

The five cases use a 1 m × 0.5 m PEC enclosure at 299.792458 MHz:

| Case | Interior |
| --- | --- |
| `pec` | Vacuum |
| `pec_lossy` | Uniform dielectric, relative epsilon = 2 − 0.02j |
| `pec_post` | Vacuum and an internal PEC rectangle |
| `pec_sibc` | Vacuum and an internal SIBC rectangle, Zs = 20 + 20j ohm |
| `pec_sibc_lossy` | Lossy dielectric and the SIBC rectangle |

The rectangle occupies x = 0.56–0.68 m, y = 0.11–0.23 m. SIBC applies
to its exposed facets; the exterior wall remains PEC. Uniform sizing disables
material and boundary refinement so the independent variable is mesh size.
Actual reduced physical DOF are reported; the companion eigenproblem is twice
that size. `--order 2` tests second-order elements; `--modes 4` tests a larger
requested mode set.

Reports in `outputs/benchmarks/gpu_waveguide_2d/` include every timed sample,
median/min/max public solve timings, mesh/assembly timing, CPU/GPU effective-index
agreement, residuals, an environment record and a speedup plot. Speedup means
CPU time / GPU time, so values above 1 favor GPU. The total-time estimate adds
the same measured mesh/assembly cost to each backend's median solve time.
Construction, imports, runtime startup and warmups are excluded. Meshing/assembly
is measured once per configuration. Each configuration gets an untimed solve
on each backend before the alternating measured runs; every solve refactorizes.
Sparse shift-invert is forced on CPU to avoid comparing dense QZ with sparse GPU.
CPU uses ARPACK; GPU uses device-resident refined Arnoldi. Mode reconstruction
and validation remain on CPU for both paths.
GPU timings include transfers, cuDSS setup, analysis, factorization and solves.
CPU/GPU mode sets are matched by minimum effective-index distance and must agree
within 1e-6 relative error; each returned mode also passes solver residual and
Gauss-law validation. This checks backend agreement, not mesh convergence.

## Additional 3D periodic runner

`gpu_periodic_3d.py` applies the same warmup, alternating backend order and
three-repeat median protocol to first-order tetrahedral Nedelec elements:

```powershell
python benchmarks/gpu_periodic_3d.py --sizes-mm 4 2.5 1.6 1 0.7 --repeats 3
```

Cases are `uniform_pec`, `uniform_lossy`, `iris_pec`, and `iris_lossy`.
All use a WR-90 PEC enclosure (22.86 × 10.16 mm), an 8 mm period and
12 GHz. The uniform cases serve as TE10 analytical controls. The iris cases
add PEC blocks spanning x = 0–4 mm and 18.86–22.86 mm, the full height,
and z = 3.6–4.4 mm. They therefore exercise longitudinally varying periodic
geometry. Lossy cases fill the air region with relative epsilon = 2 − 0.02j.
Material-aware sizing is disabled; requested size is a Gmsh target, not a
guarantee on the longest mesh edge. Reports record the actual reduced DOF
after PEC and signed periodic-edge constraints; companion dimension is twice
that size. One forward mode is requested by default; `--modes` changes it.

Both backends explicitly use sparse shift-invert with the same seed, target,
candidate count and tolerances (eigensolver 1e-8, QEP residual 1e-8, weak
Gauss energy residual 1e-6). The 3D periodic implementation explicitly rejects
SIBC/GoodConductor assignments, so those cases are unsupported and are not
approximated as PEC. Outputs go to `outputs/benchmarks/gpu_periodic_3d/`.

## Resident GPU experiment

```powershell
python benchmarks/benchmark_resident_periodic_3d.py --sizes-mm 0.9 0.6 --modes 2 --ncv 96 --repeats 3
python benchmarks/benchmark_resident_cached_cpu.py --sizes-mm 0.9 0.6 --repeats 3
```

Use the single-thread BLAS/OpenMP environment above and run these sequentially.
The benchmark compares public ``hybrid`` and ``gpu`` selections with the same
GPU postprocessing. The first compares complete hybrid and resident GPU solves, including
physical filters, field export and the jump estimator. It records first-solve
and warmed timings separately and requires matching effective indices,
normalized coefficients and adaptive residuals. The second controls for the
resident path's prepared/batched trace data by performing the same estimator
arithmetic on CPU. Its timings are warm only; host trace arrays are copied once
from prepared device geometry before measurement. The resident CUDA path also
uses CUDA Ritz eigensolving/SVD and device normalization; see
[execution scope](../docs/acceleration.md#resident-3d-periodic-solve).


## Direct scattering with cuDSS

```powershell
python benchmarks/benchmark_scattering_cudss.py --sizes-um 0.12 0.06 0.03 --repeats 3
```

Run with the single-thread BLAS/OpenMP environment above. Cases use a 1 × 6 μm
PEC guide at 1.55 μm wavelength with a 0.6 μm lossy slab (epsilon = 2 − 0.02j),
and either z PML or matched ports. The same CPU-prepared incident mode is used
for both backends. `backend="auto"` is checked to select `cudss-direct` in the GPU
warmup. Measurements alternate CPU/GPU after per-case warmup, with fresh analysis
and factorization every time. The direct-stage timer includes PEC condensation,
transfers and the linear residual check. Public timing also includes assembly,
sources, fields, modal projections and the adaptive estimator. Actual DOF,
three-repeat medians, coefficient agreement, S-parameter checks and residuals are
saved under `outputs/benchmarks/scattering_cudss/`. Speedup is CPU / GPU; values
below one indicate GPU overhead exceeds its compute benefit for that case.
