# Changelog

## Unreleased

- Add explicit CPU, hybrid GPU Arnoldi/CPU Ritz, and resident GPU eigenvalue backends.
- Auto now tries hybrid and warns before CPU fallback; resident GPU requires explicit selection.
- Keep cuDSS direct scattering for both accelerated choices and preserve strict explicit-backend errors.


## 2.0.1 — 2026-10-16

Restore custom CUDA Arnoldi and full GPU-resident eigenvalue iteration, including Ritz eigenvalue/SVD extraction, sparse solves, residuals, and supported periodic adaptive calculations. All eigenvalue families use the resident path when the GPU backend is selected.

Automatic CPU fallback emits a warning; an unavailable explicit GPU request raises an error. Scattering direct solves use cuDSS when selected. Optional CUDA dependencies require the supported cuDSS ABI; the default wheel remains py3-none-any.

Fix Boolean-difference conductor-frame boundary classification and retain numerical cleanup regressions and joint FEM/FDFD analytical validation tools. Correct the documented normalized waveguide attenuation migration.

Validation: joint analytical checks pass on CPU/GPU, including modal phase/attenuation, slab fields, scattering, electrostatics, and copper wall loss. Full release qualification and package checks are recorded with the release artifacts.


## 2.0.0

A breaking release that turns six loosely coupled packages into one pure-Python
`fem` package with one vocabulary and a Python viewer. The numerical methods are
unchanged; the regression tests reproduce the 1.x results.

### Layout and installation

- One package: `src/fem/{common,adaptivity,electrostatics,waveguide,periodic,scattering}`.
  Install with `pip install -e .`; the wheel is `py3-none-any`.
- Everything public is importable from `fem` (`fem.ModeSolver2D`, `fem.Material`,
  `fem.materials`, `fem.shapes`, `fem.load`, `fem.show`, ...).
- Examples moved to `examples/<family>/` and no longer modify `sys.path`; outputs
  go to an ignored `outputs/` folder next to each script.
- Docs are a single tree in `docs/` (guides, solver pages, generated API pages).
- CUDA is optional: `pip install -e .[cuda12]` (or `cuda13`) installs CuPy and cuDSS.
- Removed: the C++/Qt/VTK native viewers, the transmission-line calculator and its
  TUI, CMake/vcpkg packaging, `requirements.txt`, the custom refined Arnoldi
  eigensolver with its Cython/CUDA kernels.

### Viewer

- `result.show()`, `fem.show(result_or_path)` and `fem view result.h5` open one
  PySide6 window for every family: case, mode, component and quantity selectors
  with Mesh and Material toggles; E and H components (potential, E and D for
  electrostatics); Matplotlib for 1D/2D fields and PyVista for 3D periodic cells,
  where E and H are drawn as arrows; S-parameter spectra for sweeps; far-field
  patterns for scattering with an NF2FF contour.
- 1D waveguide meshes record material region tags like the 2D meshes.
- `fem inspect result.h5` prints a summary; `fem info` reports the backend.

### Migration

| 1.x | 2.0 |
| --- | --- |
| `from fem_waveguide_modes import ModeSolver2D` (and the other `fem_*` packages) | `import fem; fem.ModeSolver2D` |
| `from fem_common import Material, materials, shapes` | `from fem import Material, materials, shapes` |
| `<family>.load_result(path)` | `fem.load(path)` |
| `WaveguideScatteringSolver2D` | `fem.ScatteringSolver2D` |
| `FrequencySweepResult` | `fem.ScatteringSweepResult` |
| scattering `Mode`, `ModeSet` | `fem.scattering.PortMode`, `fem.scattering.PortModeSet` |
| `frequency_hz`, `frequencies_hz` | `frequency`, `frequencies` |
| `CEMError` | `fem.FEMError` |
| `mode.propagation_constant` | `mode.neff.real` |
| waveguide `mode.attenuation_constant` | `-mode.neff.imag` (normalized); `mode.alpha` is dimensional and equals `-mode.neff.imag * k0` |
| periodic `mode.attenuation_constant` | `mode.alpha` (inverse metres, unchanged units) |
| periodic `plot_dispersion(f, neff)` | `fem.plot_dispersion(f, neff, connect=False)` |
| `visualize(..., mesh_overlay=..., ex=True)` | `visualize(..., mesh=..., component="Ex")` |
| `backend="python"` / `"cython"` / `"cuda"` | `backend="cpu"` / `"cpu"` / `"gpu"` |
| `eigensolver="arnoldi"`, `max_restarts=` | `eigensolver="sparse"`, `max_iterations=` |
| native viewers, `transmission-line-calculator` | `result.show()`, `fem view`; calculator removed |

### Behaviour changes

- Scalar numeric controls reject numeric strings, booleans and arrays; mesh
  connectivity must be exact in-range integers.
- Result archives use the `fem-results` 1.0 envelope; 1.x archives are not read.
- Waveguide solvers now reject zero relative permittivity entries, like the
  periodic solvers (both share one internal `DiagonalMaterial`).
- `fem.plot_dispersion` labels modes from zero, matching mode indices.
