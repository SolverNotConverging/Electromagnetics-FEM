# FEM

Advanced finite-element solvers for electrostatics, guided waves, periodic structures,
and waveguide scattering. FEM uses conforming meshes and supports material interfaces,
full-vector fields, and adaptive refinement. Each solver folder contains its own
Python modules, `docs/`, `examples/`, and `outputs/`.

FEM includes native C++ applications and shared CPU/GPU sparse solvers. Windows and Apple silicon macOS
users can install release binaries. This 2.0.0 checkout is a breaking source update; see the migration guide.

## Choose a solver

| Solver | Problems it solves | Import | Documentation | Examples |
|---|---|---|---|---|
| electrostatics | Potentials, electric fields, and capacitor energies with dielectric interfaces and charge densities | `fem_electrostatics` | [Guide](fem_electrostatics/docs/guide.rst) | [Examples](fem_electrostatics/examples/README.rst) |
| periodic modes | Bloch modes and complex effective indices in 2D and 3D periodic cells, including leaky structures | `fem_periodic_modes` | [Guide](fem_periodic_modes/docs/guide.rst) | [Examples](fem_periodic_modes/examples/README.rst) |
| waveguide modes | Propagation modes and complex effective indices of 1D and 2D waveguide cross-sections | `fem_waveguide_modes` | [Guide](fem_waveguide_modes/docs/guide.rst) | [Examples](fem_waveguide_modes/examples/README.rst) |
| waveguide scattering | Reflection, transmission, fields, and radiation from 2D waveguide discontinuities | `fem_waveguide_scattering` | [Guide](fem_waveguide_scattering/docs/guide.rst) | [Examples](fem_waveguide_scattering/examples/README.rst) |

## Native applications

The Windows and macOS wheels install these applications and their runtime libraries:

| Application | What it does | Launch |
|---|---|---|
| Transmission-line calculator | Cross-section E/H fields, impedance, effective index, and RLGC parameters | `python -m fem calculator` |
| Transmission-line calculator CLI/TUI | Terminal interface to the same calculations | `python -m fem calculator-cli` |
| Periodic mode viewer | Interactive 2D/3D fields, modes, meshes, and sweeps | `python -m fem periodic-viewer` |
| Periodic mode inspector | Print HDF5 archive metadata and mode data | `fem-periodic-mode-inspect result.h5` |
| Waveguide scattering viewer | Fields, lead modes, S-parameters, and radiation | `python -m fem scattering-viewer` |
| Waveguide scattering inspector | Print scattering archive and frequency-case data | `fem-waveguide-scattering-viewer-inspect result.h5` |

See the [calculator](apps/transmission_line_calculator/README.rst),
[periodic viewer](apps/fem_periodic_mode_viewer/README.rst), and
[scattering viewer](apps/fem_waveguide_scattering_viewer/README.rst) guides.
Electrostatics and waveguide-mode viewers use Matplotlib.

## Get started

Read the [installation guide](docs/installation.md), then run a small example:

```sh
python fem_waveguide_modes/examples/1d_parallel_plate_waveguide.py
```

Build a solver, assign materials and geometry, mesh, then solve. Results expose
`plot()`, `show()`, and `save(path)`; saving and opening viewers are explicit.
Lengths are in metres and frequencies in hertz. Every solver defaults to
`backend="auto"`: use GPU when CuPy, the CUDA device and cuDSS are available,
otherwise SciPy CPU. Set `backend="cpu"` or `backend="gpu"` to force execution.

- [Documentation and solver capabilities](docs/README.md)
- [Breaking changes in 2.0](docs/migration-2.0.md)
- [Contributing and adding a solver](CONTRIBUTING.md)
- [Test commands](tests/README.rst)
- [Analytical benchmarks](benchmarks/README.md)
- [Native application builds](apps/native_build.md)

## License

FEM source is [MIT licensed](LICENSE). Bundled dependencies retain their licenses.
The Gmsh-linked calculator is distributed under GPL-3.0-or-later; Qt is
dynamically linked under LGPL-3.0. Release assets include native dependency
sources and build recipes, and wheels include dependency licenses.
See the [native build guide](apps/native_build.md) for dependency sources and packaging.
