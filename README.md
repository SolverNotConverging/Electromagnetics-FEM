# FEM

Electromagnetic solvers using the finite element method.

Shared materials, shapes, persistence, and periodic eigensolver libraries are included.

| Solver | Python package | Documentation |
|---|---|---|
| electrostatics | `fem_electrostatics` | [Guide](doc/solvers/fem/electrostatics/guide.rst) |
| periodic modes | `fem_periodic_modes` | [Guide](doc/solvers/fem/periodic_modes/guide.rst) |
| waveguide modes | `fem_waveguide_modes` | [Guide](doc/solvers/fem/waveguide_modes/guide.rst) |
| waveguide scattering | `fem_waveguide_scattering` | [Guide](doc/solvers/fem/waveguide_scattering/guide.rst) |

## Install from source

Python 3.11–3.13 is supported; `.python-version` selects Python 3.12.
Install uv, clone this repository, and build from the repository root:

```sh
git clone https://github.com/SolverNotConverging/Electromagnetics-FEM.git FEM
cd FEM
uv sync
uv run python -m fem info
uv run python examples/fem/waveguide_modes/rectangular_waveguide_2d.py
```


Native applications include the FEM Periodic Mode Viewer, FEM Waveguide Scattering
Viewer, and Transmission Line Calculator. Build instructions are in each app's
README under `apps/`. Windows builds use MSVC and vcpkg:

```powershell
. ./scripts/setup_msvc_windows.ps1
uv sync
uv run python -m fem info
uv run python -m fem calculator
uv run python -m fem periodic-viewer
uv run python -m fem scattering-viewer
```

Install vcpkg dependencies first:

```powershell
& C:/opt/vcpkg/vcpkg.exe install "qtbase[core,concurrent,widgets,opengl,png]" "hdf5[core,hl,zlib]" eigen3 "gmsh[occ]" ftxui "vtk[core,qt,opengl]" --triplet x64-windows --overlay-ports=./vcpkg-ports
```

For a Python-only build, disable the three native applications:

```sh
uv sync --config-setting=cmake.define.CEM_BUILD_TRANSMISSION_LINE_CALCULATOR=OFF --config-setting=cmake.define.CEM_BUILD_FEM_PERIODIC_MODE_VIEWER=OFF --config-setting=cmake.define.CEM_BUILD_FEM_WAVEGUIDE_SCATTERING_VIEWER=OFF
```

The source is MIT licensed; bundled dependencies retain their own licenses.
See [native dependency provenance](doc/development/native_dependency_sources.md).

## Examples and checks

See [examples](examples/README.rst), [documentation](doc/README.rst), and
[benchmarks](benchmarks/README.md). Electromagnetic solvers use `exp(+i*omega*t)`;
passive permittivity has nonpositive imaginary part.

```sh
uv run python -m pytest
uv run python scripts/check_documentation.py
```

Original source is available under the [MIT license](LICENSE).
