# FEM

Advanced finite-element solvers for electrostatics, guided waves, periodic structures,
and waveguide scattering. FEM uses conforming meshes and supports material interfaces,
full-vector fields, and adaptive refinement. Each solver folder contains its own
`src/`, `docs/`, `examples/`, and `outputs/`.

FEM includes native C++ applications and a compiled periodic eigensolver. Windows
users can install the prebuilt Python 3.12 binary; other platforms build from source.

## Choose a solver

| Solver | Problems it solves | Import | Documentation | Examples |
|---|---|---|---|---|
| electrostatics | Potentials, electric fields, and capacitor energies with dielectric interfaces and charge densities | `fem_electrostatics` | [Guide](fem_electrostatics/docs/guide.rst) | [Examples](fem_electrostatics/examples/README.rst) |
| periodic modes | Bloch modes and complex effective indices in 2D and 3D periodic cells, including leaky structures | `fem_periodic_modes` | [Guide](fem_periodic_modes/docs/guide.rst) | [Examples](fem_periodic_modes/examples/README.rst) |
| waveguide modes | Propagation modes and complex effective indices of 1D and 2D waveguide cross-sections | `fem_waveguide_modes` | [Guide](fem_waveguide_modes/docs/guide.rst) | [Examples](fem_waveguide_modes/examples/README.rst) |
| waveguide scattering | Reflection, transmission, fields, and radiation from 2D waveguide discontinuities | `fem_waveguide_scattering` | [Guide](fem_waveguide_scattering/docs/guide.rst) | [Examples](fem_waveguide_scattering/examples/README.rst) |

The [native transmission-line calculator](apps/transmission_line_calculator/README.rst)
also computes cross-section fields, impedance, effective index, and RLGC parameters.

## Windows: install the binary

Use **64-bit Python 3.12**. Download the wheel from the
[release](https://github.com/SolverNotConverging/Electromagnetics-FEM/releases/latest)
and install it:

```powershell
python -m pip install .\electromagnetics_fem-1.1.0-cp312-cp312-win_amd64.whl
```

Or install directly:

```powershell
python -m pip install https://github.com/SolverNotConverging/Electromagnetics-FEM/releases/download/v1.1.0/electromagnetics_fem-1.1.0-cp312-cp312-win_amd64.whl
```

With uv, use `uv pip install` instead of `python -m pip install` in an activated
environment. With conda, create and activate a Python 3.12 environment, then use
that environment's `python -m pip install` command above.

The wheel includes the native viewers, transmission-line calculator, Cython kernel,
and their runtime libraries. Windows users do not need MSVC, Qt, or vcpkg to run it.
The 3D viewer needs an OpenGL-capable graphics driver.

## Download and run examples

Download this repository using GitHub's **Code → Download ZIP**, and extract it.
After installing FEM, open a terminal in the extracted folder and run:

```sh
python fem_waveguide_modes/examples/1d_parallel_plate_waveguide.py
python fem_periodic_modes/examples/2d_uniform_cell.py
```

Examples import the source solver directly. The installed distribution supplies the
native applications and compiled kernel. Installation also lets you import FEM
from anywhere in the same Python environment:

```python
from fem_waveguide_modes import ModeSolver2D
```

Each example writes to `<solver>/outputs/<example>/`. For example,
`fem_periodic_modes/outputs/2d_uniform_cell/results.h5`. `result.show()` opens its
result directly. Launching a native viewer without a file from the repository or a
solver folder automatically finds that solver's `outputs/` and lists its saved
results, including results in nested example folders:

```sh
python -m fem periodic-viewer
python -m fem scattering-viewer
python -m fem calculator
```

You can also supply an HDF5 file or results directory explicitly.

## Other platforms: install from source

A source installation builds the native applications and Cython kernel. Install a
C++20 compiler, CMake 3.24 or newer, Ninja, Qt 6.2 or newer, HDF5, Eigen 3.4 or newer,
Gmsh with OpenCASCADE, and FTXUI. VTK with Qt support enables the 3D periodic viewer.
See [native build instructions](apps/native_build.md) for platform setup and dependency
source information, and the application guides for individual builds.

In the downloaded repository, with those native dependencies available to CMake:

```sh
python -m pip install .
# Or, in an activated environment:
uv pip install .
```

Python dependencies are installed automatically. `pip install -r requirements.txt`
or `uv sync` alone installs Python dependencies; it does not build the native
applications. Windows developers building from source also need MSVC and the
vcpkg dependencies described in the native build instructions.

## License

FEM source is [MIT licensed](LICENSE). Bundled native dependencies retain their own
licenses, included in the wheel. The Gmsh-linked calculator is distributed under
GPL-3.0-or-later; Qt is dynamically linked under LGPL-3.0. Release assets include
native dependency sources and build recipes.
