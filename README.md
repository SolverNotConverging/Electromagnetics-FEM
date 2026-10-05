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

## Native applications

The Windows wheel installs these applications and their runtime libraries:

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

Install the wheel into the **same Python environment** used to run examples.
For this repository's `.venv`, use `.venv/Scripts/python.exe -m pip install <wheel>`
or `uv pip install --python .venv/Scripts/python.exe <wheel>`. Running `uv sync`
alone installs Python dependencies and does not install the native applications.
Source builds under `build/native-release` are also found automatically.

## Windows: install the binary

Use **64-bit Python 3.12**. Download the wheel from the
[release](https://github.com/SolverNotConverging/Electromagnetics-FEM/releases/latest)
and install it:

```powershell
python -m pip install .\electromagnetics_fem-1.1.1-cp312-cp312-win_amd64.whl
```

Or install directly:

```powershell
python -m pip install https://github.com/SolverNotConverging/Electromagnetics-FEM/releases/download/v1.1.1/electromagnetics_fem-1.1.1-cp312-cp312-win_amd64.whl
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
from fem_waveguide_modes import ModeSolver2D, Material, materials, shapes
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

## Install from source

A source installation builds all native applications and the Cython kernel.
Download and extract the repository, then run the commands below from its root
folder. Use Python 3.12 for these instructions. Python dependencies are installed
automatically by `pip install .`; no `requirements-dev.txt` is needed.

Native dependencies are a C++20 compiler, CMake 3.24+, Ninja, Qt 6.2+,
HDF5, Eigen 3.4+, Gmsh with OpenCASCADE, and FTXUI. The commands also install
VTK with **Qt 6** support and enable the 3D periodic viewer. Use the same Qt
version for Qt and VTK. The first dependency build can take considerable time.

### Windows: MSVC

Install [Python 3.12](https://www.python.org/downloads/),
[Git](https://git-scm.com/downloads), and
[Visual Studio or Build Tools](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation).
Select **Desktop development with C++**, including the x64 MSVC compiler,
Windows SDK, and **C++ CMake tools for Windows** (CMake and Ninja).

Install the native dependencies with [vcpkg](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started).
Run this setup once; if vcpkg is already installed, use its existing folder:

```powershell
git clone https://github.com/microsoft/vcpkg.git C:/opt/vcpkg
C:/opt/vcpkg/bootstrap-vcpkg.bat
C:/opt/vcpkg/vcpkg.exe install qtbase hdf5 eigen3 "gmsh[occ]" ftxui "vtk[qt]" --triplet=x64-windows --overlay-ports=./vcpkg-ports
```

Build and install FEM in its own Python environment:

```powershell
py -3.12 -m venv .venv
. ./scripts/setup_msvc_windows.ps1
./.venv/Scripts/python.exe -m pip install . --config-settings=cmake.define.FEM_PERIODIC_MODE_VIEWER_WITH_VTK=ON
./.venv/Scripts/python.exe -m fem info
```

The setup script selects MSVC, Ninja, and the vcpkg CMake toolchain for this
terminal. For another vcpkg folder, use
`. ./scripts/setup_msvc_windows.ps1 -VcpkgRoot C:/path/to/vcpkg`.

### macOS: Clang

Install Apple's command-line tools, which provide Clang and the macOS SDK:

```sh
xcode-select --install
```

Install [Homebrew](https://brew.sh/) if needed, then install the dependencies:

```sh
brew install cmake ninja python@3.12 qtbase hdf5 eigen gmsh ftxui vtk
```

Use a recent Xcode command-line tools version with C++20 `std::format` support.
Homebrew's Gmsh includes OpenCASCADE; its VTK uses Qt 6. Build using Clang and
point CMake to the Homebrew dependency folders:

```sh
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate
export CC="$(xcrun --find clang)"
export CXX="$(xcrun --find clang++)"
export CMAKE_GENERATOR=Ninja
export CMAKE_PREFIX_PATH="$(brew --prefix qtbase);$(brew --prefix hdf5);$(brew --prefix eigen);$(brew --prefix gmsh);$(brew --prefix ftxui);$(brew --prefix vtk)"
python -m pip install . --config-settings=cmake.define.FEM_PERIODIC_MODE_VIEWER_WITH_VTK=ON
python -m fem info
```

### Linux: GCC

Use GCC 13 or newer for C++20 `std::format` support. These commands use
**Ubuntu 24.04, x86-64**, with GCC 13 and Python 3.12. Other distributions need
the equivalent compiler, Python headers, and X11/OpenGL development packages.

Install the compiler, build tools, Python, and libraries needed to build Qt:

```sh
sudo apt update
sudo apt install build-essential gcc g++ cmake ninja-build git curl zip unzip tar pkg-config python3-dev python3-venv autoconf autoconf-archive automake libtool xorg-dev libxkbcommon-dev libxkbcommon-x11-dev libxcb-cursor-dev libgl1-mesa-dev libglu1-mesa-dev libegl1-mesa-dev libfontconfig1-dev libwayland-dev
```

Use vcpkg to build Qt 6, HDF5, Eigen, Gmsh/OpenCASCADE, FTXUI, and VTK together.
This avoids mixing a distribution's Qt 5 VTK with Qt 6. Install vcpkg once:

```sh
git clone https://github.com/microsoft/vcpkg.git "$HOME/vcpkg"
"$HOME/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
```

Create a configuration for shared native libraries, then install the dependencies:

```sh
export VCPKG_ROOT="$HOME/vcpkg"
export CC=gcc
export CXX=g++
export CMAKE_GENERATOR=Ninja
mkdir -p build/triplets
cat > build/triplets/x64-linux-fem.cmake <<'EOF'
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)
EOF
"$VCPKG_ROOT/vcpkg" install qtbase hdf5 eigen3 'gmsh[occ]' ftxui 'vtk[qt]' --triplet=x64-linux-fem --overlay-triplets=./build/triplets --overlay-ports=./vcpkg-ports
```

Build and install FEM:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
export CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
export CMAKE_ARGS="-DCMAKE_TOOLCHAIN_FILE=$CMAKE_TOOLCHAIN_FILE -DVCPKG_TARGET_TRIPLET=x64-linux-fem -DVCPKG_OVERLAY_TRIPLETS=$PWD/build/triplets"
python -m pip install . --config-settings=cmake.define.FEM_PERIODIC_MODE_VIEWER_WITH_VTK=ON
python -m fem info
```

For ARM64 Linux, replace `x64` with `arm64` in the configuration, filename,
and commands. Keep the installed native dependencies available after installation;
macOS and Linux applications use those libraries at runtime.

In an activated environment, `uv pip install .` can replace `python -m pip install .`
with the same `--config-settings` option. With conda, activate a Python 3.12
environment instead of creating `.venv`, then use its `python -m pip install .`.
`pip install -r requirements.txt` or `uv sync` alone installs Python dependencies;
it does not build the native applications.

See the [native build guide](apps/native_build.md) for individual application
builds and native dependency sources.

## License

FEM source is [MIT licensed](LICENSE). Bundled native dependencies retain their own
licenses, included in the wheel. The Gmsh-linked calculator is distributed under
GPL-3.0-or-later; Qt is dynamically linked under LGPL-3.0. Release assets include
native dependency sources and build recipes.
