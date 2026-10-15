# Installation

These instructions apply to version 2.0.0. Build this checkout or install a
matching locally built 2.0 wheel. See [migration](migration-2.0.md) when updating.

## Solvers only

Install without compiling native applications or Python extensions:

```sh
python -m pip install . --config-settings=cmake.define.FEM_BUILD_NATIVE_APPS=OFF
```

This configuration requires no C/C++ compiler, Qt, VTK or vcpkg. Python's
Gmsh and h5py runtime dependencies are still installed. For editable development,
use the same option with `pip install -e .`. For a dependency-only checkout,
run `uv sync --locked` and run examples directly from the checkout.

## Install a locally built wheel

Use Python 3.11–3.13 and a wheel matching your operating system and architecture.
The complete Windows wheel includes the native applications and runtime libraries:

```powershell
python -m pip install .\electromagnetics_fem-2.0.0-py3-none-win_amd64.whl
```

For a locally built Apple silicon wheel on macOS 15 or later:

```sh
python -m pip install ./electromagnetics_fem-2.0.0-py3-none-macosx_15_0_arm64.whl
```

These commands refer to local build outputs, not published release assets. See the
[native build guide](../apps/native_build.md) for building and packaging the apps.
The native viewers require a working graphics driver. Solver-only wheels omit
the native applications; use the source-install command above to build one.

With uv, use `uv pip install` instead of `python -m pip install` in the activated
environment. With conda, use the active environment's Python.

## Download and run examples

Download this repository using GitHub's **Code → Download ZIP**, and extract it.
After installing FEM, open a terminal in the extracted folder and run:

```sh
python fem_waveguide_modes/examples/1d_parallel_plate_waveguide.py
python fem_periodic_modes/examples/2d_uniform_cell.py
```

Examples import the source solver directly. The installed distribution supplies native applications. Installation also lets you import FEM
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

A default source installation builds all native applications.
Download and extract the repository, then run the commands below from its root
folder. Use Python 3.12 for these instructions. Python dependencies are installed
automatically by `pip install .`; no `requirements-dev.txt` is needed.

Native dependencies are a C++20 compiler, CMake 3.24+, Ninja, Qt 6.2+,
HDF5, Eigen 3.4+, Gmsh with OpenCASCADE, and FTXUI. The Windows and macOS commands also install
VTK with **Qt 6** support and enable the 3D periodic viewer. Linux setup below
explains how to enable that optional viewport. Use the same Qt version for Qt
and VTK. The first dependency build can take considerable time.

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

### Linux: GCC and system packages

Use GCC 13 or newer for C++20 `std::format` support. These commands target
**Ubuntu 24.04** with GCC 13 and Python 3.12; on other distributions, use the
equivalent development packages from `dnf`, `pacman`, or your package manager.

Install the compiler, build tools, and native libraries through apt:

```sh
sudo apt update
sudo apt install build-essential gcc-13 g++-13 cmake ninja-build git pkg-config python3.12-dev python3.12-venv qt6-base-dev libqt6opengl6-dev libhdf5-dev libeigen3-dev libgmsh-dev libgl1-mesa-dev libglu1-mesa-dev
```

The calculator also needs FTXUI. If your distribution provides `libftxui-dev`,
install it with `sudo apt install libftxui-dev`. If the package is unavailable,
build [FTXUI](https://github.com/ArthurSonzogni/FTXUI) into a local prefix:

```sh
git clone --branch v5.0.0 --depth 1 https://github.com/ArthurSonzogni/FTXUI.git build/deps/ftxui
cmake -S build/deps/ftxui -B build/deps/ftxui-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/build/deps/install" \
  -DFTXUI_BUILD_EXAMPLES=OFF -DFTXUI_BUILD_DOCS=OFF -DFTXUI_BUILD_TESTS=OFF
cmake --build build/deps/ftxui-build --parallel
cmake --install build/deps/ftxui-build
export CMAKE_PREFIX_PATH="$PWD/build/deps/install${CMAKE_PREFIX_PATH:+;$CMAKE_PREFIX_PATH}"
```

Build and install FEM with the 2D periodic viewer:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
export CC=gcc-13
export CXX=g++-13
export CMAKE_GENERATOR=Ninja
python -m pip install . --config-settings=cmake.define.FEM_PERIODIC_MODE_VIEWER_WITH_VTK=OFF
python -m fem info
```

The 3D periodic viewer requires **VTK 9.2+ built against Qt 6**.
[Ubuntu 24.04's VTK package](https://packages.ubuntu.com/noble/libvtk9-qt-dev)
is VTK 9.1 with Qt 5, so the commands above disable that optional viewport.
On distributions offering a compatible VTK development package, install it with
the system package manager and replace `OFF` with `ON`. Alternatively, build
VTK 9.2+ from upstream with `VTK_GROUP_ENABLE_Qt=YES` and `VTK_QT_VERSION=6`,
add its install prefix to `CMAKE_PREFIX_PATH`, and reinstall with `ON`.
The FEM solvers can still compute 3D modes with the 2D-only viewer installed.

Keep the installed native dependencies available after installation; macOS and
Linux applications use those libraries at runtime.

In an activated environment, `uv pip install .` can replace `python -m pip install .`
with the same `--config-settings` option. With conda, activate a Python 3.12
environment instead of creating `.venv`, then use its `python -m pip install .`.
`pip install -r requirements.txt` or `uv sync` alone installs Python dependencies;
it does not build the native applications.

See the [native build guide](../apps/native_build.md) for individual application
builds and native dependency sources.

## CPU and GPU backends

Every solver defaults to `backend="auto"`. It checks that CuPy, an NVIDIA
GPU and cuDSS can complete a small sparse solve, then selects GPU if available
or SciPy CPU otherwise. `backend="cpu"` forces CPU execution; `backend="gpu"`
requires GPU support and reports runtime failures. Numerical failures during
an actual solve propagate without silently retrying on CPU.

Periodic eigensolvers accept `eigensolver="auto"`, `"dense"`, or `"sparse"`.
Dense eigensolving requires `backend="cpu"`; GPU uses sparse shift-invert.
SciPy/ARPACK manages sparse eigenvalue iteration on the host; cuDSS performs
GPU factorizations and linear solves. Mesh assembly remains on the CPU.

Shared source files are `fem_common/linalg.py` (eigenvalue iteration),
`fem_common/gpu.py` (GPU operators) and `fem_common/_cudss.py` (cuDSS bindings).
The custom refined Arnoldi and its Cython/CUDA kernels have been removed.

### CUDA acceleration (Windows and Linux)

CUDA is optional and requires an NVIDIA GPU, a compatible NVIDIA driver,
a CUDA 12 or CUDA 13 toolkit, one matching CuPy package, and **cuDSS >= 0.8**.
Install the driver/toolkit using [NVIDIA's CUDA downloads](https://developer.nvidia.com/cuda-downloads).
In the Python environment where FEM is installed, install exactly one CuPy package:

```sh
# CUDA 12:
python -m pip install "cupy-cuda12x>=13.6,<15"
# Or, for CUDA 13:
python -m pip install "cupy-cuda13x>=14,<15"
```

For a source installation, `python -m pip install ".[cuda12]"` or
`python -m pip install ".[cuda13]"` installs FEM and the corresponding CuPy
extra; retain the platform's CMake settings from the source-build instructions.

Install cuDSS separately using [NVIDIA's installation instructions](https://docs.nvidia.com/cuda/cudss/getting_started.html),
or download and extract a matching CUDA-version archive from the
[Linux x86-64 redistributables](https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/)
or [Windows x86-64 redistributables](https://developer.download.nvidia.com/compute/cudss/redist/libcudss/windows-x86_64/).
Set the library path in each terminal used to run FEM (replace the example paths
with your extracted archive's actual paths):

```sh
# Linux:
export CUDSS_LIBRARY="/absolute/path/to/cudss/lib/libcudss.so"
export LD_LIBRARY_PATH="$(dirname "$CUDSS_LIBRARY"):${LD_LIBRARY_PATH:-}"
```

```powershell
# Windows:
$env:CUDSS_LIBRARY="C:/path/to/cudss/bin/cudss64_0.dll"
```

CuPy and cuDSS are not bundled in FEM wheels. To require GPU execution:

```python
modes = solver.solve(neff_guess=0.5 - 0.02j, num_modes=2,
                     eigensolver="sparse", backend="gpu")
```

See [GPU tests](../tests/README.rst) for device verification commands.

cuDSS library discovery checks `CUDSS_LIBRARY` first, then system paths,
NVIDIA packages in the active Python environment, and the repository
`build/cuda-deps` directory. With a local installation in that directory,
`backend="gpu"` works without setting an environment variable.

Waveguide scattering also supports cuDSS acceleration:

```python
simulation.solve_modes(backend="gpu", num_modes=1, neff_guess=1.0)
simulation.set_incident_mode(0)
result = simulation.solve(backend="gpu")
# Or accelerate both stages at every frequency:
sweep = simulation.sweep(frequencies_hz, backend="gpu")
```

Scattering assembly and ARPACK lead-mode iteration run on the CPU;
cuDSS performs the sparse factorizations and shifted/direct solves on CUDA.
Lead-mode iteration transfers vectors between host and device.
