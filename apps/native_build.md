# Build FEM from source

FEM source builds compile the native applications and the Cython eigensolver.
The [root README](../README.md) describes installation and example usage.

## Windows: MSVC and vcpkg

Install Visual Studio with the **Desktop development with C++** workload, Git,
and vcpkg at `C:/opt/vcpkg`. Use x64 MSVC and the `x64-windows` triplet.
From the FEM repository folder:

```powershell
C:/opt/vcpkg/vcpkg.exe install qtbase hdf5 eigen3 "gmsh[occ]" ftxui "vtk[qt]" --triplet=x64-windows --overlay-ports=./vcpkg-ports
. ./scripts/setup_msvc_windows.ps1
python -m pip install .
```

The setup script selects MSVC, Ninja, and the vcpkg CMake toolchain. For an
individual native build, use:

```powershell
. ./scripts/setup_msvc_windows.ps1
cmake -S . -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release -DFEM_BUILD_PYTHON=OFF
cmake --build build/native --parallel
```

## macOS and Linux

Use a C++20 compiler with `std::format` support, CMake 3.24 or newer, Ninja,
Qt 6.2 or newer (Widgets and Concurrent), HDF5 with C headers, Eigen 3.4 or newer,
Gmsh with OpenCASCADE and its C++ headers/library, and FTXUI. VTK built with Qt
support enables the 3D periodic viewer. Install these dependencies with your
platform's package manager or build them from their upstream sources.

If they are outside CMake's normal search paths, set `CMAKE_PREFIX_PATH` to their
installation prefixes before installing:

```sh
export CMAKE_PREFIX_PATH="/path/to/qt;/path/to/hdf5;/path/to/other/dependencies"
python -m pip install .
# Or: uv pip install .
```

For a 2D-only periodic viewer, add
`--config-settings=cmake.define.FEM_PERIODIC_MODE_VIEWER_WITH_VTK=OFF` to the pip
command. Application-specific platform instructions:

- [Transmission-line calculator](transmission_line_calculator/README.rst)
- [Periodic viewer](fem_periodic_mode_viewer/README.rst)
- [Scattering viewer](fem_waveguide_scattering_viewer/README.rst)

## Release dependency sources

The [Gmsh overlay](../vcpkg-ports/README.rst) enables meshing, Eigen, and
OpenCASCADE compatibility. vcpkg records installed versions, source URLs, and
hashes in its SPDX records. Cached upstream archives are in
`C:/opt/vcpkg/downloads`; installed libraries are in
`C:/opt/vcpkg/installed/x64-windows`.

`scripts/package_native_windows.py` preserves dependency recipes and licenses,
and copies cached source archives with verifiable hashes. Release source assets
contain `SOURCE_INDEX.md`; the wheel contains recipes, licenses, and
`build-manifest.json` with the toolchain and vcpkg revision. The index explicitly
identifies any unexpanded source variables in installed vcpkg SPDX records.
Use the recorded versions and recipes when rebuilding dependencies.

FEM application sources and packaging scripts are in this repository. Release
notes identify the corresponding source commit. Original FEM source is MIT
licensed; dependency license terms accompany the wheel and source assets.
