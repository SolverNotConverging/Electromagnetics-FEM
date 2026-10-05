cmake_minimum_required(VERSION 3.24)

include(cmake/WindowsToolchain.cmake)

project(FEM VERSION 1.1.0 LANGUAGES C CXX)

# A Command Line Tools upgrade can leave an obsolete, incomplete libc++ header
# directory beside the compiler. Apple Clang prefers that directory over the
# complete headers in the active SDK, causing even basic includes such as
# <array> to fail. Use the SDK copy only when the compiler-local copy is
# incomplete; healthy Xcode and Command Line Tools installations are unchanged.
if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    get_filename_component(FEM_CXX_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set(FEM_COMPILER_LIBCXX_DIR
        "${FEM_CXX_COMPILER_DIR}/../include/c++/v1")
    if(NOT EXISTS "${FEM_COMPILER_LIBCXX_DIR}/array")
        execute_process(
            COMMAND xcrun --sdk macosx --show-sdk-path
            OUTPUT_VARIABLE FEM_MACOS_SDK
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE FEM_XCRUN_RESULT
        )
        set(FEM_SDK_LIBCXX_DIR "${FEM_MACOS_SDK}/usr/include/c++/v1")
        if(FEM_XCRUN_RESULT EQUAL 0
           AND EXISTS "${FEM_SDK_LIBCXX_DIR}/array")
            message(STATUS
                "Using libc++ headers from the active macOS SDK: ${FEM_SDK_LIBCXX_DIR}")
            include_directories(BEFORE SYSTEM "${FEM_SDK_LIBCXX_DIR}")
        endif()
    endif()
endif()

if(DEFINED SKBUILD)
    set(BUILD_TESTING OFF CACHE BOOL "Do not compile C++ test programs into Python builds" FORCE)
endif()
include(CTest)

set(FEM_BUILD_PYTHON_DEFAULT OFF)
if(DEFINED SKBUILD)
    set(FEM_BUILD_PYTHON_DEFAULT ON)
endif()
option(FEM_BUILD_PYTHON
    "Build the Cython extension for the active Python interpreter"
    ${FEM_BUILD_PYTHON_DEFAULT})
if(FEM_BUILD_PYTHON)
    # scikit-build-core provides the active uv interpreter and makes build
    # requirements such as cython-cmake visible through CMAKE_PREFIX_PATH.
    find_package(Python 3.11...<3.14 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(Cython 3 MODULE REQUIRED)
    include(UseCython)

    set(FEM_CYTHON_SOURCE
        "${CMAKE_CURRENT_SOURCE_DIR}/libraries/periodic_eigensolver/src/periodic_eigensolver/_cython_kernels.pyx")
    cython_transpile(
        "${FEM_CYTHON_SOURCE}"
        LANGUAGE C
        MODULE_NAME periodic_eigensolver._cython_kernels
        CYTHON_ARGS -X language_level=3
        OUTPUT_VARIABLE FEM_CYTHON_GENERATED_SOURCE
    )
    Python_add_library(fem-cython-kernels MODULE
        "${FEM_CYTHON_GENERATED_SOURCE}"
        WITH_SOABI
    )
    set_target_properties(fem-cython-kernels PROPERTIES
        OUTPUT_NAME "_cython_kernels"
    )
    install(TARGETS fem-cython-kernels
        LIBRARY DESTINATION periodic_eigensolver
        RUNTIME DESTINATION periodic_eigensolver
    )
endif()

if(SKBUILD)
    set(FEM_NATIVE_INSTALL_DIR "fem/native/bin")
    set(FEM_NATIVE_BUNDLE_INSTALL_DIR "fem/native")
    set(FEM_NATIVE_LICENSE_INSTALL_DIR "fem/native/licenses")
else()
    set(FEM_NATIVE_INSTALL_DIR "bin")
    set(FEM_NATIVE_BUNDLE_INSTALL_DIR ".")
    set(FEM_NATIVE_LICENSE_INSTALL_DIR "share/licenses")
endif()

include(cmake/NativeRuntime.cmake)

# Release builds may reuse an already qualified, self-contained runtime bundle.
# Ordinary source installs compile the applications in this same CMake build.
set(FEM_NATIVE_BUNDLE "$ENV{FEM_NATIVE_BUNDLE}" CACHE PATH
    "Qualified native runtime bundle to install instead of compiling the apps")

option(CEM_BUILD_TRANSMISSION_LINE_CALCULATOR
    "Build the native transmission-line calculator" ON)
option(CEM_BUILD_FEM_WAVEGUIDE_SCATTERING_VIEWER
    "Build the FEM waveguide scattering viewer" ON)
option(CEM_BUILD_FEM_PERIODIC_MODE_VIEWER
    "Build the standalone periodic-FEM HDF5 viewer and inspector" ON)

if(FEM_NATIVE_BUNDLE)
    if(NOT EXISTS "${FEM_NATIVE_BUNDLE}/bin")
        message(FATAL_ERROR "FEM_NATIVE_BUNDLE is incomplete: ${FEM_NATIVE_BUNDLE}")
    endif()
    if(WIN32)
        if(NOT EXISTS "${FEM_NATIVE_BUNDLE}/build-manifest.json")
            message(FATAL_ERROR "Windows native bundles require an MSVC build-manifest.json. Repackage the native apps.")
        endif()
        file(READ "${FEM_NATIVE_BUNDLE}/build-manifest.json" FEM_BUNDLE_MANIFEST)
        string(JSON FEM_BUNDLE_TOOLCHAIN ERROR_VARIABLE FEM_BUNDLE_ERROR
            GET "${FEM_BUNDLE_MANIFEST}" toolchain)
        if(FEM_BUNDLE_ERROR OR NOT FEM_BUNDLE_TOOLCHAIN MATCHES "^MSVC")
            message(FATAL_ERROR "Windows native bundles must use MSVC. MinGW bundles are not supported; rebuild and repackage the apps.")
        endif()
    endif()
    install(DIRECTORY "${FEM_NATIVE_BUNDLE}/"
        DESTINATION "fem/native"
    )
elseif(CEM_BUILD_TRANSMISSION_LINE_CALCULATOR)
    add_subdirectory(apps/transmission_line_calculator)
endif()

if(NOT FEM_NATIVE_BUNDLE AND CEM_BUILD_FEM_WAVEGUIDE_SCATTERING_VIEWER)
    add_subdirectory(apps/fem_waveguide_scattering_viewer)
endif()

if(NOT FEM_NATIVE_BUNDLE AND CEM_BUILD_FEM_PERIODIC_MODE_VIEWER)
    add_subdirectory(apps/fem_periodic_mode_viewer)
endif()

if(NOT FEM_NATIVE_BUNDLE
   AND NOT CEM_BUILD_TRANSMISSION_LINE_CALCULATOR
   AND NOT CEM_BUILD_FEM_WAVEGUIDE_SCATTERING_VIEWER
   AND NOT CEM_BUILD_FEM_PERIODIC_MODE_VIEWER)
    message(WARNING "All native applications are disabled; there is nothing to build.")
endif()
