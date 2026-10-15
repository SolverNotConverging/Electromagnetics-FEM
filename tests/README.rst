FEM tests
=========

Run ``python -m pytest`` from the repository root. Solver tests live under
``tests/fem/`` and shared library tests under ``tests/libraries/``.

Use the project virtual environment (``.venv/Scripts/python.exe`` on Windows).
Install dependencies with ``uv sync --locked``. Gmsh needs its platform runtime
libraries; Linux typically needs libGLU. Plotting tests use the Agg backend.

Fast cross-family checks::

    python -m pytest tests/test_fem_documentation.py tests/test_material_first_public_api.py tests/libraries/fem_common

Lifecycle and archive checks::

    python -m pytest tests/test_fem_lifecycle.py tests/test_fem_archive_contract.py

Full regression suite::

    python -m pytest

Ordinary regression tests simulate unavailable GPU support so defaults exercise
SciPy CPU consistently, including on GPU hosts. Set ``FEM_TEST_CUDA=1`` for
the separate real-device suite, including automatic GPU selection.

CUDA tests skip when their hardware/runtime prerequisites are absent. Native
C++ tests run through CTest in a configured native build; see the
`native build guide <../apps/native_build.md>`_. Analytical checks live in
`benchmarks <../benchmarks/README.md>`_. Run those when changing numerical methods.

Real-device CUDA qualification on Windows (CuPy and cuDSS must be installed)::

    $env:FEM_TEST_CUDA = '1'
    $env:CUDSS_LIBRARY = & .venv/Scripts/python.exe -c "from fem_common._cudss import find_cudss_library; print(find_cudss_library())"
    .venv/Scripts/python.exe -m pytest tests/fem/test_cuda_solvers.py tests/libraries/fem_common/test_gpu.py -v

This runs CPU/CUDA agreement checks for electrostatics, waveguide modes,
2D/3D periodic modes, scattering, and the shared GPU eigensolver. Verify that the
tests pass rather than skip. A fresh ``CUPY_CACHE_DIR`` also exercises kernel
compilation instead of reusing cached CUDA kernels.

Generated documentation and metadata::

    python scripts/sync_metadata.py --check
    python scripts/generate_api_reference.py --check
    python scripts/check_documentation.py

For a restricted Windows environment, set TEMP and TMP to a writable local
directory and pass a fresh ``--basetemp=build/pytest-run`` directory. Pytest
clears the selected base directory, so use a directory dedicated to that run.

See the `installation guide <../docs/installation.md>`_ for setup.
