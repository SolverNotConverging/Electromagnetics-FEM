FEM tests
=========

Run ``python -m pytest`` from the repository root after ``uv sync --locked`` (or
``pip install -e .``). The layout mirrors the package: ``tests/common/``,
``tests/electrostatics/``, ``tests/waveguide/``, ``tests/periodic/`` and
``tests/scattering/``; cross-family checks live directly in ``tests/``.
``tests/conftest.py`` selects the Agg and offscreen Qt backends, so no display is
needed. Gmsh needs its platform runtime libraries; Linux typically needs libGLU.

Fast cross-family checks::

    python -m pytest tests/test_documentation.py tests/test_public_api.py tests/test_examples.py tests/common

Lifecycle and archive checks::

    python -m pytest tests/test_lifecycle.py tests/test_archive_contract.py

Run every example (opens no windows; outputs go to each example's ``outputs/``)::

    FEM_RUN_EXAMPLES=1 python -m pytest tests/test_examples.py

Ordinary regression tests simulate unavailable GPU support so defaults exercise
SciPy CPU consistently, including on GPU hosts. Set ``FEM_TEST_CUDA=1`` for the
separate real-device suite, including automatic GPU selection::

    FEM_TEST_CUDA=1 python -m pytest tests/test_cuda_solvers.py tests/common/test_gpu.py -v

This runs CPU/CUDA agreement checks for every solver family and the shared GPU
eigensolver; verify that the tests pass rather than skip. A fresh
``CUPY_CACHE_DIR`` also exercises kernel compilation.

Generated documentation::

    python scripts/generate_api_reference.py --check
    python scripts/check_documentation.py

Analytical checks live in `benchmarks <../benchmarks/README.md>`_; run them when
changing numerical methods.
