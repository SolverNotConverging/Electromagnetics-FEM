periodic eigensolver
====================

Source is under ``src/``. Install only the Python dependencies from the
`root README <../README.md>`_, then import this folder directly.

* `Guide <docs/guide.rst>`_
* `API reference <docs/API_REFERENCE.rst>`_
* `CUDA implementation <docs/CUDA_IMPLEMENTATION.rst>`_

Select a backend through the shared solver API:

.. code-block:: python

    from periodic_eigensolver import solve_generalized

    result = solve_generalized(
        A, B, sigma=shift, num_modes=4, backend="cuda",
    )

Supported backends are ``auto``, ``python``, ``cython``, and ``cuda``.
``auto`` keeps CPU selection. The CUDA backend is implemented in this package
and uses optional CuPy and cuDSS >= 0.8. Set ``CUDSS_LIBRARY`` to the cuDSS
shared library path; see `CUDA setup and benchmarks
<../benchmarks/periodic_eigensolver/README.md>`_.

Run the same example on either CPU or GPU:

.. code-block:: console

    python periodic_eigensolver/examples/diagonal_pencil.py --backend auto
    python periodic_eigensolver/examples/diagonal_pencil.py --backend cuda
