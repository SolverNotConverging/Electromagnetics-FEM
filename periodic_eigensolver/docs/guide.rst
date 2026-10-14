Periodic eigensolver
====================

``periodic_eigensolver`` provides a shared implementation of refined shift-and-invert Arnoldi for generalized sparse pencils. The refined Ritz vectors use the paper-defined small rectangular Hessenberg residual, while convergence is validated against the original ``A x = lambda B x`` pencil.

The optional Cython extension accelerates large complex orthogonalization and residual-norm kernels. If it is unavailable, the same public API uses a portable NumPy/SciPy implementation.

First example
-------------

With the packages installed, run this from the repository root::

   python periodic_eigensolver/examples/diagonal_pencil.py

The example prints eigenvalues near 3 and 4 and their original-pencil residuals.
Open the `example <../examples/diagonal_pencil.py>`_
to see how to supply the matrices and choose the spectral shift.

For environment setup, native extension requirements, and wheel builds, see
the `root README <../../README.md>`_.

Backend contract
----------------

The primary API is:

.. code-block:: python

   from periodic_eigensolver import solve_generalized

   result = solve_generalized(
       A,
       B,
       sigma=shift,
       num_modes=4,
       backend="auto",
   )

Public backend names are ``auto``, ``cython``, ``python``, and ``cuda``. ``auto`` selects the CPU Cython extension when it can be imported and otherwise selects Python/NumPy. An explicitly requested but unavailable Cython backend raises ``ImportError``; an installed but broken native binary is never silently hidden by the fallback.

The returned immutable ``ArnoldiResult`` contains eigenvalues, eigenvectors, physical original-pencil residuals, projected smallest singular values, restart and Arnoldi-step counts, convergence status, and the resolved backend. When the restart budget is exhausted it returns the best candidate set found with ``converged=False``.

The sparse shift-and-invert factorization and sparse matrix products remain in SciPy. The native extension owns the repeated complex two-pass Arnoldi orthogonalization and batched final residual norms. Refined extraction follows the paper's ``(k + 1) x k`` Hessenberg SVD. Roundoff-identical repeated roots are refined as an independent right-singular subspace; distinct clustered roots retain their unrestricted refined vectors even when a non-normal pencil makes those vectors nearly parallel. Every returned mode is validated with the original sparse pencil residual.

CUDA backend
------------

The `CUDA implementation document <CUDA_IMPLEMENTATION.rst>`_ describes the
algorithm, custom kernels, cuDSS binding, data transfers, and restart policy.

``backend="cuda"`` uses optional CuPy and cuDSS >= 0.8. Set ``CUDSS_LIBRARY``
to the absolute path of ``cudss64_0.dll`` (Windows) or ``libcudss.so`` (Linux).
Install one matching CuPy distribution: ``cupy-cuda12x`` or ``cupy-cuda13x``.
The corresponding project extras are ``cuda12`` and ``cuda13``.

Numeric sparse factorization and shifted solves use cuDSS on the device.
Sparse products, custom CUDA two-pass orthogonalization, basis storage,
vector reconstruction, and original-pencil residuals also use the device.
cuDSS performs its symbolic analysis on the host; the small Hessenberg
eigendecomposition and refined SVDs use the shared CPU extraction code.
Results return as immutable NumPy arrays, matching the CPU API. The CUDA
backend supports complex nonsymmetric pencils and singular B when the
shifted pencil is invertible. It raises on missing dependencies or cuDSS
failure and reports nonconvergence through ``converged=False``.

In periodic FEM solvers, select ``eigensolver="refined"`` and
``backend="cuda"``. ``eigensolver="auto"`` also selects refined
Arnoldi when CUDA is explicitly requested. Dense QZ remains a CPU method.

See `CUDA setup and benchmarks
<../../benchmarks/periodic_eigensolver/README.md>`_ for reproducible commands,
DOF definitions, runtime requirements, and saved benchmark measurements.

Controlled performance gate
---------------------------

Performance is intentionally kept out of ordinary CI because BLAS thread counts and shared-runner load make timing assertions noisy. A release runner with the native extension installed can execute the specified gate with:

.. code-block:: console

   python benchmarks/periodic_eigensolver/benchmark_mgs.py --enforce

The script forces the common BLAS thread-control variables to one before importing NumPy. At ``n=100000, ncv=32`` it requires native MGS to be at least 20% faster than the optimized Fortran-order Python fallback and at least 2x faster than the former C-order layout.

Run the complementary LU-inclusive release check with:

.. code-block:: console

   python benchmarks/periodic_eigensolver/benchmark_end_to_end.py --enforce

It uses alternating paired Python/Cython solves on a deterministic complex five-point pencil (``n=16384``, ``ncv=16``, two modes), reports the fraction spent in SuperLU and whether the run is LU-dominated, checks eigenvalue and invariant-subspace agreement, and rejects a median end-to-end Cython/Python ratio above ``1.05``. Both performance commands are intended for an otherwise idle controlled runner, not ordinary CI.
