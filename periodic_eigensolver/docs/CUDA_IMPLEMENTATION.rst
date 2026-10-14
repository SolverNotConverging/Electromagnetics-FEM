CUDA refined Arnoldi implementation
===================================

The ``periodic_eigensolver`` CUDA backend solves a complex generalized sparse
eigenproblem using refined shift-invert Arnoldi. cuDSS performs numeric sparse
factorization and shifted linear solves on the GPU. CuPy stores the sparse
matrices and Arnoldi basis, launches the CUDA orthogonalization kernels, and
evaluates the original-pencil residuals. The small projected eigenproblem and
refined-vector SVDs use the shared CPU extraction routine.

This document describes the implementation in this repository. The
`setup and benchmark guide <../../benchmarks/periodic_eigensolver/README.md>`_
covers dependency installation and measured CPU/GPU timings.

Public entry point and source files
-----------------------------------

Users select CUDA through the same API as the CPU backends:

.. code-block:: python

    import numpy as np
    from scipy.sparse import diags, eye
    from periodic_eigensolver import solve_generalized

    A = diags(np.arange(1.0, 51.0) + 0.03j, format="csc")
    B = eye(50, dtype=np.complex128, format="csc")
    result = solve_generalized(
        A, B, sigma=10.2 - 0.01j, num_modes=3,
        ncv=32, tol=1e-10, backend="cuda",
    )
    print(result.eigenvalues, result.physical_residuals)

``solve_generalized`` validates the common options and dispatches to
``solve_cuda`` when ``backend="cuda"``. Imports of CuPy and the cuDSS runtime
are deferred until CUDA is requested. ``backend="auto"`` retains its existing
CPU selection.

.. list-table:: Implementation files
   :header-rows: 1
   :widths: 32 68

   * - File
     - Responsibility
   * - `src/refined.py <../src/refined.py>`_
     - Public dispatch, immutable result type, shared refined extraction,
       and candidate scoring.
   * - `src/cuda.py <../src/cuda.py>`_
     - GPU Arnoldi loop, host/device transfers, residual validation,
       breakdown handling, and restarts.
   * - `src/_cudss.py <../src/_cudss.py>`_
     - ``ctypes`` binding, matrix descriptors, factorization, shifted solves,
       stream association, and cleanup.
   * - `src/arnoldi_cuda.cu <../src/arnoldi_cuda.cu>`_
     - Complex double-precision projection and projection-subtraction kernels.

The spectral transformation
---------------------------

For input matrices A and B, the requested eigenpairs satisfy

.. math::

    A x = \lambda B x.

At a user-supplied shift sigma, the backend constructs the operator

.. math::

    K = A - \sigma B, \qquad T = K^{-1} B.

For a finite eigenvalue whose distance from the shift is nonzero,

.. math::

    T x = \mu x, \qquad \mu = (\lambda - \sigma)^{-1},
    \qquad \lambda = \sigma + \mu^{-1}.

Eigenvalues near sigma become large-magnitude eigenvalues of T. Applying T to
an Arnoldi vector requires one sparse product with B and one solve with K.
K is factorized once per ``solve_generalized`` call and reused throughout
all Arnoldi cycles and restarts.

A and B can be complex and nonsymmetric. B can be singular, provided K is
invertible. The projected extraction keeps finite transformed eigenvalues
with ``abs(mu) > 1e-14`` before mapping them back to lambda.

CPU and GPU responsibilities
----------------------------

.. list-table:: Execution placement
   :header-rows: 1
   :widths: 55 45

   * - Operation
     - Placement
   * - Input conversion to complex CSC and finite-value checks
     - CPU, NumPy/SciPy
   * - CSR matrices, shifted matrix K, and working vectors
     - GPU, CuPy
   * - cuDSS ordering and symbolic analysis
     - Host-side cuDSS analysis
   * - Numeric factorization and shifted linear solves
     - GPU, cuDSS
   * - Sparse products, orthogonalization, and vector normalization
     - GPU, CuPy and custom CUDA kernels
   * - Initial and restart random perturbations
     - CPU, seeded NumPy generator; uploaded to the GPU
   * - Small Hessenberg eigenvalues and refined SVDs
     - CPU, shared SciPy extraction
   * - Eigenvector reconstruction and physical residuals
     - GPU, CuPy
   * - Candidate selection and immutable returned arrays
     - CPU

The large Arnoldi basis remains on the GPU. Each cycle downloads the small
Hessenberg matrix, then uploads the refined coefficient matrix. Candidate
eigenvectors and residuals are downloaded for best-candidate selection and
restart construction.

cuDSS factorization and buffer ownership
----------------------------------------

``CudssFactor`` loads the library named by ``CUDSS_LIBRARY`` or located through
the system library lookup. It checks that cuDSS is version 0.8 or newer,
matching the CSR descriptor ABI used by the binding. NVIDIA binaries remain
external runtime dependencies.

The shifted matrix is converted to complex128 CSR, duplicates are combined,
and column indices are sorted. The wrapper creates:

* A cuDSS handle bound to the current CuPy CUDA stream.
* Configuration and solver-data objects.
* A general, full, zero-based CSR descriptor referencing the GPU matrix.
* Dense column-major descriptors for persistent right-hand-side and solution
  buffers.

The binding selects 32-bit or 64-bit index types to match the CSR arrays.
CuPy owns the arrays; cuDSS descriptors reference their device pointers.
The wrapper keeps those arrays alive while the descriptors and factors exist.

Initialization runs analysis and numeric factorization, synchronizes the
stream, and checks ``CUDSS_DATA_INFO`` for numerical failure. Each application
of T copies ``B @ v`` into the persistent right-hand-side buffer and executes
the cuDSS solve phase, including its refinement sub-phase. A copy of the
solution becomes the Arnoldi work vector so the next solve can reuse the
persistent buffer safely.

The factor must be used on its original CUDA stream. Its context manager
synchronizes pending work before destroying matrix descriptors, solver data,
configuration, and the library handle. cuDSS status failures raise errors;
the backend does not substitute a CPU solve.

On Windows, ``configure_dll_search`` registers the selected CUDA toolkit's
``bin`` and ``bin/x64`` directories and the cuDSS library directory. It keeps
those registrations alive and explicitly loads NVRTC's companion builtins
DLL, because runtime compilation can load it after earlier factors have
already been destroyed.

Arnoldi basis construction
--------------------------

Let n be the pencil dimension and m the requested Arnoldi subspace size,
``ncv``. The backend allocates a column-major complex128 basis of shape
``(n, m + 1)`` and a Hessenberg array of shape ``(m + 1, m)``. The default
subspace size is ``max(20, 4 * num_modes + 8)``, bounded by n; explicit values
are raised to at least ``num_modes + 2`` before the same upper bound is applied.

The initial vector is an all-ones vector with a seeded complex random
perturbation of amplitude ``1e-3``. It is normalized on the GPU. For column j,
the algorithm computes

.. math::

    w = T v_j.

It then performs two passes of classical Gram-Schmidt against the active
basis. In each pass,

.. math::

    c = V_{j+1}^{*} w, \qquad
    w \leftarrow w - V_{j+1} c, \qquad
    h_{0:j,j} \leftarrow h_{0:j,j} + c.

The ``project`` CUDA kernel launches one block of 256 threads per active
basis column. Threads accumulate conjugate complex dot products over rows,
then reduce their partial sums in shared memory. ``subtract_projection``
assigns one thread to each vector row and subtracts the weighted basis
columns. Column-major storage gives contiguous accesses within each basis
column. CuPy's ``RawModule`` compiles these kernels at runtime with NVRTC.

After both passes, CuPy computes the work-vector norm. The scalar norm is
copied to the host for the breakdown decision, recorded as the Hessenberg
subdiagonal, and used to normalize the next basis vector. This produces the
rectangular Arnoldi relation

.. math::

    T V_k \approx V_{k+1} \overline{H}_k.

A norm at or below ``1e-14`` triggers breakdown handling. For multiple
requested modes, the backend attempts to continue with a seeded random vector
orthogonalized twice against the current basis. This supports independent
vectors for repeated roots. If fewer than ``num_modes`` Arnoldi steps can be
completed, the solve raises an error.

Refined Ritz extraction
-----------------------

The backend downloads only the active Hessenberg array and calls the same
``_refined_candidate_data`` routine used by the CPU implementation. An identity
basis and small identity pencil let this routine return coefficient vectors
in projected coordinates. Its temporary small-pencil residuals are discarded;
the physical residuals are computed later with the actual A and B.

The routine finds eigenvalues of the leading square Hessenberg block, maps
them back to lambda, and selects candidates by distance from sigma. For a
transformed Ritz value mu, the refined coefficient vector solves

.. math::

    y = \mathop{\mathrm{argmin}}_{\|y\|_2=1}
    \left\|\left(\overline{H}_k - \mu
    \begin{bmatrix} I_k \\ 0 \end{bmatrix}\right)y\right\|_2.

It is the smallest right singular vector of that rectangular residual matrix.
Roundoff-identical repeated roots are extracted as independent vectors from
the corresponding right-singular subspace. The projected residual reported
in the result is the associated smallest singular value.

The coefficients are uploaded, reconstructed as ``X = V_k @ Y`` on the GPU,
and normalized column by column.

Convergence and restart policy
------------------------------

Each reconstructed vector is checked against the original generalized pencil:

.. math::

    r_i = \frac{\|A x_i - \lambda_i B x_i\|_2}
    {\|A x_i\|_2 + |\lambda_i|\,\|B x_i\|_2}.

The implementation substitutes a denominator of one when the denominator is
zero. Nonfinite residuals raise an error. A cycle converges when every
requested candidate has ``r_i <= tol``.

Across cycles, the best candidate set is selected first by its maximum
physical residual and then by the norm of its residual vector. A restart
sums the best set's unconverged eigenvectors and adds a new seeded complex
random perturbation of amplitude ``1e-3``. The next cycle builds a fresh basis
from that vector while reusing the existing cuDSS factorization.

There are at most ``max_restarts + 1`` cycles. If the restart budget is
exhausted, the best candidate set is returned with ``converged=False``.
``ArnoldiResult`` contains immutable NumPy eigenvalues, eigenvectors, physical
and projected residuals, restart and step counts, and ``backend="cuda"``.

Periodic FEM integration
------------------------

The 2D and 3D periodic FEM solvers pass their linearized quadratic-eigenproblem
pencils to this library. If q is the reduced FEM unknown count, that pencil
has dimension ``n = 2*q``. The eigenvalue is the effective index, so sigma is
the requested ``neff_guess``.

.. code-block:: python

    modes = solver.solve(
        neff_guess=0.5 - 0.02j,
        num_modes=2,
        eigensolver="refined",
        backend="cuda",
        ncv=80,
        eigensolver_tolerance=1e-8,
        max_refinements=0,
    )

``eigensolver="auto"`` also chooses refined Arnoldi when CUDA is explicitly
requested. Dense QZ remains a CPU operation. After the library returns,
the FEM solver applies its QEP residual, direction, PML-energy, and, in 3D,
Gauss-law filters. It requests additional eigenpair candidates so these
filters can select the requested number of physical modes.

Memory and timing
-----------------

The main basis allocation consumes approximately

.. math::

    16\,n\,(\mathrm{ncv}+1)\ \text{bytes}.

Sparse matrices, cuDSS factors, reconstructed vectors, and work buffers add
to this allocation. Factor fill-in depends on the mesh connectivity and
ordering, so DOF count alone does not determine GPU memory demand.

The implementation synchronizes for each Arnoldi work-vector norm, after
factorization, during host transfers, and before factor cleanup. Small
problems can be dominated by launch, synchronization, and analysis overhead.
Larger problems can benefit from GPU factorization and vector operations.

``benchmark_cuda.py`` measures synchronized solve times including input
transfer, cuDSS analysis/factorization, Arnoldi, extraction, and output
transfer. Mesh assembly and warmup are excluded. The
`saved laptop results <../../benchmarks/reference_results/cuda_laptop_antenna/README.md>`_
record both reduced FEM and linearized-pencil DOF counts.

Verification
------------

`test_cuda.py <../../tests/libraries/periodic_eigensolver/test_cuda.py>`_
checks complex nonsymmetric linear solves, known generalized eigenvalues,
singular mass matrices, independent vectors for repeated roots, and explicit
nonconvergence. The eigenproblem tests call the public ``solve_generalized`` API. They
skip on CPU-only installations unless the cuDSS runtime is configured.

A full 3D dielectric-rod antenna solve has also reproduced the CPU's two
forward leaky modes while passing the FEM residual, Gauss-law, and PML filters.
The benchmark compares returned eigenvalues with CPU ARPACK and records
original-pencil residuals before reporting speedups.
