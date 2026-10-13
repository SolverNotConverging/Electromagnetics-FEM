CUDA server GPU development tests
=================================

This document belongs to the ``dev`` branch. It records the proposed TITAN
server validation and development work. The public ``main`` branch contains
the CUDA implementation, user documentation, benchmark script, and completed
laptop measurements.

Current evidence
----------------

The RTX 4070 Laptop GPU runs cuDSS 0.8.0.10 with CUDA 13 and CuPy 14.2.
The antenna-pencil benchmark requests four eigenpairs with ``ncv=80`` and
``tol=1e-8``. Median solve timings include GPU transfer and factorization.
The CPU benchmark uses one BLAS thread.

.. list-table:: Completed laptop antenna measurements
   :header-rows: 1

   * - Pencil DOFs
     - CPU eigs (s)
     - CUDA/cuDSS (s)
   * - 4,962
     - 1.323
     - 0.181
   * - 7,440
     - 3.181
     - 0.534
   * - 15,628
     - 17.072
     - 0.802

GPU original-pencil residuals are below ``6e-13``. A separate full FEM solve
reproduces the two forward leaky modes and passes the QEP, Gauss, and PML
filters. The numerical regression run passed 67 tests with two slow tests
deselected. Server measurements remain to be collected.

Identify the TITAN and select the runtime
-----------------------------------------

Record the GPU, available memory, driver, and toolkit before installation:

.. code-block:: sh

    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv
    nvcc --version
    git rev-parse HEAD

CUDA 13 removed Maxwell, Pascal, and Volta library/compiler support.
TITAN X/Xp with Pascal and TITAN V with Volta therefore require CUDA 12.
TITAN RTX with Turing can use CUDA 12 or 13. cuDSS supports Pascal and newer
architectures when supported by the matching CUDA toolkit. The older Maxwell
TITAN X is outside cuDSS's supported architecture range.

Confirm the exact model using NVIDIA's `architecture matrix
<https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html>`_
and `cuDSS support information <https://docs.nvidia.com/cuda/cudss/#support>`_.

Linux CUDA 12 setup
-------------------

After cloning the repository, select ``dev`` and install the project's normal
dependencies using the root README. Download the CUDA 12 cuDSS runtime:

.. code-block:: sh

    git switch dev
    python -m pip install cupy-cuda12x
    mkdir -p build/cuda-deps
    curl -fL https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-0.8.0.10_cuda12-archive.tar.xz -o build/cuda-deps/cudss.tar.xz
    tar -xf build/cuda-deps/cudss.tar.xz -C build/cuda-deps
    export CUDSS_LIBRARY="$PWD/build/cuda-deps/libcudss-linux-x86_64-0.8.0.10_cuda12-archive/lib/libcudss.so"
    export LD_LIBRARY_PATH="$(dirname "$CUDSS_LIBRARY"):${LD_LIBRARY_PATH:-}"
    python -m pytest tests/libraries/periodic_eigensolver/test_cuda.py -q

For a CUDA 13-compatible GPU, select ``cupy-cuda13x`` and the ``_cuda13``
cuDSS archive. Install only one CuPy distribution in the environment.

Validation sequence
-------------------

1. Run the optional GPU tests. Confirm complex solves, singular mass matrices,
   repeated roots, and explicit nonconvergence work on the server runtime.
2. Reproduce the laptop problem sizes before increasing DOFs.
3. Run the full 3D antenna example with ``arnoldi_backend="cuda"`` and compare
   effective indices and physical mode filters with the CPU solve.
4. Increase one problem size at a time and record memory and solve time.

.. code-block:: sh

    python benchmarks/periodic_eigensolver/benchmark_cuda.py --sides 8 12 20 28 --repeats 5 --output build/cuda-titan-stencil-baseline
    python benchmarks/periodic_eigensolver/benchmark_cuda.py --problem antenna --mesh-mm 4 3 2 --ncv 80 --repeats 5 --output build/cuda-titan-antenna-baseline

The synthetic stencil has ``side**3`` pencil DOFs. The linearized FEM pencil
has twice the reduced FEM DOF count. The next stencil sizes are 40, 48, and
64, corresponding to 64,000, 110,592, and 262,144 pencil DOFs. Treat these as
candidate sizes; stop increasing the scale when factor fill-in or available
host/device memory makes a size impractical.

.. code-block:: sh

    python benchmarks/periodic_eigensolver/benchmark_cuda.py --sides 40 --repeats 5 --output build/cuda-titan-stencil-40
    python benchmarks/periodic_eigensolver/benchmark_cuda.py --sides 48 --repeats 5 --output build/cuda-titan-stencil-48
    python benchmarks/periodic_eigensolver/benchmark_cuda.py --sides 64 --repeats 5 --output build/cuda-titan-stencil-64

For the antenna, try a 1.5 mm maximum mesh size after the 2 mm case passes.
Use ``--modes 14 --ncv 80`` to measure the candidate count used by a two-mode
FEM solve. The benchmark's default four eigenpairs measure the eigenproblem
stage before direction, Gauss, and PML filtering.

Acceptance and reporting
------------------------

Run the benchmark without concurrent compute jobs. Record the GPU model,
driver/toolkit, CuPy/cuDSS/SciPy versions, source revision, thread settings,
DOFs, nonzero counts, and peak GPU memory. Compare both a one-thread CPU
baseline and the intended production CPU thread configuration, retaining
separate output folders.

Require all requested eigenpairs to converge at the chosen original-pencil
tolerance. Match CPU and GPU eigenvalues before reporting speedups. The
benchmark requires relative eigenvalue agreement below ``1e-5`` and records
each backend's residuals, timing range, convergence, and errors. For full FEM
solves, also compare direction, QEP/Gauss residuals, and PML fractions.

Keep CSV, JSON, timing plots, and failures with the server run's metadata.
Server results should be committed on ``dev`` for review. Public benchmark
snapshots can be promoted to ``main`` after their numerical checks and
measurement conditions have been reviewed.

Follow-up development
---------------------

Use server measurements to decide which changes to implement:

* Separate symbolic analysis, numeric factorization, triangular solve,
  orthogonalization, extraction, and transfer timings.
* Measure peak memory including cuDSS allocations and factor fill-in.
* Investigate factor/analysis reuse across frequency sweeps with fixed sparsity.
* Profile norm synchronization and orthogonalization kernels at large DOFs.
* Explore alternate subspace sizes or block operations if iteration overhead
  becomes significant.

Preserve the original-pencil convergence test and FEM physical filters while
evaluating these changes. Large-DOF performance and TITAN compatibility are
development validation items until the server runs complete.
