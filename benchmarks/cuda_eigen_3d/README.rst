Periodic 3D CPU/GPU eigenvalue benchmarks
.. csv-table::
   :header: "Fixture", "Mesh sizes (mm)", "Description"

   "iris", "4, 3, 2, 1.4", "Existing iris-loaded WR-90 cell"
   "siw", "2, 1.4, 1, 0.75", "New periodic via-loaded SIW cell"
   "perforated_waveguide", "5, 3.5, 2.5, 2", "New perforated WR-90 antenna cell"
   "dielectric_rod", "4, 3, 2, 1.7", "Existing PEC-patched rod antenna"

The SIW has a 1.6 mm dielectric substrate, PEC ground planes, two via fences,
and two interior loading vias. Transverse PML absorbs lateral leakage beyond
the fences. The perforated guide has a 22.86 x 10.16 mm air cavity beneath a
0.5 mm PEC top wall with an off-centre circular hole, plus air and PML above
the aperture. Both are numerical fixtures, not optimized device designs.
The filter fixtures measure unit-cell Bloch eigenvalues; they do not compute
a finite filter's S parameters. The antenna fixtures include complex PML
pencils, but this benchmark alone does not certify radiation predictions.

Measurements
------------

* Sixteen mesh cases (four densities per geometry): four requested eigenpairs, ``ncv=80``, original-pencil relative
  residual tolerance ``1e-8``, at most 24 refined-Arnoldi restarts.
* Twenty additional medium-mesh thread cases: BLAS/OpenMP limits of
  2, 4, 8, 10 and 20, in addition to the one-thread baseline for each geometry.
* Four medium-mesh cases with 14 eigenpair candidates and ``ncv=96``.
  Fourteen candidates match the candidate count used for a two-mode 3D FEM
  request. Baseline cases use one CPU thread.
* One excluded warmup per backend and case; three timed repetitions by
  default. Backend order rotates between repetitions. Cases run sequentially.
* Mesh assembly is measured separately and excluded from solve time.
  Solve times include conversion/upload, symbolic analysis, factorization,
  iteration, extraction, residual evaluation, and output download. CUDA
  synchronization brackets every timing. Factorization is repeated per solve.
* CPU ARPACK applies ``(A-sigma*B)^-1 B`` explicitly, supporting singular B.
  Its internal ARPACK tolerance is ``0.01*tol`` as in the existing benchmark;
  all backends must meet the same original-pencil residual target.
* Every timed sample must converge, contain the requested number of finite
  eigenpairs, meet the original-pencil residual tolerance, and match a converged
  CPU ARPACK warmup. As requested for this benchmark, agreement is defined
  by one-to-one matching of eigenvalue magnitudes, accepting symmetric
  forward/backward partners. The maximum relative magnitude difference is
  ``abs(abs(lambda_cpu)-abs(lambda_test))/max(1,abs(lambda_cpu)) < 1e-5``.
  Full complex eigenvalues and complex-set mismatch are retained for inspection.
* A speedup requires every timed sample of both the tested backend and CPU
  reference to pass. Failures and nonconvergence are retained, not discarded.
* Returned eigenpairs are measured before QEP, Gauss-law, PML-energy, and
  propagation-direction filtering. Numerical agreement on one mesh is
  distinct from physical mesh convergence.

Outputs include eigenvalues, residuals, steps/restarts, raw timings, median
and range, hardware/software versions, source commit, thread settings, exact
suite command, and initial/final host load and GPU usage. The report includes
CPU topology, clocks/cache, OS-visible RAM, GPU model/VRAM, selected device,
CUDA/driver and numerical-library versions; hardware.json retains the host
specification capture. The host peak RSS
is cumulative across all backends in a case; it is not a per-backend memory
comparison. A fresh worker process applies each case's thread environment
before importing numerical libraries. Shared-host timings should be read
with the recorded background load in mind.

Run
---

From the repository root, with CuPy and cuDSS installed:

.. code-block:: bash

   source .venv/bin/activate
   export CUDSS_LIBRARY="$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13/lib/libcudss.so.0"
   export LD_LIBRARY_PATH="$(dirname "$CUDSS_LIBRARY"):/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
   python -u benchmarks/cuda_eigen_3d/run_benchmarks.py \
       --output build/cuda-eigen-3d/run --gpu 3 --repeats 3 --timeout 1800
   python benchmarks/cuda_eigen_3d/validate_magnitudes.py \
       build/cuda-eigen-3d/run build/cuda-eigen-3d/validated
   python benchmarks/cuda_eigen_3d/audit_results.py build/cuda-eigen-3d/validated
   python benchmarks/cuda_eigen_3d/plot_results.py build/cuda-eigen-3d/validated

The runner uses magnitude matching and preserves complex-set mismatch as a
diagnostic. The initial 20-configuration batch used strict complex matching;
``validate_magnitudes.py`` revalidates every saved sample using the requested
magnitude criterion and writes a separate result tree without rerunning
solves or altering timings. It can also audit newer magnitude-validated runs.
``merge_results.py OUTPUT SOURCE...`` combines disjoint validated batches
and retains every batch's metadata. The full sweep has 40 configurations
and 360 timed solves at three repetitions per backend.
``audit_results.py`` checks every saved warmup/timed sample, compares thread
cases against the corresponding baseline mesh statistics and spectrum,
verifies the large cases exceed 10,000 reduced FEM DOFs, and checks production
eigensolver source hashes. The pencil dimension is twice the reduced FEM DOFs.

Use ``--case iris-coarse siw-coarse perforated_waveguide-coarse
dielectric_rod-coarse --repeats 1`` for a pilot. ``--timeout`` limits total
time per case including assembly, warmups and repeats (default 900 seconds).
Always select a new output directory to preserve previous results.

``progress.json`` is suitable for the Luna monitor described in the root
``AGENTS.md``. Per-case ``results.json`` is updated after every sample;
``summary.csv`` and ``summary.json`` accumulate completed cases. A failed
or timed-out process keeps its partial results and appears in progress data.

Related solver compatibility
----------------------------

See ``solver_compatibility.rst`` for the current CUDA integration boundaries
and what extending it to other mode solvers would require.

Recorded results
----------------

The `2026-10-13 report <results/2026-10-13/REPORT.rst>`_ includes all four
mesh densities, CPU thread comparisons, timing ranges, validation statistics,
and detailed CPU/GPU/software specifications. The same directory contains
summary.csv, plots, audit.json, and all per-case eigenvalues/residuals/timings.
The raw/ subdirectory preserves the original three measurement batches and
logs, including the initial strict-complex flags before magnitude revalidation.
The geometry scripts regenerate the assembled matrices; the recorded data
stores timings, mesh statistics, eigenvalues and residuals.
