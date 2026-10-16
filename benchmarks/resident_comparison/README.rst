Resident CUDA comparison on the original four 3D cells
==========================================================

This compares main's CPU ARPACK, its retained hybrid GPU control, and its
production resident GPU eigensolver on identical assembled quadratic pencils.
The iris, SIW, perforated waveguide and PEC-patched dielectric rod reproduce the
physical definitions and four mesh targets from the 2026-10-13 benchmark.
Actual reduced FEM DOFs are recorded rather than inferred from mesh targets.

Recorded results
----------------

``results/2026-10-16/REPORT.rst`` contains the validated TITAN RTX comparison:
19 mesh configurations, 171 timed solves, and 57 excluded warmups. The
perforated waveguide extends to 108,261 reduced FEM DOFs. This directory
includes plots, CSV summaries, raw per-case eigenvalues/residuals, the audit,
and original execution logs. ``sweeps/`` retains each of the three runs'
metadata and completion records. Absolute paths in metadata describe the
original machine; the copied result files are self-contained.

The preliminary pilot is excluded because its hybrid readiness-check timing
differed from the final method. CUDA caches and local environments are not
part of the recorded results.

Method
------

All three paths factor the reduced n-by-n Q(shift), instead of the old suite's
2n-by-2n companion factorization. Consequently the old reported CPU/GPU ratios
are not a like-for-like measure of the resident-kernel change. The paired
hybrid/resident comparison here retains the same current operator and factors.
The control uses main's retained iterate_hybrid, not an entire historical build.

Settings: four raw eigenpairs, ncv=80, tolerance 1e-8, random seed 0,
max_iterations=2000, complex double precision, one BLAS/OpenMP thread.
The iteration cap counts ARPACK iterations on CPU and operator applications
on GPU; it is a safety limit, not an equal-work stopping criterion.
Both GPU paths include the same readiness probe in their solve timing.
Each call performs fresh analysis/factorization. Timing includes operator
construction, transfers, iteration and eigenpair export. The hybrid control
exports the full companion vectors and then selects the physical block; the
production resident API exports just the physical block. Resident status bytes
and iteration counts are saved. CPU/GPU initial random vectors follow each
production implementation's RNG. Matrix hashes identify each shared pencil.

Mesh assembly, initial runtime readiness and one per-backend warmup are excluded.
Three timed repetitions rotate backend order. Each configuration uses a fresh
process, and configurations run sequentially on the selected GPU.
Independent host QEP residual/magnitude validation runs after the timer for all
backends. GPU iteration also includes its required internal QEP residual checks;
the CPU ARPACK stage does not include this additional physical-residual pass.
One-to-one magnitude matching accepts symmetry-related roots as requested.
Every sample must return four finite roots, residual <=1e-8, and magnitude
agreement <1e-5 to the valid CPU warmup. Failures remain in the saved records.

These are eigenvalue-stage timings before physical mode filtering, field
reconstruction and adaptive estimation. They do not measure the complete
resident 3D public solve or certify physical mesh convergence. Thread scaling
is not repeated in this kernel comparison; the CPU baseline has one thread.

Run from the repository with its CUDA-enabled environment:

.. code-block:: bash

   CUDSS_LIBRARY="$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13/lib/libcudss.so.0" \
   LD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
   .venv/bin/python benchmarks/resident_comparison/run.py \
       --output outputs/benchmarks/resident_comparison_20261016/full --gpu 3 --repeats 3
   .venv/bin/python benchmarks/resident_comparison/report.py \
       outputs/benchmarks/resident_comparison_20261016/full

Use a new output directory. --case selects configurations such as iris-large.
The pilot is separate and excluded from final statistics. Source commit/hashes,
hardware, package versions, thread limits, samples, residuals and complex roots
are retained. The machine is shared; repeated timings include background-load
variability. The earlier source/layout required refreshing the editable install;
NumPy, SciPy, CuPy and cuDSS versions were retained.

Larger mesh extension
---------------------

Custom cases retain the same settings and validation. For example, add
``--mesh-case perforated_waveguide-xl:perforated_waveguide:1.7`` and
``--mesh-case perforated_waveguide-xxl:perforated_waveguide:1.3`` to a run with
an unused output directory. These replace the standard case list. Actual
DOFs depend on meshing and are saved in results.json.

To plot an extension alongside the original sweep, use ``combine.py OUTPUT
INPUT1 INPUT2`` followed by ``report.py OUTPUT``. Combining checks source
consistency and successful completion, copies raw results, and retains each
sweep's original metadata in source_metadata_*.json. These are separate sweeps
on a shared host, not interleaved repetitions across mesh densities.
