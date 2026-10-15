FEM waveguide modes examples
============================

Install FEM first (``pip install -e .``; see `getting started <../../docs/getting_started.md>`_).
The `user guide <../../docs/solvers/waveguide.rst>`_ and
`API reference <../../docs/api/waveguide.rst>`_ explain supported controls.

These examples use the ``mesh / solve / show`` workflow. Geometry and material
configuration precede meshing; the typed result is returned by ``solve()``.

Recommended order
-----------------

Runtime depends on hardware and mesh size. Single solves are the starting point;
dispersion and band-structure scripts perform many eigenproblems and can take
substantially longer. 3D cases also require more memory.

1. `1d_parallel_plate_waveguide.py <1d_parallel_plate_waveguide.py>`_ — 1D modes and analytic cutoff checks. Single solve.
2. `2d_rectangular_waveguide.py <2d_rectangular_waveguide.py>`_ — Second-order vector elements and the TE10 cutoff. Single solve.
3. `1d_dielectric_slab.py <1d_dielectric_slab.py>`_ — Guided modes in a dielectric slab. Single solve.
4. `2d_ridge_waveguide.py <2d_ridge_waveguide.py>`_ — A ridge waveguide cross section. Single solve.
5. `2d_microstrip_surface_impedance.py <2d_microstrip_surface_impedance.py>`_ — A copper microstrip with a surface-impedance boundary. Single solve.

Run a script from this directory, or pass its path from the repository root.
Scripts that save results use
``outputs/<example>/`` next to the scripts.

Dispersion sweeps
-----------------

Each script below runs from top to bottom, uses ``tqdm`` to report completed
frequencies, saves each case and ``dispersion.csv`` in its own outputs folder,
and opens real/imaginary ``neff`` curves when the sweep finishes. Each trace
uses the mode order returned by the solver.

* `1d_parallel_plate_waveguide_dispersion.py <1d_parallel_plate_waveguide_dispersion.py>`_
* `1d_dielectric_slab_dispersion.py <1d_dielectric_slab_dispersion.py>`_
* `2d_rectangular_waveguide_dispersion.py <2d_rectangular_waveguide_dispersion.py>`_
* `2d_ridge_waveguide_dispersion.py <2d_ridge_waveguide_dispersion.py>`_
* `2d_microstrip_surface_impedance_dispersion.py <2d_microstrip_surface_impedance_dispersion.py>`_

Post-processing saved dispersion
--------------------------------

Run these scripts after a dispersion sweep to plot its saved ``dispersion.csv``
without solving again. Each reads the matching example output by default and
writes ``dispersion.png`` beside the CSV. Supply another CSV path to plot
any sweep from the same solver family. Both real and imaginary ``neff`` are
included; mode labels are zero-based.

* `1d_plot_dispersion.py <post_processing/1d_plot_dispersion.py>`_
* `2d_plot_dispersion.py <post_processing/2d_plot_dispersion.py>`_

From the repository folder:

.. code-block:: sh

    python examples/waveguide/post_processing/1d_plot_dispersion.py
    python examples/waveguide/post_processing/1d_plot_dispersion.py path/to/dispersion.csv

Backend selection
-----------------

Solver calls use ``backend="auto"``. FEM selects GPU when CuPy, the CUDA device,
and cuDSS are available, otherwise it uses SciPy CPU.
