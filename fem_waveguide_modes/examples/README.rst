FEM waveguide modes examples
============================

Install the Python dependencies first; see `setup <../../README.md>`_.
The `user guide <../docs/guide.rst>`_ and
`public API <../docs/API_REFERENCE.rst>`_ explain supported controls.

A Matplotlib GUI backend is required to display interactive figures.
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
``fem_waveguide_modes/outputs/<example>/`` in the downloaded repository.

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
