FEM periodic modes examples
===========================

Install the Python dependencies first; see `setup <../../README.md>`_.
The `user guide <../docs/guide.rst>`_ and
`public API <../docs/API_REFERENCE.rst>`_ explain supported controls.

The ``fem-periodic-mode-viewer`` native application is required for ``show()``.
These examples use the ``mesh / solve / show`` workflow. Geometry and material
configuration precede meshing; the typed result is returned by ``solve()``.

Recommended order
-----------------

Runtime depends on hardware and mesh size. Single solves are the starting point;
dispersion and band-structure scripts perform many eigenproblems and can take
substantially longer. 3D cases also require more memory.

1. `2d_uniform_cell.py <2d_uniform_cell.py>`_ — TEM effective index in a uniform 2D cell. Single solve.
2. `3d_uniform_cell.py <3d_uniform_cell.py>`_ — TE10 effective index in a uniform 3D cell. Single solve.
3. `2d_leaky_wave_antenna.py <2d_leaky_wave_antenna.py>`_ — A leaky-wave cell with an outgoing PML. Single solve.
4. `3d_iris_loaded_waveguide_filter.py <3d_iris_loaded_waveguide_filter.py>`_ — An iris-loaded rectangular waveguide cell. Single solve.

Run a script from this directory, or pass its path from the repository root.
Scripts that save results use
``fem_periodic_modes/outputs/<example>/`` in the downloaded repository.

Dispersion sweeps
-----------------

Each script below runs from top to bottom, uses ``tqdm`` to report completed
frequencies, saves each case and ``dispersion.csv`` in its own outputs folder,
and opens real/imaginary ``neff`` curves when the sweep finishes. Each trace
uses the mode order returned by the solver.

* `2d_uniform_cell_dispersion.py <2d_uniform_cell_dispersion.py>`_
* `3d_uniform_cell_dispersion.py <3d_uniform_cell_dispersion.py>`_
* `2d_leaky_wave_antenna_dispersion.py <2d_leaky_wave_antenna_dispersion.py>`_
* `3d_iris_loaded_waveguide_filter_dispersion.py <3d_iris_loaded_waveguide_filter_dispersion.py>`_
