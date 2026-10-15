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
5. `3d_leaky_wave_antenna.py <3d_leaky_wave_antenna.py>`_ — A dielectric rod with one PEC patch per z period, surrounded by air and transverse PML. Full-vector single solve.

The 3D antenna example uses a 6 mm by 4 mm dielectric rod (relative
permittivity 6), a 10 mm period, and a 4 mm by 3 mm metal patch on the
rod's upper face at 20 GHz. ``build_solver()`` exposes the geometry for
parameter changes. The search guess ``0.5 - 0.02j`` targets a fast-wave
branch directly inside the air light cone (``|Re(neff)| < 1``).
It prints complex effective indices, attenuation, PML
energy fractions, and space harmonics inside the air light cone, then saves
``results.h5`` and opens the viewer. This is an exploratory unit-cell model;
check mesh, air-buffer, and PML convergence before interpreting leakage rates.

Run a script from this directory, or pass its path from the repository root.
Scripts that save results use
``fem_periodic_modes/outputs/<example>/`` in the downloaded repository.

Dispersion sweeps
-----------------

Each script below runs from top to bottom, uses ``tqdm`` to report completed
frequencies, saves each case and ``dispersion.csv`` in its own outputs folder,
and opens real/imaginary ``neff`` scatter plots when the sweep finishes.
Modes use the returned order at each frequency; samples are not connected.

* `2d_uniform_cell_dispersion.py <2d_uniform_cell_dispersion.py>`_
* `3d_uniform_cell_dispersion.py <3d_uniform_cell_dispersion.py>`_
* `2d_leaky_wave_antenna_dispersion.py <2d_leaky_wave_antenna_dispersion.py>`_
* `3d_iris_loaded_waveguide_filter_dispersion.py <3d_iris_loaded_waveguide_filter_dispersion.py>`_

Post-processing saved dispersion
--------------------------------

Run these scripts after a dispersion sweep to plot its saved ``dispersion.csv``
without solving again. Each reads the matching example output by default and
writes ``dispersion.png`` beside the CSV. Supply another CSV path to plot
any sweep from the same solver family. Both real and imaginary ``neff`` are
included, with mode labels starting at 1.

* `2d_plot_dispersion.py <post_processing/2d_plot_dispersion.py>`_
* `3d_plot_dispersion.py <post_processing/3d_plot_dispersion.py>`_

From the repository folder:

.. code-block:: sh

    python fem_periodic_modes/examples/post_processing/2d_plot_dispersion.py
    python fem_periodic_modes/examples/post_processing/2d_plot_dispersion.py path/to/dispersion.csv

Periodic dispersion uses scatter markers because modes can exchange places.

Backend selection
-----------------

Solver calls use ``backend="auto"``. FEM selects GPU when CuPy, the CUDA device,
and cuDSS are available, otherwise it uses SciPy CPU.
