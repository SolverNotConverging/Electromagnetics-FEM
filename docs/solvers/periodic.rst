FEM Periodic Mode Solver
========================

.. contents:: On this page
   :local:
   :depth: 2

First example
-------------

After ``pip install -e .`` (see `getting started <../getting_started.md>`_), run from
the repository root::

    python examples/periodic/2d_uniform_cell.py

The computed TEM effective index should be close to 1.5. The interactive viewer opens the completed result.

This example opens the interactive viewer when calling ``show()``.
For numerical runs without windows, omit ``show()`` or the plotting call;
FEM results also offer ``plot()`` for static figures.

Open the `first example <../../examples/periodic/2d_uniform_cell.py>`_ to change
geometry and controls. The `example index <../../examples/periodic/README.rst>`_
provides a learning order and more physical problems. Scripts run from any working directory once FEM is installed.

Scripts that save results write to ``examples/periodic/outputs/<example>/``.

Working with the solver
-----------------------

Compare lossless and lossy TEM propagation with the homogeneous-medium solution::

    python benchmarks/analytical/uniform_periodic_medium.py --check

The `benchmark guide <../../benchmarks/README.md>`_ explains the analytical
complex effective index, passive attenuation, and output tables and plots.

1. Construct the solver with keyword arguments and physical lengths in metres.
2. Add geometry, material properties, and boundary or excitation conditions.
3. Call ``mesh(...)`` to choose spatial and element resolution.
4. Call ``solve(...)`` to obtain a typed result; ``max_refinements=0`` uses one fixed mesh.
5. Inspect diagnostics, then call ``show()``, ``plot()``, or ``save(path)`` explicitly.


Workflow
--------

.. code-block:: python

    import fem

    dielectric = fem.Material(name="uniform dielectric", epsilon=2.25)
    solver = fem.PeriodicModeSolver2D(frequency=10e9, x_range=.02, z_range=.005,
                                      background_material=dielectric)
    solver.mesh(max_element_size=.003)
    result = solver.solve(num_modes=1, neff_guess=.66, max_refinements=0)
    result.save("periodic.h5")
    loaded = fem.load("periodic.h5")
    figure = loaded.plot(component="Ey", mode=0)
    loaded.show()

``solve()`` automatically meshes when needed and never saves or opens a window.
The adaptive defaults are two refinements and a relative tolerance of 0.05.
The example uses a fixed mesh for reproducibility. Geometry edits invalidate
``mesh_data`` and ``result``; an automatic rebuild reuses explicit mesh settings.

``show()`` opens the interactive PyVista/Qt viewer (``fem view result.h5`` opens saved
archives). Saving and loading work without a display.

Select ``Et`` or ``Ht`` for transverse (Ex, Ey) or (Hx, Hy) vector arrows,
excluding Ez/Hz. A 2D xz cell uses an xyz arrow view to retain the out-of-plane y
direction; 3D modes use PyVista arrows. ``real``/``imag`` show the selected phasor
part, while ``magnitude`` uses a global phase alignment and colours by transverse
magnitude. Mesh and material overlays remain available with the arrows.

Electromagnetic fields use ``exp(+i*omega*t)`` and guided propagation
``exp(-i*beta*z)``. Passive constitutive values have nonpositive imaginary
parts; forward attenuation is ``-Im(beta)``.

Persistence and API
-------------------

Results use the ``fem-results`` HDF5 envelope, schema ``1.0``. Old archives
are rejected. Loading supports inspection, plotting, and saving, not solver
restart or Python callback restoration. Mode indices are zero-based.

See the `API reference <../api/periodic.rst>`_ for supported configuration,
results, units, defaults, and exceptions. Run bundled examples from the repository
root with ``python examples/periodic/<example>.py``.

See the `family example index <../../examples/periodic/README.rst>`_
for learning order, output locations, and viewer requirements.

Frequency sweeps
----------------

The ``_dispersion.py`` examples solve each frequency with a ``tqdm`` progress
bar and display both real and imaginary effective indices after the sweep.
Plot them with ``fem.plot_dispersion``:

.. code-block:: python

    fem.plot_dispersion(frequencies, neff_sweep, connect=False)

Use frequencies in Hz and one row of complex effective indices per frequency.
The real and imaginary panels use unconnected scatter markers. Mode numbers
refer to the returned order at each frequency; modes can exchange places, so
points are not joined across frequencies.
