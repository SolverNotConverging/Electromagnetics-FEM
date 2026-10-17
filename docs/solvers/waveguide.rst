FEM Waveguide Mode Solver
=========================

.. contents:: On this page
   :local:
   :depth: 2

First example
-------------

After ``pip install -e .`` (see `getting started <../getting_started.md>`_), run from
the repository root::

    python examples/waveguide/2d_rectangular_waveguide.py

The script prints the computed and analytic TE10 effective indices and opens the interactive viewer.

A Matplotlib GUI backend is needed to display the figures.
For numerical runs without windows, omit ``show()`` or the plotting call;
FEM results also offer ``plot()`` for static figures.

Open the `first example <../../examples/waveguide/2d_rectangular_waveguide.py>`_ to change
geometry and controls. The `example index <../../examples/waveguide/README.rst>`_
provides a learning order and more physical problems. Scripts run from any working directory once FEM is installed.

Scripts that save results write to ``examples/waveguide/outputs/<example>/``.

Working with the solver
-----------------------

For a quantitative comparison with theory, run::

    python benchmarks/analytical/rectangular_waveguide_modes.py --check

This compares FEM TE10 effective indices with the rectangular PEC
waveguide solution at several resolutions. The `benchmark guide <../../benchmarks/README.md>`_
explains the geometry, error measures, and generated CSV/plot reports.

The `coaxial TEM benchmark <../../benchmarks/analytical/coaxial_waveguide_adaptivity.py>`_
compares adaptive fields against the analytical radial solution::

    python benchmarks/analytical/coaxial_waveguide_adaptivity.py --check

Assign ``materials.PEC`` to a ``shapes.Circle`` for a circular conductor.
``shapes.Annulus`` models the outer conductor and ``clip=True`` explicitly
intersects out-of-bounds geometry with the model domain.
The benchmark demonstrates both conductors and the mesh/solve workflow.
Its plots distinguish physical field errors, eigenproblem residuals, and the
adaptive estimator. The current 2D waveguide backend globally remeshes by a
factor of 1.5 when its estimator requests refinement; it does not mark local
elements. The benchmark records budget exhaustion separately from convergence.

1. Construct the solver with keyword arguments and physical lengths in metres.
2. Add geometry, material properties, and boundary or excitation conditions.
3. Call ``mesh(...)`` to choose spatial and element resolution.
4. Call ``solve(...)`` to obtain a typed result; ``max_refinements=0`` uses one fixed mesh.
5. Inspect diagnostics, then call ``show()``, ``plot()``, or ``save(path)`` explicitly.


Workflow
--------

.. code-block:: python

    import fem

    solver = fem.ModeSolver2D(frequency=10e9, x_range=22.86e-3,
                              y_range=10.16e-3, boundary=fem.materials.PEC)
    solver.mesh(max_element_size=1e-3)
    result = solver.solve(num_modes=4, max_refinements=0)
    result.save("modes.h5")
    loaded = fem.load("modes.h5")
    figure = loaded.plot(component="Ey", mode=0)
    loaded.show()

``solve()`` automatically meshes when needed and never saves or opens a window.
Omit ``neff_guess`` to let the solver choose its search target from the
materials at each frequency. The examples use this automatic default.
The adaptive defaults are two refinements and a relative tolerance of 0.05.
The example uses a fixed mesh for reproducibility. Geometry edits invalidate
``mesh_data`` and ``result``; an automatic rebuild reuses explicit mesh settings.

``show()`` opens the interactive viewer. ``plot()`` returns
a Figure without opening a GUI. Both dimensions support scalar and diagonal
relative materials; the 2D solver also supports surface-impedance conductors.

Select ``Et`` or ``Ht`` in the viewer for transverse electric (Ex, Ey) or magnetic
(Hx, Hy) arrows. ``real``/``imag`` show that phasor part; ``magnitude`` draws a
globally phase-aligned real field, coloured by the complex transverse magnitude.
Arrow lengths share a scale within the view. Mesh/material overlays retain the
arrows. In 1D, the samples lie along y=0 and the arrows show x/y components.

Symmetry and mode identity in 2D
------------------------------------

Run the `square-guide example <../../examples/waveguide/2d_square_waveguide.py>`_
for 24 square-guide modes, or the
`cylindrical-guide example <../../examples/waveguide/2d_cylindrical_waveguide.py>`_
for 24 circular-guide modes. Both print mode summaries and open the viewer
to inspect every mode and field component; the
`example index <../../examples/waveguide/README.rst>`_ explains their controls.

Enable identification when comparing degenerate modes across meshes::

    options = dict(num_modes=2, track_modes=True, max_refinements=0)
    first = solver.solve(**options)
    solver.refine(1.5)
    second = solver.solve(**options)
    for mode in second:
        print(mode.metadata["mode_id"], mode.metadata["symmetry_irrep"],
              mode.metadata["symmetry_partner"], mode.metadata["tracking_overlap"])

These controls are opt-in. Existing solves retain their nearest-target ordering.
``track_modes`` is the on/off switch. ``track_options`` is an optional mapping;
omit it, pass ``None`` or ``{}``, or supply only the settings you want to change::

    modes = solver.solve(track_modes=True)
    modes = solver.solve(track_modes=True, track_options={
        "symmetry": "C4v",
        "degeneracy_tolerance": 1e-4,
        "overlap_threshold": 0.8,
    })
    modes = solver.solve(track_modes=False)

Supported ``track_options`` keys and defaults:

* ``symmetry="auto"``: detect a supported continuous-scene subgroup. Use
  ``None`` for overlap tracking without symmetry, or declare a group explicitly.
* ``degeneracy_tolerance=1e-5``: positive relative effective-index cluster diameter.
* ``overlap_threshold=0.5``: minimum squared individual or principal-angle
  overlap in (0, 1] for retaining identities and aligning subspaces.
* ``symmetry_tolerance=0.25``: positive maximum sampled symmetry-subspace residual
  for assigning a resolved irrep and mirror partner.

Unknown keys, invalid values, and non-mapping configurations raise a
configuration error. The caller's mapping is not modified. Supplying options
alone does not enable tracking: ``track_modes=False`` disables both symmetry
identification and tracking, while still validating supplied options. Effective
settings are recorded in the tracked result's ``metadata["track_options"]``.

Tracking retains identities and phases within one solver across explicit
remeshing and automatic adaptive refinement. Geometry changes clear the reference;
``track_modes=False`` also clears it. Tracking is local to the solver, and loading
a saved result does not restore a live tracking session. Mode IDs are stored in
mode metadata, while ``mode.index`` remains the zero-based position in this result.

The ``symmetry`` tracking option tests Cnv subgroups with n from 1 through 8, about the centre
of the rectangular domain, with x- or y-normal mirror planes. An explicit
An explicit ``"symmetry": "C4v"`` (or another ``C1v`` through ``C8v``) uses an x-normal mirror
and verifies the declared group. Tests use deterministic probes of the continuous
materials, rotated constitutive tensors, conductor regions, impedance values,
outer boundaries, and PML. Results are cached by geometry revision. This is a
sampled check, not an exact CAD proof; arbitrarily oriented or off-centre groups
are not inferred, and thin or intricate features can require independent checking.
``symmetry_group`` records the subgroup found, rather than claiming the complete
point group; a circular guide can support more symmetry than the finite subgroup.
Local mesh sizing does not define physical symmetry. Unsupported declarations
raise a configuration error rather than imposing new physical boundary conditions.

The hierarchy is symmetry sector, longitudinal field character, then spatial
overlap. A multidimensional ``E1`` irrep remains one irrep, with ``+1`` and ``-1``
mirror partners used to choose its basis. One-dimensional sectors use
``m<angular order>:<mirror sign>`` labels. Already separated TE/TM spaces are
preserved. ``longitudinal_electric_fraction`` and
``longitudinal_magnetic_fraction`` record physical-domain, quadrature-weighted
fractions of the squared electric and magnetic field norms. Hybrid modes keep
these continuous descriptors; no universal EH/HE naming convention is assumed.
Repeated copies of a symmetry sector are aligned to the previous subspace before
matching individual mode IDs by normalized electric-field overlap.
For an initial degenerate basis, longitudinal content is followed by a radial
second-moment observable to distinguish spatial variation within unresolved
multiplicities. These observables choose a basis; they are not extra symmetry
labels or a universal radial mode numbering scheme.
``subspace_overlap`` records the smallest squared cosine of the principal angles
to the best previous cluster of the same dimension. It distinguishes a converging
modal subspace from unstable individual field patterns.

Candidate clusters have a bounded effective-index diameter, using
``degeneracy_tolerance * max(1, abs(neff))``. The solver retains extra candidates
around the requested boundary to resolve a multiplet before truncating the output.
``truncated_degenerate_cluster`` reports when the requested count returns only
part of a recovered cluster; request the whole multiplet for subspace tracking.
This look-ahead cannot guarantee recovery of every eigenvector from an incomplete
eigensolver candidate set. Identification uses a common 16-by-16 physical probe
grid, so fine spatial structure or small disconnected domains can require higher
spatial resolution or a dedicated modal comparison. Fewer than eight common
interior probes, or a rank-deficient sampled subspace, raise a configuration error.

Basis rotations must pass the original ``residual_tolerance`` and
``divergence_tolerance``. Mesh-split eigenvalues are not forcibly averaged.
``symmetry_projection="residual_rejected"`` means returned fields retain valid
eigenvectors, with approximate identification from the projected basis; they may
still mix analytical partners. ``symmetry_overlap`` measures their correspondence
to that basis. ``symmetry_subspace_residual`` measures how well the candidate
basis closes under the symmetry generators on the common probes. Values above
``symmetry_tolerance`` leave ``symmetry_resolved=False`` and the irrep/partner
unset. A lost overlap match (squared overlap below ``overlap_threshold``) receives
a new ID and ``tracking_overlap=None``.
Inspect these diagnostics rather than assuming that a label proves convergence.
The original pencil residual also remains available as ``mode.residual``.

Electromagnetic fields use ``exp(+i*omega*t)`` and guided propagation
``exp(-i*beta*z)``. Passive constitutive values have nonpositive imaginary
parts; forward attenuation is ``-Im(beta)``.

Persistence and API
-------------------

Results use the ``fem-results`` HDF5 envelope, schema ``1.0``. Old archives
are rejected. Loading supports inspection, plotting, and saving, not solver
restart or Python callback restoration. Mode indices are zero-based.

See the `API reference <../api/waveguide.rst>`_ for supported configuration,
results, units, defaults, and exceptions. Run bundled examples from the repository
root with ``python examples/waveguide/<example>.py``.

See the `family example index <../../examples/waveguide/README.rst>`_
for learning order, output locations, and viewer requirements.

Frequency sweeps
----------------

The ``_dispersion.py`` examples solve each frequency with a ``tqdm`` progress
bar and display both real and imaginary effective indices after the sweep.
Plot them with ``fem.plot_dispersion``:

.. code-block:: python

    fem.plot_dispersion(frequencies, neff_sweep)

Use frequencies in Hz and one row of complex effective indices per frequency.
Each column is one plotted mode, using the supplied ordering.

CUDA acceleration
-----------------

``backend="auto"`` (the default) selects GPU when CuPy, an NVIDIA device and
cuDSS are available, otherwise SciPy CPU; ``backend="gpu"`` requires the GPU.
Assembly, meshing and adaptive refinement stay on the CPU. See
`acceleration <../acceleration.md>`_ for installation and device checks.
