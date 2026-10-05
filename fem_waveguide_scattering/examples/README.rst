FEM waveguide scattering examples
=================================

Install the Python dependencies first; see `setup <../../README.md>`_.
The `user guide <../docs/guide.rst>`_ and
`public API <../docs/API_REFERENCE.rst>`_ explain supported controls.

The ``fem-waveguide-scattering-viewer`` native application is required for ``show()``.
These examples use the ``mesh / solve / show`` workflow. Geometry and material
configuration precede meshing; the typed result is returned by ``solve()``.

Recommended order
-----------------

Runtime depends on hardware and mesh size. Single solves are the starting point;
dispersion and band-structure scripts perform many eigenproblems and can take
substantially longer. 3D cases also require more memory.

1. `2d_uniform_waveguide.py <2d_uniform_waveguide.py>`_ — Port modes and transmission through a uniform guide. Single solve.
2. `2d_dielectric_insert.py <2d_dielectric_insert.py>`_ — Reflection, transmission, and power balance for a weak insert. Single solve.
3. `2d_dielectric_insert_frequency_sweep.py <2d_dielectric_insert_frequency_sweep.py>`_ — A frequency sweep saved as a multi-case HDF5 archive. Frequency sweep.
4. `2d_slab_waveguide_oblique_incidence.py <2d_slab_waveguide_oblique_incidence.py>`_ — Oblique incidence with nonzero invariant-direction wavenumber. Single solve.
5. `2d_grounded_slab_slot.py <2d_grounded_slab_slot.py>`_ — A PEC slot in a grounded slab. Single solve.
6. `2d_closed_contour_farfield.py <2d_closed_contour_farfield.py>`_ — Matched modal ports and a closed four-sided NF2FF contour through the layered guide. Saves a complex radiation pattern, polar PNG, and reloadable HDF5 contour without opening a window.

Run a script from this directory, or pass its path from the repository root.
Scripts that save results use
``fem_waveguide_scattering/outputs/<example>/`` in the downloaded repository.
