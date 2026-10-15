FEM electrostatics examples
===========================

Install FEM first (``pip install -e .``; see `getting started <../../docs/getting_started.md>`_).
The `user guide <../../docs/solvers/electrostatics.rst>`_ and
`API reference <../../docs/api/electrostatics.rst>`_ explain supported controls.

These examples use the ``mesh / solve / show`` workflow. Geometry and material
configuration precede meshing; the typed result is returned by ``solve()``.

Recommended order
-----------------

Runtime depends on hardware and mesh size. Single solves are the starting point;
dispersion and band-structure scripts perform many eigenproblems and can take
substantially longer. 3D cases also require more memory.

1. `1d_parallel_plate_capacitor.py <1d_parallel_plate_capacitor.py>`_ — Electrodes and a dielectric inclusion in a 1D capacitor. Single solve.
2. `1d_layered_capacitor.py <1d_layered_capacitor.py>`_ — Analytic energy for two dielectric layers. Single solve.
3. `2d_parallel_plate_space_charge.py <2d_parallel_plate_space_charge.py>`_ — Uniform space charge and an analytic potential check. Single solve.
4. `2d_embedded_electrode_anisotropic.py <2d_embedded_electrode_anisotropic.py>`_ — An embedded electrode and anisotropic dielectric. Single solve.

Run a script from this directory, or pass its path from the repository root.
Scripts that save results use
``outputs/<example>/`` next to the scripts.

Backend selection
-----------------

Solver calls use ``backend="auto"``. FEM selects GPU when CuPy, the CUDA device,
and cuDSS are available, otherwise it uses SciPy CPU.
