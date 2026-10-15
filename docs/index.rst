fem
===

Finite-element electromagnetic solvers with one workflow for every problem type:

.. code-block:: python

   import fem

   fr4 = fem.Material(name="FR-4", epsilon=4.4 * (1 - 0.02j))  # lossy dielectric, tan(delta) = 0.02
   copper = fem.materials.copper                                # surface-impedance conductor

   solver = fem.ModeSolver2D(frequency=10e9, x_range=(-6e-3, 6e-3), y_range=(-35e-6, 6e-3),
                             boundary=fem.materials.PEC)
   solver.add_rectangle(x_range=(-6e-3, 6e-3), y_range=(0, 1.6e-3), name="substrate", material=fr4)
   solver.add_rectangle(x_range=(-6e-3, 6e-3), y_range=(-35e-6, 0), name="ground", material=copper)
   solver.add_rectangle(x_range=(-1.5e-3, 1.5e-3), y_range=(1.6e-3, 1.635e-3), name="strip", material=copper)
   solver.mesh(max_element_size=0.5e-3)
   modes = solver.solve(num_modes=1)

   print(f"neff = {modes[0].neff:.4f}, loss = {modes[0].alpha * 8.686:.1f} dB/m")
   modes.save("microstrip.h5")
   modes.show()  # fields, mesh and material regions

.. list-table:: Solvers
   :header-rows: 1
   :widths: 35 65

   * - Class
     - Problem
   * - ``fem.ElectrostaticSolver``
     - Potentials, fields, charges and energies in 1D or 2D
   * - ``fem.ModeSolver1D``, ``fem.ModeSolver2D``
     - Guided modes of a z-invariant slab or cross-section
   * - ``fem.PeriodicModeSolver2D``, ``fem.PeriodicModeSolver3D``
     - Complex Bloch modes of a structure periodic along z (leaky-wave antennas)
   * - ``fem.ScatteringSolver2D``
     - S-parameters, fields and radiation of 2D waveguide discontinuities

.. toctree::
   :maxdepth: 2
   :caption: Guide

   getting_started
   conventions
   materials_and_shapes
   adaptivity
   acceleration

.. toctree::
   :maxdepth: 2
   :caption: Solvers

   solvers/electrostatics
   solvers/waveguide
   solvers/periodic
   solvers/scattering

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api/common
   api/electrostatics
   api/waveguide
   api/periodic
   api/scattering
   api/adaptivity
   changelog
