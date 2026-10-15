"""End-to-end scattered-field solve for a weak dielectric insert.

This deliberately small PEC guide runs quickly and exercises the same mixed
Nedelec--Lagrange assembly, z-PML, mode projection, and power accounting used
by open transverse examples.
"""

from pathlib import Path

import fem

OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
insert = fem.Material(name="weak dielectric insert", epsilon=1.002)
solver = fem.ScatteringSolver2D(frequency=193414e9, angle=0.0, x_range=(0.0, 1e-06), z_range=(-3e-06, 3e-06), background_material=fem.materials.vacuum, boundary=fem.materials.PEC)
solver.add_rectangle(x_range=(0.0, 1e-06), z_range=(-3e-07, 3e-07), name='weak_insert', material=insert)
solver.add_pml(order=3, target_reflection=1e-08, thickness=8e-07, direction='z')
mesh = solver.mesh(wavelength_elements=8)
print("selected max element size =", mesh.info.requested_maximum_edge)

modes = solver.solve_modes(backend="auto", max_refinements=0, num_modes=1, neff_guess=1.0)
solver.set_incident_mode(modes[0])
result = solver.solve(backend="auto", max_refinements=0)
output_path = result.save(OUTPUT_DIR / 'results.h5')

print("beta =", modes[0].beta)
print("S11 =", result.S11)
print("S21 =", result.S21)
print("power-balance error =", result.power_balance_error)
print("HDF5 result =", output_path)
print("diagnostics =", result.check())
result.show()
