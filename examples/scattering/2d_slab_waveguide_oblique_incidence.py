"""Open transverse slab guide at a nonzero oblique propagation angle."""

from pathlib import Path

import numpy as np

import fem

OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
frequency = 193.414489e12
cladding = fem.Material(name="silica cladding", epsilon=1.44**2)
core = fem.Material(name="silicon core", epsilon=3.45**2)
perturbed_core = fem.Material(name="perturbed silicon", epsilon=3.451**2)
solver = fem.ScatteringSolver2D(frequency=frequency, angle=np.degrees(np.arcsin(0.1)), x_range=(-1.5e-06, 1.5e-06), z_range=(-2.5e-06, 2.5e-06), background_material=cladding)
solver.add_rectangle(x_range=(-2.2e-07, 2.2e-07), z_range=solver.z_range, background=True, name='core', material=core)
solver.add_rectangle(x_range=(-2.2e-07, 2.2e-07), z_range=(-2.5e-07, 2.5e-07), name='perturbation', material=perturbed_core)
solver.add_pml(thickness=3.5e-07, direction='x')
solver.add_pml(thickness=6.5e-07, direction='z')
solver.mesh(wavelength_elements=9)

modes = solver.solve_modes(backend="auto", max_refinements=0, num_modes=1, neff_guess=3.2, num_elements=54)
incident = solver.set_incident_mode(modes[0])
result = solver.solve(backend="auto", max_refinements=0, linear_solver_tolerance=1e-08)
output_path = result.save(OUTPUT_DIR / 'results.h5')

print("propagation angle (deg) =", solver.angle)
print("resolved ky =", solver.ky)
print("z-directed neff =", incident.mode.neff)
print("S11 =", result.S11)
print("S21 =", result.S21)
print("power-balance error =", result.power_balance_error)
print("HDF5 result =", output_path)
print("diagnostics =", result.check())
result.show()
