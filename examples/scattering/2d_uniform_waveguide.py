"""Fixed-mesh full-vector lead modes and second-order uniform fem.scattering."""

from pathlib import Path

import fem

OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

solver = fem.ScatteringSolver2D(frequency=0.299792e9 / 1.0, x_range=(0.0, 0.5), z_range=(-2.0, 2.0), boundary=fem.materials.PEC)
solver.add_pml(thickness=0.5, direction='z')
solver.mesh(max_element_size=0.15, element_order=2)
# First solve the lead modes, then choose the zero-based incident mode.
modes = solver.solve_modes(backend="auto", num_modes=1, neff_guess=1., num_elements=16, max_refinements=0)
solver.set_incident_mode(0)
# The scattering solve returns power-normalized reflection and transmission.
result = solver.solve(backend="auto", max_refinements=0)
print("TEM effective index (exact 1):", modes[0].neff)
print("Reflection (exact 0):", result.reflection)
print("Transmission (exact 1):", result.transmission)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT_DIR / "results.h5")
result.show()
