"""Layered 1D capacitor compared with the analytic dielectric energy."""

from pathlib import Path

import fem

OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

dielectric = fem.Material(name="upper dielectric", epsilon=4.0)
solver = fem.ElectrostaticSolver(dim=1, outer_potential=None, x_range=(0.0, 1.0))
solver.add_layer(x_range=(0.5, 1.0), material=dielectric)
solver.set_potential(potential=0.0, name="ground", geometry="left")
solver.set_potential(potential=1.0, name="drive", geometry="right")
solver.mesh(max_element_size=0.1)
result = solver.solve(backend="auto", max_refinements=0)

print("Layered capacitor energy:", result.energy)
print("Expected energy:", 0.5 * fem.EPSILON_0 / (0.5 + 0.5 / 4.0))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT_DIR / "results.h5")
result.show()
