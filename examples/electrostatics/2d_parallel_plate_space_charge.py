"""Parallel plates with uniform space charge and an analytic peak potential."""

from pathlib import Path

import fem

OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

solver = fem.ElectrostaticSolver(dim=2, outer_potential=None, x_range=1., y_range=.5)
solver.set_potential(potential=0.0, geometry='left')
solver.set_potential(potential=0.0, geometry='right')
solver.add_charge_density(density=fem.EPSILON_0, geometry=fem.shapes.Rectangle(bounds=((0.0, 1.0), (0.0, 0.5))))
solver.mesh(max_element_size=.12)
result = solver.solve(backend="auto", max_refinements=0)
print("Charged guide peak potential (exact 0.125 V):", result.potential.max())
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT_DIR / "results.h5")
result.show()
