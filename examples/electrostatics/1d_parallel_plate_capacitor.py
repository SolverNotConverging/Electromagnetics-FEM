"""Layered 1D FEM capacitor with geometry created before the mesh."""

from pathlib import Path

import fem

OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

dielectric = fem.Material(name="dielectric slab", epsilon=6.0)
solver = fem.ElectrostaticSolver(dim=1, outer_potential=None, x_range=(0.0, 0.01))
solver.add_geometry(name='dielectric', material=dielectric, shape=fem.shapes.Interval(bounds=(0.004, 0.007)))
solver.set_potential(potential=0.0, name='ground', geometry='left')
solver.set_potential(potential=10.0, name='drive', geometry='right')

# Potentials are in volts; max_element_size is in metres.
solver.mesh(max_element_size=0.0004)
result = solver.solve(backend="auto", max_refinements=0)
print(f"nodes={len(result.mesh_data.coordinates)}, energy={result.energy:.6e} J/m2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT_DIR / "results.h5")
result.show()
