"""10 GHz copper microstrip on a low-loss laminate.

The 35 um copper strip and ground are opaque surface-impedance conductors: their
dielectric-facing walls receive the copper surface impedance and no volume mesh
is generated inside the metal.
"""

from pathlib import Path

import fem

UM, MM = 1e-6, 1e-3
OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

laminate = fem.Material(name="microwave laminate", epsilon=3.55 * (1 - 0.0027j))  # tan(delta) = 0.0027
copper = fem.materials.copper

solver = fem.ModeSolver2D(frequency=10e9, x_range=(-6 * MM, 6 * MM), y_range=(-35 * UM, 6 * MM),
                          boundary=fem.materials.PEC, background_material=fem.materials.air)
solver.add_rectangle(x_range=(-6 * MM, 6 * MM), y_range=(0.0, 1.524 * MM), name="substrate", material=laminate)
solver.add_rectangle(x_range=(-6 * MM, 6 * MM), y_range=(-35 * UM, 0.0), name="copper_ground", material=copper)
solver.add_rectangle(x_range=(-1.5 * MM, 1.5 * MM), y_range=(1.524 * MM, 1.524 * MM + 35 * UM),
                     name="copper_strip", material=copper)
mesh = solver.mesh(max_element_size=0.6 * MM, wavelength_elements=10)
print(f"mesh: {mesh.info.nodes} nodes, {mesh.info.elements} triangles")

modes = solver.solve(backend="auto", num_modes=1, max_refinements=0)
mode = modes[0]
print(f"neff = {mode.neff:.6f}, loss = {mode.alpha * 8.686:.3f} dB/m, residual = {mode.residual:.1e}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
modes.save(OUTPUT_DIR / "results.h5")
modes.show()
