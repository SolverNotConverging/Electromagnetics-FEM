"""Leaky-wave antenna formed by a finite slot in a grounded dielectric slab.

The lead contains a zero-thickness PEC ground plane invariant in z. The device
removes a 2 mm piece of that plane, so the aperture is a boundary perturbation
rather than a permittivity perturbation.
"""

from pathlib import Path

import fem

MM = 1e-3
OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

solver = fem.ScatteringSolver2D(frequency=20e9, x_range=(-20 * MM, 20 * MM), z_range=(-30 * MM, 30 * MM),
                                background_material=fem.materials.air)
solver.add_rectangle(x_range=(0.0, 1.27 * MM), z_range=solver.z_range, background=True,
                     name="dielectric_slab", material=fem.Material(name="grounded slab", epsilon=10.2))
ground = solver.add_geometry(background=True, name="ground_plane", material=fem.materials.PEC,
                             shape=fem.shapes.Segment(start=(0.0, -30 * MM), end=(0.0, 30 * MM)))
solver.add_slot(name="ground_slot", geometry=ground, z_range=(-1 * MM, 1 * MM))
solver.add_pml(target_reflection=1e-8, thickness=4 * MM, direction="x")
solver.add_pml(target_reflection=1e-8, thickness=6 * MM, direction="z")
solver.set_monitors(left=-20 * MM, right=20 * MM)
solver.mesh(max_element_size=2 * MM, wavelength_elements=10)

modes = solver.solve_modes(backend="auto", num_modes=1, neff_guess=1.8, num_elements=96, max_refinements=0)
solver.set_incident_mode(modes[0])
result = solver.solve(backend="auto", max_refinements=0)

print("surface-mode effective index =", modes[0].neff)
print("S11 =", result.S11, " S21 =", result.S21)
print("R, T =", result.reflection, result.transmission)
print("radiated, absorbed power (W/m) =", result.radiated_power, result.absorbed_power)
print("power-balance error =", result.power_balance_error)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT_DIR / "results.h5")
result.show()
