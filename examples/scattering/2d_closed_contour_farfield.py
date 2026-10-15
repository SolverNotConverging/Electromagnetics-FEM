"""Radiation pattern of the grounded-slab slot from a closed NF2FF contour.

Same device as 2d_grounded_slab_slot.py, terminated with matched modal ports
instead of z PML. A closed rectangle around the slot records the scattered
fields, from which the far field is computed. The far-field tab of the viewer
shows the pattern; result.far_field(theta) returns it as data.
"""

from pathlib import Path

import numpy as np

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
solver.set_matched_ports()
solver.set_monitors(left=-20 * MM, right=20 * MM)
solver.set_nf2ff_contour(x_range=(-12 * MM, 12 * MM), z_range=(-15 * MM, 15 * MM))
solver.mesh(max_element_size=1 * MM, element_order=2, wavelength_elements=10)

# Resolve the thin substrate in the separate one-dimensional lead mesh.
solver.solve_modes(backend="auto", num_modes=1, neff_guess=1.8, num_elements=384, max_refinements=0)
solver.set_incident_mode(0)
result = solver.solve(backend="auto", max_refinements=0)

# Midpoint angles avoid exact grazing and a repeated 2*pi endpoint.
theta = (np.arange(360) + 0.5) * 2 * np.pi / 360
far = result.far_field(theta)
print("closed-contour radiation (W/m):", far.integrated_power())
print("independent flux/modal estimate (W/m):", result.radiated_power)
print("peak directivity (dB):", 10 * np.log10(np.nanmax(far.directivity)))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
result.save(OUTPUT_DIR / "results.h5")
result.show()
