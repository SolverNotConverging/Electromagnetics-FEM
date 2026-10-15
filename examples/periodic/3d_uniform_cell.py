"""Uniform 3D periodic cell compared with the analytic TE10 effective index."""

from pathlib import Path

import numpy as np
import fem

OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

dielectric = fem.Material(name="uniform dielectric", epsilon=2.25)
common = dict(frequency=10e9, x_range=.02, z_range=.005,
              background_material=dielectric)
solver = fem.PeriodicModeSolver3D(**common, y_range=0.01)
# A fixed 3D mesh must resolve the Gauss filter without adaptive retries.
solver.mesh(max_element_size=0.006, wavelength_elements=8)
modes = solver.solve(backend="auto", num_modes=1, max_refinements=0, neff_guess=1.3)
expected = np.sqrt(2.25 - (np.pi / (.02 * solver.k0))**2)
print("3D TE10 effective index:", modes[0].neff)
print("Analytic TE10 effective index:", expected)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
modes.save(OUTPUT_DIR / "results.h5")
modes.show()
