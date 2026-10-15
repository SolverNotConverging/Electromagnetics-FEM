"""Uniform 2D periodic cell with the analytic TEM effective index."""

from pathlib import Path

import fem

OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

dielectric = fem.Material(name="uniform dielectric", epsilon=2.25)
common = dict(frequency=10e9, x_range=.02, z_range=.005,
              background_material=dielectric)
solver = fem.PeriodicModeSolver2D(**common, polarization='TM')
solver.mesh(max_element_size=0.003, wavelength_elements=8)
modes = solver.solve(backend="auto", num_modes=1, max_refinements=0, neff_guess=1.5)
print("2D TEM effective index (exact 1.5):", modes[0].neff)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
modes.save(OUTPUT_DIR / "results.h5")
modes.show()
