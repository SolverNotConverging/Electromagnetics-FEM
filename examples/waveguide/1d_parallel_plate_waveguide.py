"""Parallel-plate 1D modes compared with analytic cutoffs."""

from pathlib import Path

import fem

OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

frequency = 299_792_458.0  # vacuum wavelength = 1 metre
solver = fem.ModeSolver1D(frequency=frequency, x_range=1.0)
solver.mesh(resolution=48)
modes = solver.solve(backend="auto", max_refinements=0, num_modes=3)
print("1D effective indices (TEM and TE/TM cutoff pair):", modes.neff)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
modes.save(OUTPUT_DIR / "results.h5")
modes.show()
