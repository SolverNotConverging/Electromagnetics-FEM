"""Second-order rectangular waveguide modes compared with the analytic TE10 cutoff."""

from pathlib import Path

import numpy as np
import fem

OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

frequency = 299_792_458.0
expected = np.sqrt(.75)
solver = fem.ModeSolver2D(frequency=frequency, x_range=1.0, y_range=0.5)
solver.mesh(resolution=(5, 3), element_order=2)
modes = solver.solve(backend="auto", max_refinements=0, num_modes=1)
print("2D TE10 effective index:", modes[0].neff)
print("Analytic TE10 effective index:", expected)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
modes.save(OUTPUT_DIR / "results.h5")
modes.show()
