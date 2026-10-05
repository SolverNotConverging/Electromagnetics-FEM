"""TE10 dispersion in a uniform 3D periodic dielectric cell."""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np
from tqdm.auto import tqdm
from fem_periodic_modes import Material, PeriodicModeSolver3D, plot_dispersion

OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / Path(__file__).stem
OUTPUT.mkdir(parents=True, exist_ok=True)

dielectric = Material(name="uniform dielectric", epsilon=2.25)
frequencies = np.linspace(8e9, 12e9, 7)
neff_sweep = []
for case, frequency in enumerate(tqdm(frequencies, desc="Frequency sweep", unit="frequency"), start=1):
    solver = PeriodicModeSolver3D(frequency=frequency, x_range=20e-3, y_range=10e-3,
        z_range=5e-3, background_material=dielectric)
    solver.mesh(max_element_size=6e-3, wavelength_elements=8)
    # A 1e-8 tolerance resolves this fixed-mesh sweep without chasing roundoff.
    modes = solver.solve(num_modes=1, max_refinements=0, neff_guess=1.3,
        eigensolver_tolerance=1e-8)
    modes.save(OUTPUT / f"case_{case:03d}.h5")
    neff_sweep.append(modes.neff)

rows = [(frequency, mode, value.real, value.imag)
        for frequency, values in zip(frequencies, neff_sweep)
        for mode, value in enumerate(values, start=1)]
np.savetxt(OUTPUT / "dispersion.csv", rows, delimiter=",", comments="",
           header="frequency_hz,mode,neff_real,neff_imag",
           fmt=("%.12e", "%d", "%.12e", "%.12e"))
plot_dispersion(frequencies, neff_sweep)
