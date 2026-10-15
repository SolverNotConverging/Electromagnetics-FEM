"""TEM dispersion in a uniform 2D periodic dielectric cell."""
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
import fem

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem
OUTPUT.mkdir(parents=True, exist_ok=True)

dielectric = fem.Material(name="uniform dielectric", epsilon=2.25)
frequencies = np.linspace(8e9, 12e9, 11)
neff_sweep = []
for case, frequency in enumerate(tqdm(frequencies, desc="Frequency sweep", unit="frequency"), start=1):
    solver = fem.PeriodicModeSolver2D(frequency=frequency, x_range=20e-3, z_range=5e-3,
        background_material=dielectric, polarization="TM")
    solver.mesh(max_element_size=3e-3, wavelength_elements=8)
    modes = solver.solve(backend="auto", num_modes=1, max_refinements=0, neff_guess=1.5)
    modes.save(OUTPUT / f"case_{case:03d}.h5")
    neff_sweep.append(modes.neff)

rows = [(frequency, mode, value.real, value.imag)
        for frequency, values in zip(frequencies, neff_sweep)
        for mode, value in enumerate(values)]
np.savetxt(OUTPUT / "dispersion.csv", rows, delimiter=",", comments="",
           header="frequency,mode,neff_real,neff_imag",
           fmt=("%.12e", "%d", "%.12e", "%.12e"))
fem.plot_dispersion(frequencies, neff_sweep, connect=False)
