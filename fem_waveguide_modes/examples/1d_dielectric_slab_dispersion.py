"""Guided-mode dispersion of a silicon slab in silica."""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np
from tqdm.auto import tqdm
from fem_waveguide_modes import Material, ModeSolver1D, plot_dispersion

OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / Path(__file__).stem
OUTPUT.mkdir(parents=True, exist_ok=True)

cladding = Material(name="silica cladding", epsilon=1.44 ** 2)
silicon = Material(name="silicon core", epsilon=3.45 ** 2)
frequencies = np.linspace(180e12, 210e12, 11)
neff_sweep = []
for case, frequency in enumerate(tqdm(frequencies, desc="Frequency sweep", unit="frequency"), start=1):
    solver = ModeSolver1D(frequency=frequency, x_range=(-3e-6, 3e-6),
                          background_material=cladding)
    solver.add_layer(x_range=(-250e-9, 250e-9), name="core", material=silicon)
    solver.mesh(max_element_size=60e-9)
    modes = solver.solve(neff_guess=3.6, max_refinements=0, num_modes=4)
    modes.save(OUTPUT / f"case_{case:03d}.h5")
    neff_sweep.append(modes.neff)

rows = [(frequency, mode, value.real, value.imag)
        for frequency, values in zip(frequencies, neff_sweep)
        for mode, value in enumerate(values, start=1)]
np.savetxt(OUTPUT / "dispersion.csv", rows, delimiter=",", comments="",
           header="frequency_hz,mode,neff_real,neff_imag",
           fmt=("%.12e", "%d", "%.12e", "%.12e"))
plot_dispersion(frequencies, neff_sweep)
