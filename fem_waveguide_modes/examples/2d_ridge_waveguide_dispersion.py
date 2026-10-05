"""Full-vector guided-mode dispersion of a silicon ridge on silica."""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np
from tqdm.auto import tqdm
from fem_waveguide_modes import Material, materials, ModeSolver2D, plot_dispersion

OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / Path(__file__).stem
OUTPUT.mkdir(parents=True, exist_ok=True)

silicon = Material(name="silicon core", epsilon=3.45**2)
silica = Material(name="silica slab", epsilon=1.44**2)
frequencies = np.linspace(900e12, 1100e12, 7)
neff_sweep = []
for case, frequency in enumerate(tqdm(frequencies, desc="Frequency sweep", unit="frequency"), start=1):
    solver = ModeSolver2D(frequency=frequency, x_range=(-2e-6, 2e-6),
        y_range=(-1.5e-6, 1.5e-6), boundary=materials.PEC, background_material=materials.vacuum)
    solver.add_rectangle(x_range=(-100e-9, 100e-9), y_range=(-110e-9, 110e-9),
        name="silicon_core", material=silicon)
    solver.add_rectangle(x_range=(-1e-6, 1e-6), y_range=(-200e-9, -110e-9),
        name="slab", material=silica)
    solver.mesh(max_element_size=100e-9, quadrature_order=4)
    modes = solver.solve(max_refinements=0, num_modes=4, neff_guess=3.6)
    modes.save(OUTPUT / f"case_{case:03d}.h5")
    neff_sweep.append(modes.neff)

rows = [(frequency, mode, value.real, value.imag)
        for frequency, values in zip(frequencies, neff_sweep)
        for mode, value in enumerate(values, start=1)]
np.savetxt(OUTPUT / "dispersion.csv", rows, delimiter=",", comments="",
           header="frequency_hz,mode,neff_real,neff_imag",
           fmt=("%.12e", "%d", "%.12e", "%.12e"))
plot_dispersion(frequencies, neff_sweep)
