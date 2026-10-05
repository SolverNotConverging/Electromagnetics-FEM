"""Complex dispersion of a copper microstrip with surface impedance."""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np
from tqdm.auto import tqdm
from fem_waveguide_modes import Material, materials, shapes, ModeSolver2D, plot_dispersion

OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / Path(__file__).stem
OUTPUT.mkdir(parents=True, exist_ok=True)

substrate = Material(name="microwave laminate", epsilon=3.55 * (1. - 1j * 0.0027))
frequencies = np.linspace(8e9, 12e9, 9)
neff_sweep = []
for case, frequency in enumerate(tqdm(frequencies, desc="Frequency sweep", unit="frequency"), start=1):
    solver = ModeSolver2D(frequency=frequency, x_range=(-6e-3, 6e-3),
                          y_range=(-35e-6, 6e-3), boundary=materials.PEC, background_material=materials.air)
    solver.add_rectangle(x_range=(-6e-3, 6e-3), y_range=(0., 1.524e-3),
                         name="substrate", material=substrate)
    solver.add_geometry(name="copper_ground", material=materials.copper,
                        shape=shapes.Rectangle(bounds=((-6e-3, 6e-3), (-35e-6, 0.))))
    solver.add_geometry(name="copper_strip", material=materials.copper,
                        shape=shapes.Rectangle(bounds=((-1.5e-3, 1.5e-3), (1.524e-3, 1.559e-3))))
    solver.mesh(max_element_size=600e-6, wavelength_elements=10, material_aware=True)
    modes = solver.solve(max_refinements=0, num_modes=1)
    modes.save(OUTPUT / f"case_{case:03d}.h5")
    neff_sweep.append(modes.neff)

rows = [(frequency, mode, value.real, value.imag)
        for frequency, values in zip(frequencies, neff_sweep)
        for mode, value in enumerate(values, start=1)]
np.savetxt(OUTPUT / "dispersion.csv", rows, delimiter=",", comments="",
           header="frequency_hz,mode,neff_real,neff_imag",
           fmt=("%.12e", "%d", "%.12e", "%.12e"))
plot_dispersion(frequencies, neff_sweep)
