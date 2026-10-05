"""Complex Bloch-mode dispersion of the grounded-slab leaky-wave cell."""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import numpy as np
from tqdm.auto import tqdm
from fem_periodic_modes import Material, materials, shapes, PeriodicModeSolver2D, plot_dispersion

OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / Path(__file__).stem
OUTPUT.mkdir(parents=True, exist_ok=True)

substrate = Material(name="antenna substrate", epsilon=10.2)
frequencies = np.linspace(18e9, 22e9, 9)
neff_sweep = []
for case, frequency in enumerate(tqdm(frequencies, desc="Frequency sweep", unit="frequency"), start=1):
    solver = PeriodicModeSolver2D(frequency=frequency, x_range=(0., 10e-3),
        z_range=(0., 8e-3), polarization="TM", boundary=materials.PEC)
    solver.add_rectangle(x_range=(0., 1.27e-3), z_range=(0., 8e-3),
        name="grounded_dielectric_slab", material=substrate)
    solver.add_geometry(name="top_pec_perturbation", material=materials.PEC,
        shape=shapes.Rectangle(bounds=((1.27e-3, 1.32e-3), (1e-3, 2e-3))))
    solver.add_pml(thickness=2.5e-3, direction="x+")
    solver.mesh(max_element_size=350e-6)
    modes = solver.solve(max_refinements=0, direction="all", eigensolver="auto",
        max_pml_fraction=None, num_modes=4, neff_guess=0.)
    modes.save(OUTPUT / f"case_{case:03d}.h5")
    neff_sweep.append(modes.neff)

rows = [(frequency, mode, value.real, value.imag)
        for frequency, values in zip(frequencies, neff_sweep)
        for mode, value in enumerate(values, start=1)]
np.savetxt(OUTPUT / "dispersion.csv", rows, delimiter=",", comments="",
           header="frequency_hz,mode,neff_real,neff_imag",
           fmt=("%.12e", "%d", "%.12e", "%.12e"))
plot_dispersion(frequencies, neff_sweep)
