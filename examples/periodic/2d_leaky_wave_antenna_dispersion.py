"""Complex Bloch-mode dispersion of the grounded-slab leaky-wave cell."""
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
import fem

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem
OUTPUT.mkdir(parents=True, exist_ok=True)

substrate = fem.Material(name="antenna substrate", epsilon=10.2)
frequencies = np.linspace(18e9, 22e9, 100)
neff_sweep = []
for case, frequency in enumerate(tqdm(frequencies, desc="Frequency sweep", unit="frequency"), start=1):
    solver = fem.PeriodicModeSolver2D(frequency=frequency, x_range=(0., 10e-3),
                                  z_range=(0., 8e-3), polarization="TM", boundary=fem.materials.PEC)
    solver.add_rectangle(x_range=(0., 1.27e-3), z_range=(0., 8e-3),
                         name="grounded_dielectric_slab", material=substrate)
    solver.add_geometry(name="top_pec_perturbation", material=fem.materials.PEC,
                        shape=fem.shapes.Rectangle(bounds=((1.27e-3, 1.32e-3), (1e-3, 2e-3))))
    solver.add_pml(thickness=2.5e-3, direction="x+")
    solver.mesh(max_element_size=350e-6)
    modes = solver.solve(backend="auto", max_refinements=0, direction="all", eigensolver="auto",
                         max_pml_fraction=None, num_modes=4, neff_guess=0.)
    modes.save(OUTPUT / f"case_{case:03d}.h5")
    neff_sweep.append(modes.neff)

rows = [(frequency, mode, value.real, value.imag)
        for frequency, values in zip(frequencies, neff_sweep)
        for mode, value in enumerate(values)]
np.savetxt(OUTPUT / "dispersion.csv", rows, delimiter=",", comments="",
           header="frequency,mode,neff_real,neff_imag",
           fmt=("%.12e", "%d", "%.12e", "%.12e"))
fem.plot_dispersion(frequencies, neff_sweep, connect=False)
