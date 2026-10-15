"""Complex Bloch-mode dispersion of a periodic iris-loaded WR-90 cell."""
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
import fem

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem
OUTPUT.mkdir(parents=True, exist_ok=True)

frequencies = np.linspace(11e9, 13e9, 5)
neff_sweep = []
for case, frequency in enumerate(tqdm(frequencies, desc="Frequency sweep", unit="frequency"), start=1):
    solver = fem.PeriodicModeSolver3D(frequency=frequency, x_range=(0., 22.86e-3),
        y_range=(0., 10.16e-3), z_range=(0., 8e-3), boundary=fem.materials.PEC)
    solver.add_geometry(name="left_iris", material=fem.materials.PEC,
        shape=fem.shapes.Box(bounds=((0., 4e-3), (0., 10.16e-3), (3.6e-3, 4.4e-3))))
    solver.add_geometry(name="right_iris", material=fem.materials.PEC,
        shape=fem.shapes.Box(bounds=((18.86e-3, 22.86e-3), (0., 10.16e-3), (3.6e-3, 4.4e-3))))
    solver.mesh(max_element_size=4e-3)
    modes = solver.solve(backend="auto", direction="all", eigensolver="auto", max_refinements=0,
        eigensolver_tolerance=1e-8, num_modes=2, neff_guess=0.7)
    modes.save(OUTPUT / f"case_{case:03d}.h5")
    neff_sweep.append(modes.neff)

rows = [(frequency, mode, value.real, value.imag)
        for frequency, values in zip(frequencies, neff_sweep)
        for mode, value in enumerate(values)]
np.savetxt(OUTPUT / "dispersion.csv", rows, delimiter=",", comments="",
           header="frequency,mode,neff_real,neff_imag",
           fmt=("%.12e", "%d", "%.12e", "%.12e"))
fem.plot_dispersion(frequencies, neff_sweep, connect=False)
