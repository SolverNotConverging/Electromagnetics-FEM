"""Complex dispersion of a copper microstrip with surface impedance."""
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
import fem

OUTPUT = Path(__file__).parent / "outputs" / Path(__file__).stem
OUTPUT.mkdir(parents=True, exist_ok=True)

substrate = fem.Material(name="microwave laminate", epsilon=3.55 * (1. - 1j * 0.0027))
frequencies = np.linspace(8e9, 12e9, 9)
neff_sweep = []
for case, frequency in enumerate(tqdm(frequencies, desc="Frequency sweep", unit="frequency"), start=1):
    solver = fem.ModeSolver2D(frequency=frequency, x_range=(-6e-3, 6e-3),
                          y_range=(-35e-6, 6e-3), boundary=fem.materials.PEC, background_material=fem.materials.air)
    solver.add_rectangle(x_range=(-6e-3, 6e-3), y_range=(0., 1.524e-3),
                         name="substrate", material=substrate)
    solver.add_geometry(name="copper_ground", material=fem.materials.copper,
                        shape=fem.shapes.Rectangle(bounds=((-6e-3, 6e-3), (-35e-6, 0.))))
    solver.add_geometry(name="copper_strip", material=fem.materials.copper,
                        shape=fem.shapes.Rectangle(bounds=((-1.5e-3, 1.5e-3), (1.524e-3, 1.559e-3))))
    solver.mesh(max_element_size=600e-6, wavelength_elements=10, material_aware=True)
    modes = solver.solve(backend="auto", max_refinements=0, num_modes=1)
    modes.save(OUTPUT / f"case_{case:03d}.h5")
    neff_sweep.append(modes.neff)

rows = [(frequency, mode, value.real, value.imag)
        for frequency, values in zip(frequencies, neff_sweep)
        for mode, value in enumerate(values)]
np.savetxt(OUTPUT / "dispersion.csv", rows, delimiter=",", comments="",
           header="frequency,mode,neff_real,neff_imag",
           fmt=("%.12e", "%d", "%.12e", "%.12e"))
fem.plot_dispersion(frequencies, neff_sweep)
