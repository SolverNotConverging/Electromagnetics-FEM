"""Sweep ordinary frequency and save complete results to one HDF5 file.

Every point is independently meshed, solved for its lead modes, launched,
and scattered.  The resulting HDF5 file can be opened with
``fem view <output-directory>/results.h5``.
"""

from pathlib import Path

import numpy as np

import fem

OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
frequencies = np.linspace(191.0e12, 195.0e12, 3)
insert = fem.Material(name="weak dielectric insert", epsilon=1.002)
solver = fem.ScatteringSolver2D(frequency=float(frequencies[0]), angle=0.0, x_range=(0.0, 1e-06), z_range=(-3e-06, 3e-06), background_material=fem.materials.vacuum, boundary=fem.materials.PEC)
solver.add_rectangle(x_range=(0.0, 1e-06), z_range=(-3e-07, 3e-07), name='weak_insert', material=insert)
solver.add_pml(order=3, target_reflection=1e-08, thickness=8e-07, direction='z')

sweep = solver.sweep(
    frequencies,
    backend="auto",
    mesh_options={'wavelength_elements': 8},
    mode_options={'backend': 'auto', 'max_refinements': 0, 'num_modes': 1, 'neff_guess': 1.0},
    max_refinements=0,
)
output_path = sweep.save(OUTPUT_DIR / 'results.h5')

print("frequency (Hz) =", sweep.frequencies)
print("S11 =", sweep.S11)
print("S21 =", sweep.S21)
print("R =", sweep.reflection)
print("T =", sweep.transmission)
print("power-balance error =", sweep.power_balance_error)
print("HDF5 sweep =", output_path)

sweep.show()
