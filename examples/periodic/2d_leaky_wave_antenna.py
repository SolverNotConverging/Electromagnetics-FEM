"""Solve the grounded-slab leaky-wave cell used by the FDFD example.

The lower face is the PEC ground, a high-permittivity slab fills the full
period, and a short PEC patch perturbs only the top of that slab.  Radiation
leaks into the air/PML region above the periodically repeated patches.
"""

from pathlib import Path

import fem

MM = 1.0e-3

OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

substrate = fem.Material(name="antenna substrate", epsilon=10.2)
solver = fem.PeriodicModeSolver2D(frequency=20e9, x_range=(0.0, 10.0 * MM), z_range=(0.0, 8.0 * MM), polarization='TM', boundary=fem.materials.PEC)
solver.add_rectangle(x_range=(0.0, 1.27 * MM), z_range=(0.0, 8.0 * MM), name='grounded_dielectric_slab', material=substrate)
solver.add_geometry(name='top_pec_perturbation', shape=fem.shapes.Rectangle(bounds=((1.27 * MM, 1.32 * MM), (1.0 * MM, 2.0 * MM))), material=fem.materials.PEC)
solver.add_pml(thickness=2.5 * MM, direction='x+')
solver.mesh(max_element_size=0.35 * MM)

modes = solver.solve(backend="auto",
    max_refinements=0,
    direction='all',
    eigensolver='auto',
    max_pml_fraction=None,
    num_modes=4,
    neff_guess=0,
)
print("neff:", modes.neff)
print("PML energy fractions:", [mode.pml_fraction for mode in modes])

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
modes.save(OUTPUT_DIR / "results.h5")
modes.show()
