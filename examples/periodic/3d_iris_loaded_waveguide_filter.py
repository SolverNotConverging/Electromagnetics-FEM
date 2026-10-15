"""Solve a periodic iris-loaded WR-90 rectangular-waveguide filter cell."""

from pathlib import Path

import fem

MM = 1.0e-3

OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

solver = fem.PeriodicModeSolver3D(frequency=12e9, x_range=(0.0, 22.86 * MM), y_range=(0.0, 10.16 * MM), z_range=(0.0, 8.0 * MM), boundary=fem.materials.PEC)
solver.add_geometry(name='left_iris', shape=fem.shapes.Box(bounds=((0.0, 4.0 * MM), (0.0, 10.16 * MM), (3.6 * MM, 4.4 * MM))), material=fem.materials.PEC)
solver.add_geometry(name='right_iris', shape=fem.shapes.Box(bounds=((18.86 * MM, 22.86 * MM), (0.0, 10.16 * MM), (3.6 * MM, 4.4 * MM))), material=fem.materials.PEC)
solver.mesh(max_element_size=4.0 * MM)

# The fixed-mesh Arnoldi solve can plateau around 2e-10; use a practical
# eigensolver tolerance while retaining the independent QEP/Gauss filters.
modes = solver.solve(backend="auto",
    direction='all',
    eigensolver='auto',
    max_refinements=0,
    eigensolver_tolerance=1e-08,
    num_modes=2,
    neff_guess=0.7,
)
print("neff:", modes.neff)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
modes.save(OUTPUT_DIR / "results.h5")
modes.show()
