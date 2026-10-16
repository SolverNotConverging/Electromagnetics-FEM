"""One cell of a dielectric-rod leaky-wave antenna.

The rectangular dielectric rod runs along z and carries one PEC patch on its
upper (y+) face per period. Air and transverse PML surround the rod; only the
z faces are periodic.
"""

from math import pi
from pathlib import Path

import fem

MM = 1e-3
PERIOD = 10 * MM
OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

solver = fem.PeriodicModeSolver3D(frequency=20e9, x_range=(-9 * MM, 9 * MM), y_range=(-8 * MM, 8 * MM),
                                  z_range=(0.0, PERIOD), boundary=fem.materials.PEC)
solver.add_box(name="dielectric_rod", x_range=(-3 * MM, 3 * MM), y_range=(-2 * MM, 2 * MM),
               z_range=(0.0, PERIOD), material=fem.Material(name="dielectric rod", epsilon=6.0))
solver.add_box(name="periodic_metal_patch", x_range=(-2 * MM, 2 * MM), y_range=(2 * MM, 2.2 * MM),
               z_range=(3.5 * MM, 6.5 * MM), material=fem.materials.PEC)
# PML occupies the outer 3 mm on x-/x+/y-/y+; it never covers z.
solver.add_pml(thickness=3 * MM, direction="all")
solver.mesh(max_element_size=3 * MM, wavelength_elements=6)

modes = solver.solve(
    backend="auto",
    num_modes=2,
    neff_guess=0.5 - 0.02j,  # a fast-wave branch inside the air light cone
    eigensolver_tolerance=1e-8,
    ncv=80,  # enough radiation candidates for sparse shift-invert
    max_refinements=0,
)
print("neff:", modes.neff)
print("PML energy fractions:", [mode.pml_fraction for mode in modes])
for index, mode in enumerate(modes):
    # Fields vary as exp(-gamma*z), so Re(gamma) is the attenuation in Np/m.
    print(f"Mode {index}: attenuation = {mode.gamma.real:.6g} Np/m")
    # A Floquet harmonic radiates into air when |Re(beta_n)| < k0.
    radiating = [n for n in range(-2, 3) if abs(mode.beta.real + 2 * pi * n / PERIOD) < solver.k0]
    print(f"  space harmonics inside the air light cone: {radiating}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
modes.save(OUTPUT_DIR / "results.h5")
modes.show()
