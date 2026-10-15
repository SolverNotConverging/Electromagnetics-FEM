"""Solve one cell of a dielectric-rod leaky-wave antenna.

The rectangular dielectric rod runs along z and carries one PEC patch on
its upper (y+) face per period. Air and transverse PML surround the rod;
only the z faces are periodic. All dimensions are in metres.
"""

from __future__ import annotations

# Run directly from the downloaded repository without installing solver packages.
import sys as _sys
from math import pi
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from fem_periodic_modes import Material, PeriodicModeSolver3D, materials


MM = 1.0e-3
FREQUENCY = 20.0e9
PERIOD = 10.0 * MM
OUTPUT_DIR = _Path(__file__).resolve().parents[1] / "outputs" / _Path(__file__).stem


def build_solver() -> PeriodicModeSolver3D:
    """Configure the unit cell, leaving meshing and solving to the caller."""
    dielectric = Material(name="dielectric rod", epsilon=6.0)
    solver = PeriodicModeSolver3D(
        frequency=FREQUENCY,
        x_range=(-9.0 * MM, 9.0 * MM),
        y_range=(-8.0 * MM, 8.0 * MM),
        z_range=(0.0, PERIOD),
        boundary=materials.PEC,
    )
    solver.add_box(
        name="dielectric_rod",
        x_range=(-3.0 * MM, 3.0 * MM),
        y_range=(-2.0 * MM, 2.0 * MM),
        z_range=(0.0, PERIOD),
        material=dielectric,
    )
    solver.add_box(
        name="periodic_metal_patch",
        x_range=(-2.0 * MM, 2.0 * MM),
        y_range=(2.0 * MM, 2.2 * MM),
        z_range=(3.5 * MM, 6.5 * MM),
        material=materials.PEC,
    )
    # PML occupies the outer 3 mm on x-/x+/y-/y+; it never covers z.
    # The exterior PEC walls terminate the PML, not the physical air region.
    solver.add_pml(thickness=3.0 * MM, direction="all")
    return solver


def main() -> None:
    solver = build_solver()
    solver.mesh(max_element_size=3.0 * MM, wavelength_elements=6)
    modes = solver.solve(backend="auto",
        num_modes=2,
        # Target a fast-wave branch directly inside the air light cone.
        neff_guess=0.5 - 0.02j,
        direction="forward",
        eigensolver="auto",
        eigensolver_tolerance=1.0e-8,
        # Include enough radiation candidates for the ARPACK shift-invert solve.
        ncv=80,
        max_refinements=0,
        max_pml_fraction=0.5,
    )
    print("neff:", modes.neff)
    print("PML energy fractions:", [mode.pml_fraction for mode in modes])
    for index, mode in enumerate(modes, start=1):
        # Fields vary as exp(-gamma*z), so Re(gamma) is attenuation in Np/m.
        print(f"Mode {index}: attenuation = {mode.gamma.real:.6g} Np/m")
        # A Floquet harmonic can radiate into air when |Re(beta_n)| < k0.
        radiating = [
            n for n in range(-2, 3)
            if abs(mode.beta.real + 2.0 * pi * n / PERIOD) < solver.k0
        ]
        print(f"  Space harmonics inside the air light cone: {radiating}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    solver.result.save(OUTPUT_DIR / "results.h5")
    solver.show()


if __name__ == "__main__":
    main()
