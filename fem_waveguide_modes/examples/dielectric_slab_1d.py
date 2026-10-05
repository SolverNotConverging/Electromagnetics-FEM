"""Solve and visualize the leading modes of a dielectric slab."""

from __future__ import annotations

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "cem_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from cem_common import Material

from fem_waveguide_modes import ModeSolver1D


def main() -> None:
    wavelength = 1.55e-6
    frequency = 299_792_458.0 / wavelength
    cladding = Material(name="silica cladding", epsilon=1.44**2)
    silicon = Material(name="silicon core", epsilon=3.45**2)
    solver = ModeSolver1D(
        frequency=frequency,
        x_range=(-3e-6, 3e-6),
        background_material=cladding,
    )
    solver.add_layer(x_range=(-2.5e-7, 2.5e-7), name="core", material=silicon)

    solver.mesh(max_element_size=6e-08)
    modes = solver.solve(neff_guess=3.2, max_refinements=0, num_modes=4)

    for number, mode in enumerate(modes):
        print(
            f"mode {number}: polarization={mode.polarization}, "
            f"neff={mode.neff:.9g}, alpha={mode.alpha:.4g} 1/m, "
            f"residual={mode.residual:.3e}"
        )

    solver.show()


if __name__ == "__main__":
    main()
