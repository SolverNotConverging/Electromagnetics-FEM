"""Solve a small full-vector rectangular dielectric-waveguide example."""

from __future__ import annotations

# Run directly from the downloaded repository without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from fem_waveguide_modes import Material, materials

from fem_waveguide_modes import ModeSolver2D


OUTPUT_DIR = _Path(__file__).resolve().parents[1] / "outputs" / _Path(__file__).stem


def main() -> None:
    frequency = 10e14
    silicon = Material(name="silicon core", epsilon=3.45**2)
    silica = Material(name="silica slab", epsilon=1.44**2)
    solver = ModeSolver2D(frequency=frequency, x_range=(-2e-06, 2e-06), y_range=(-1.5e-06, 1.5e-06), boundary=materials.PEC, background_material=materials.vacuum)
    solver.add_rectangle(x_range=(-1e-07, 1e-07), y_range=(-1.1e-07, 1.1e-07), name='silicon_core', material=silicon)
    solver.add_rectangle(x_range=(-1e-06, 1e-06), y_range=(-2e-07, -1.1e-07), name='slab', material=silica)

    solver.mesh(max_element_size=1e-07, quadrature_order=4)
    modes = solver.solve(max_refinements=0, num_modes=4)

    for number, mode in enumerate(modes):
        print(
            f"mode {number}: neff={mode.neff:.9g}, "
            f"residual={mode.residual:.3e}, "
            f"divergence={mode.divergence_residual:.3e}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    solver.result.save(OUTPUT_DIR / "results.h5")
    solver.show()


if __name__ == "__main__":
    main()
