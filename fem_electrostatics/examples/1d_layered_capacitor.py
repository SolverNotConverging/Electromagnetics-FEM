"""Layered 1D capacitor compared with the analytic dielectric energy."""

# Run directly from the downloaded repository without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from fem_electrostatics import Material

from fem_electrostatics import ElectrostaticSolver
from fem_electrostatics import EPSILON_0


OUTPUT_DIR = _Path(__file__).resolve().parents[1] / "outputs" / _Path(__file__).stem


def main():
    dielectric = Material(name="upper dielectric", epsilon=4.0)
    capacitor = ElectrostaticSolver(dim=1, outer_potential=None, x_range=(0.0, 1.0))
    capacitor.add_layer(x_range=(0.5, 1.0), material=dielectric)
    capacitor.set_potential(potential=0.0, name='ground', geometry='left')
    capacitor.set_potential(potential=1.0, name='drive', geometry='right')
    capacitor.mesh(max_element_size=.1)
    dielectric = capacitor.solve(max_refinements=0)
    print("Layered capacitor energy:", dielectric.energy)
    print("Expected energy:", .5 * EPSILON_0 / (.5 + .5 / 4.))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    capacitor.result.save(OUTPUT_DIR / "results.h5")
    capacitor.show()
    return dielectric


if __name__ == "__main__":
    main()
