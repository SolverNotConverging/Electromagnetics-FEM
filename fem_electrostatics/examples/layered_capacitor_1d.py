"""Layered 1D capacitor compared with the analytic dielectric energy."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "cem_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from cem_common import Material

from fem_electrostatics import ElectrostaticSolver
from scipy.constants import epsilon_0 as EPSILON_0


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

    capacitor.show()
    return dielectric


if __name__ == "__main__":
    main()
