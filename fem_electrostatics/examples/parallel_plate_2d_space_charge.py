"""Parallel plates with uniform space charge and an analytic peak potential."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "cem_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from cem_common import shapes

from fem_electrostatics import ElectrostaticSolver
from scipy.constants import epsilon_0 as EPSILON_0


def main():
    charged = ElectrostaticSolver(dim=2, outer_potential=None, x_range=1., y_range=.5)
    charged.set_potential(potential=0.0, geometry='left')
    charged.set_potential(potential=0.0, geometry='right')
    charged.add_charge_density(density=EPSILON_0, geometry=shapes.Rectangle(bounds=((0.0, 1.0), (0.0, 0.5))))
    charged.mesh(max_element_size=.12)
    poisson = charged.solve(max_refinements=0)
    print("Charged guide peak potential (exact 0.125 V):", poisson.potential.max())
    charged.show()
    return poisson


if __name__ == "__main__":
    main()
