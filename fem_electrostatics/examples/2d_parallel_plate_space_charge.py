"""Parallel plates with uniform space charge and an analytic peak potential."""

# Run directly from the downloaded repository without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from fem_electrostatics import shapes

from fem_electrostatics import ElectrostaticSolver
from fem_electrostatics import EPSILON_0


OUTPUT_DIR = _Path(__file__).resolve().parents[1] / "outputs" / _Path(__file__).stem


def main():
    charged = ElectrostaticSolver(dim=2, outer_potential=None, x_range=1., y_range=.5)
    charged.set_potential(potential=0.0, geometry='left')
    charged.set_potential(potential=0.0, geometry='right')
    charged.add_charge_density(density=EPSILON_0, geometry=shapes.Rectangle(bounds=((0.0, 1.0), (0.0, 0.5))))
    charged.mesh(max_element_size=.12)
    poisson = charged.solve(max_refinements=0)
    print("Charged guide peak potential (exact 0.125 V):", poisson.potential.max())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    charged.result.save(OUTPUT_DIR / "results.h5")
    charged.show()
    return poisson


if __name__ == "__main__":
    main()
