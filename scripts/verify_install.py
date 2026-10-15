"""Smoke-test an installed wheel from outside the checkout."""

from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import numpy as np


def main():
    checkout = Path(__file__).resolve().parents[1]
    import fem

    assert not Path(fem.__file__).resolve().is_relative_to(checkout), fem.__file__
    assert fem.__version__ == version("electromagnetics-fem")
    for name in ("common", "adaptivity", "electrostatics", "waveguide", "periodic", "scattering"):
        import_module(f"fem.{name}")

    solver = fem.ElectrostaticSolver(dim=1, x_range=1.0, outer_potential=None)
    solver.set_potential(geometry="left", potential=0.0)
    solver.set_potential(geometry="right", potential=1.0)
    solver.mesh(max_element_size=0.2)
    result = solver.solve(max_refinements=0)
    np.testing.assert_allclose(result.potential, result.coordinates[:, 0], atol=1e-12)
    print(f"Installed FEM {fem.__version__}: imports and electrostatics passed.")


if __name__ == "__main__":
    main()
