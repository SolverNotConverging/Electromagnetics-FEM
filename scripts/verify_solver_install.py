"""Smoke-test an installed solver wheel from outside the checkout."""

from importlib import import_module
from importlib.metadata import version
from pathlib import Path

import numpy as np


def main():
    checkout = Path(__file__).resolve().parents[1]
    expected = version("electromagnetics-fem")
    for name in (
        "fem",
        "fem_common",
        "fem_adaptivity",
        "fem_electrostatics",
        "fem_waveguide_modes",
        "fem_periodic_modes",
        "fem_waveguide_scattering",
    ):
        module = import_module(name)
        assert not Path(module.__file__).resolve().is_relative_to(checkout / name), module.__file__
        assert module.__version__ == expected, name
        assert len(module.__path__) == 1, name

    from fem_electrostatics import ElectrostaticSolver

    solver = ElectrostaticSolver(dim=1, x_range=1.0, outer_potential=None)
    solver.set_potential(geometry="left", potential=0.0)
    solver.set_potential(geometry="right", potential=1.0)
    solver.mesh(max_element_size=0.2)
    result = solver.solve(max_refinements=0)
    np.testing.assert_allclose(result.potential, result.coordinates[:, 0], atol=1e-12)
    print(f"Installed solver wheel {expected}: imports, electrostatics passed.")


if __name__ == "__main__":
    main()
