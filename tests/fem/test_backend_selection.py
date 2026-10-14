"""Solver APIs and examples expose one execution backend selector."""
import ast
import inspect
from pathlib import Path


def test_public_backend_keyword():
    from fem_electrostatics import ElectrostaticSolver
    from fem_waveguide_modes import ModeSolver1D, ModeSolver2D
    from fem_periodic_modes import PeriodicModeSolver2D, PeriodicModeSolver3D
    for cls in (ElectrostaticSolver, ModeSolver1D, ModeSolver2D, PeriodicModeSolver2D, PeriodicModeSolver3D):
        parameters = inspect.signature(cls.solve).parameters
        assert "backend" in parameters
        assert "arnoldi_backend" not in parameters
        assert "linear_solver_backend" not in parameters


def test_examples_explicitly_select_backend():
    root = Path(__file__).resolve().parents[2]
    for package in ("fem_electrostatics", "fem_waveguide_modes", "fem_periodic_modes", "fem_waveguide_scattering"):
        for path in (root / package / "examples").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("solve", "solve_modes", "sweep"):
                    assert any(item.arg == "backend" for item in node.keywords), str(path)
