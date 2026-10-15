"""Solver APIs and examples expose one execution backend selector."""
import ast
import inspect
from pathlib import Path


def test_public_backend_keyword():
    from fem.electrostatics import ElectrostaticSolver
    from fem.waveguide import ModeSolver1D, ModeSolver2D
    from fem.periodic import PeriodicModeSolver2D, PeriodicModeSolver3D
    for cls in (ElectrostaticSolver, ModeSolver1D, ModeSolver2D, PeriodicModeSolver2D, PeriodicModeSolver3D):
        parameters = inspect.signature(cls.solve).parameters
        assert "backend" in parameters
        assert "arnoldi_backend" not in parameters
        assert "linear_solver_backend" not in parameters


def test_examples_explicitly_select_backend():
    examples = sorted((Path(__file__).resolve().parents[1] / "examples").rglob("*.py"))
    assert examples
    for path in examples:
        if True:
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("solve", "solve_modes", "sweep"):
                    assert any(item.arg == "backend" for item in node.keywords), str(path)
