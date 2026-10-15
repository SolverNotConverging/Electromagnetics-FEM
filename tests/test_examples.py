"""Examples use only the public API; FEM_RUN_EXAMPLES=1 executes them.

``FEM_RUN_EXAMPLES=1`` runs every example except the 3D periodic ones;
``FEM_RUN_EXAMPLES=all`` adds those, which take much longer. Windows are
suppressed and outputs go to each example's ``outputs/`` folder.
"""

import ast
import os
import runpy
import sys
from pathlib import Path

import pytest

import fem

ROOT = Path(__file__).resolve().parents[1] / "examples"
EXAMPLES = sorted(p for p in ROOT.rglob("*.py") if "outputs" not in p.parts)
SCRIPTS = [p for p in EXAMPLES if "post_processing" not in p.parts]


@pytest.mark.parametrize("path", EXAMPLES, ids=[p.relative_to(ROOT).as_posix() for p in EXAMPLES])
def test_example_imports_only_public_names(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("fem"):
            module = __import__(node.module, fromlist=[a.name for a in node.names])
            for alias in node.names:
                assert hasattr(module, alias.name), f"{path.name}: {node.module}.{alias.name}"
            assert not any(part.startswith("_") for part in node.module.split(".")), path.name
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "fem":
            assert hasattr(fem, node.attr), f"{path.name}: fem.{node.attr}"
    assert "sys.path" not in source
    # Examples are plain top-to-bottom scripts, not modules.
    assert "__main__" not in source and "def main" not in source, path.name


def test_every_family_has_examples_and_an_index():
    for family in ("electrostatics", "waveguide", "periodic", "scattering"):
        assert (ROOT / family / "README.rst").is_file()
        assert list((ROOT / family).glob("*.py"))


@pytest.mark.skipif(
    os.environ.get("FEM_RUN_EXAMPLES") not in ("1", "all"), reason="set FEM_RUN_EXAMPLES=1 to run examples"
)
@pytest.mark.parametrize("path", SCRIPTS, ids=[p.relative_to(ROOT).as_posix() for p in SCRIPTS])
def test_example_runs(path, monkeypatch):
    if path.name.startswith("3d_") and os.environ.get("FEM_RUN_EXAMPLES") != "all":
        pytest.skip("3D examples run with FEM_RUN_EXAMPLES=all")
    from fem.common.contracts import ResultMixin, SolverMixin

    monkeypatch.setattr(ResultMixin, "show", lambda self, **kwargs: None)
    monkeypatch.setattr(SolverMixin, "show", lambda self, **kwargs: None)
    monkeypatch.setattr("matplotlib.pyplot.show", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", [str(path)])
    runpy.run_path(str(path), run_name="__main__")


@pytest.mark.skipif(
    os.environ.get("FEM_RUN_EXAMPLES") not in ("1", "all"), reason="set FEM_RUN_EXAMPLES=1 to run examples"
)
def test_readme_example_runs(tmp_path, monkeypatch):
    import re

    from fem.common.contracts import ResultMixin

    readme = (ROOT.parent / "README.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", readme, re.S).group(1)
    monkeypatch.setattr(ResultMixin, "show", lambda self, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(compile(code, "README.md", "exec"), namespace)
    neff = namespace["modes"][0].neff
    assert neff.real == pytest.approx(1.887, abs=0.01) and neff.imag < 0
    assert (tmp_path / "microstrip.h5").is_file()
