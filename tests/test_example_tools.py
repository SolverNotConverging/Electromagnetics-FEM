"""Import qualification must not start a simulation or open a window."""
import pytest

from scripts.qualify_examples import check_imports


def test_import_check_does_not_execute_example_body(tmp_path):
    path = tmp_path / "example.py"
    path.write_text('import numpy as np\nraise RuntimeError("simulation executed")\n')
    check_imports(path)


def test_import_check_reports_missing_dependency(tmp_path):
    path = tmp_path / "example.py"
    path.write_text("import nonexistent_fem_test_dependency\n")
    with pytest.raises(ModuleNotFoundError):
        check_imports(path)
