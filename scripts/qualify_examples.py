"""Import or run direct examples without installing solver packages."""
from pathlib import Path
import argparse
import ast
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = """import runpy, sys
from unittest.mock import patch
from fem_common.contracts import SolverMixin, ResultMixin
with patch.object(SolverMixin, 'show'), patch.object(ResultMixin, 'show'), patch('matplotlib.pyplot.show'):
    runpy.run_path(sys.argv[1], run_name='__main__')
"""

def check_imports(path):
    """Compile the complete example and execute only its module-level imports."""
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    compile(tree, str(path), "exec")
    imports = ast.Module(body=[node for node in tree.body
                              if isinstance(node, (ast.Import, ast.ImportFrom))],
                         type_ignores=[])
    exec(compile(imports, str(path), "exec"), {"__name__": "example_import_check"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-only", action="store_true")
    args = parser.parse_args()
    examples = sorted(ROOT.glob("*/examples/**/*.py"))
    examples = [p for p in examples if p.name != "__init__.py"]
    runner = "import sys; from scripts.qualify_examples import check_imports; check_imports(sys.argv[1])"
    if not args.import_only:
        runner = RUN
    for path in examples:
        subprocess.run([sys.executable, "-c", runner, str(path)], cwd=ROOT,
                       env={**os.environ, "MPLBACKEND": "Agg", "CEM_EXAMPLE_QUALIFICATION": "1"}, check=True)
    print(f"Validated {len(examples)} direct examples.")

if __name__ == "__main__":
    main()
