"""Build the single complete Windows CPython 3.12 release wheel."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "fem_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from pathlib import Path
import argparse
import os
import subprocess
import sys
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
SOURCE_ROOTS = tuple(ROOT / name for name in ('fem', 'fem_common', 'fem_adaptivity', 'fem_electrostatics', 'fem_periodic_modes', 'fem_waveguide_modes', 'fem_waveguide_scattering', 'periodic_eigensolver'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/dist")
    parser.add_argument("--native-bundle", type=Path, default=ROOT / "build/native-release-1.1.0/FEM-1.1.0-windows-x64")
    args = parser.parse_args()
    if sys.platform != "win32" or sys.version_info[:2] != (3, 12):
        parser.error("The complete release wheel targets Windows x64 / CPython 3.12.")
    if sys.maxsize <= 2**32:
        parser.error("A 64-bit interpreter is required.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("*.whl")):
        parser.error("Use an output directory with no existing wheels.")
    bundle = args.native_bundle.resolve()
    if not (bundle / "SOURCE_INDEX.md").is_file():
        parser.error("Run package_native_windows.py --phase stage, then --phase finish first.")
    environment = dict(os.environ, FEM_NATIVE_BUNDLE=str(bundle))
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(output), str(ROOT)],
        cwd=ROOT,
        env=environment,
        check=True,
    )
    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1 or wheels[0].name != f"electromagnetics_fem-{VERSION}-cp312-cp312-win_amd64.whl":
        raise SystemExit(f"Unexpected release artifacts: {wheels}")
    with zipfile.ZipFile(wheels[0]) as archive:
        members = set(archive.namelist())
        expected = set()
        for source in SOURCE_ROOTS:
            expected.update(path.relative_to(ROOT).as_posix() for path in source.rglob("*.py") if path.relative_to(ROOT).parts[0] in ('fem', 'fem_common', 'fem_adaptivity', 'fem_electrostatics', 'fem_periodic_modes', 'fem_waveguide_modes', 'fem_waveguide_scattering', 'periodic_eigensolver') and not any(part in ("examples", "scripts", "docs", "__pycache__") for part in path.relative_to(ROOT).parts))
        packaged = {name for name in members if name.endswith(".py") and not name.startswith("fem/native/")}
        if packaged != expected:
            raise SystemExit(f"Wheel source mismatch: {packaged ^ expected}")
        if not any(name.startswith("periodic_eigensolver/_cython_kernels") and name.endswith(".pyd") for name in members):
            raise SystemExit("The compiled periodic eigensolver is missing.")
        for name in ("transmission-line-calculator", "transmission-line-calculator-cli", "fem-periodic-mode-viewer",
                     "fem-periodic-mode-inspect", "fem-waveguide-scattering-viewer", "fem-waveguide-scattering-viewer-inspect"):
            if f"fem/native/bin/{name}.exe" not in members:
                raise SystemExit(f"Missing native application: {name}")
        for path in bundle.rglob("*"):
            if path.is_file():
                name = "fem/native/" + path.relative_to(bundle).as_posix()
                if name not in members or archive.read(name) != path.read_bytes():
                    raise SystemExit(f"Native runtime differs from the qualified bundle: {name}")
    print(f"Complete release wheel: {wheels[0]}")


if __name__ == "__main__":
    main()
