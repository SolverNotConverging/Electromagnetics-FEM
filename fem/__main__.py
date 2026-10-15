"""Small command-line entry point for installation checks and native GUIs."""
import argparse

from . import __version__
from ._apps import launch
from fem_common._native import bundled_executable


def main():
    parser = argparse.ArgumentParser(description="FEM solvers and native applications")
    parser.add_argument("--version", action="version", version=f"FEM {__version__}")
    parser.add_argument("command", choices=("info", "calculator", "calculator-cli", "periodic-viewer", "scattering-viewer"))
    args, remaining = parser.parse_known_args()
    if args.command == "info":
        if remaining:
            parser.error(f"Unrecognized arguments: {' '.join(remaining)}")
        from fem_common.linalg import resolve_backend
        print(f"FEM {__version__}")
        print(f"Automatic solver backend: {resolve_backend()}")
        for name in ("transmission-line-calculator", "fem-periodic-mode-viewer", "fem-waveguide-scattering-viewer"):
            print(f"{name}: {bundled_executable(name) or 'not bundled'}")
        return 0
    names = {"calculator": "transmission-line-calculator", "calculator-cli": "transmission-line-calculator-cli",
             "periodic-viewer": "fem-periodic-mode-viewer", "scattering-viewer": "fem-waveguide-scattering-viewer"}
    return launch(names[args.command], remaining)


if __name__ == "__main__":
    raise SystemExit(main())
