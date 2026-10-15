"""Command-line entry point: ``fem info``, ``fem view`` and ``fem inspect``."""
import argparse
from pathlib import Path

from . import __version__


def _inspect(path: Path) -> int:
    from fem.common.persistence import load

    result = load(path)
    print(f"path={path.resolve()}")
    print(f"type={type(result).__name__}")
    for name in ("dimension", "frequency", "frequencies", "ndofs"):
        value = getattr(result, name, None)
        if value is not None:
            print(f"{name}={value}")
    try:
        print(f"count={len(result)}")
    except TypeError:
        pass
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fem", description="FEM electromagnetic solvers")
    parser.add_argument("--version", action="version", version=f"FEM {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info", help="print version and automatic solver backend")
    view = commands.add_parser("view", help="open a result archive in the interactive viewer")
    view.add_argument("path", type=Path)
    inspect = commands.add_parser("inspect", help="print a short summary of a result archive")
    inspect.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "info":
        from fem.common.linalg import resolve_backend
        print(f"FEM {__version__}")
        print(f"Automatic solver backend: {resolve_backend()}")
        return 0
    if args.command == "view":
        from fem.common.viewer import show
        show(args.path)
        return 0
    return _inspect(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
