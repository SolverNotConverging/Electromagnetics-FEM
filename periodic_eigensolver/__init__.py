"""Shared refined shift-and-invert Arnoldi solver for periodic pencils."""

from pathlib import Path as _Path
__path__.append(str(_Path(__file__).parent / "src"))

# Direct examples use source Python modules and the installed wheel's kernel.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError, distribution as _distribution
try:
    _installed = _Path(_distribution("electromagnetics-fem").locate_file("periodic_eigensolver"))
    if _installed.is_dir() and _installed.resolve() != _Path(__file__).parent.resolve():
        __path__.append(str(_installed))
except _PackageNotFoundError:
    pass


from .refined import (
    ArnoldiResult,
    native_backend_available,
    solve_generalized,
)

__all__ = [
    "ArnoldiResult",
    "native_backend_available",
    "solve_generalized",
]

__version__ = "1.1.1"
