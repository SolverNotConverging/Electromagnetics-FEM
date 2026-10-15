"""Electrostatic potentials, fields, charges and energies in 1D and 2D."""

from .exceptions import ElectrostaticsError, NotDiscretizedError
from .results import ElectrostaticResult
from .solver import ElectrostaticSolver

__all__ = ["ElectrostaticResult", "ElectrostaticSolver", "ElectrostaticsError", "NotDiscretizedError"]
