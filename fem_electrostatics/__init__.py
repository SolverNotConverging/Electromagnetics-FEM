"""Fem Electrostatics. Public user API for version 1.1.0."""

from pathlib import Path as _Path
__path__.append(str(_Path(__file__).parent / "src"))

from .solver import ElectrostaticSolver
from .results import ElectrostaticResult
from .exceptions import ElectrostaticSolverError
from fem_common.errors import GeometryError
from fem_common.errors import MeshError
from fem_common.errors import SolverError
from .result_api import load_result
from fem_common import NoResultError
from fem_common import PersistenceError

__version__ = "1.1.1"
__all__ = ['ElectrostaticSolver', 'ElectrostaticResult', 'ElectrostaticSolverError', 'GeometryError', 'MeshError', 'SolverError', 'load_result', 'NoResultError', 'PersistenceError']

# Shared construction tools are available directly from the solver family.
from fem_common import Material, GoodConductor, SurfaceImpedance, materials, shapes, EPSILON_0, MU_0, C_0
__all__ += ['Material', 'GoodConductor', 'SurfaceImpedance', 'materials', 'shapes', 'EPSILON_0', 'MU_0', 'C_0']
