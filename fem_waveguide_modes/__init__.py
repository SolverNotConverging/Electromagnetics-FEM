"""Fem Waveguide Modes. Public user API for version 1.1.0."""

from pathlib import Path as _Path
__path__.append(str(_Path(__file__).parent / "src"))

from .solver_1d import ModeSolver1D
from .solver_2d import ModeSolver2D
from .results import Mode
from .results import ModeSet
from .results import SampledFields
from fem_common.errors import BackendCapabilityError
from fem_common.errors import ConfigurationError
from .exceptions import FEMModeSolverError
from fem_common.errors import GeometryError
from fem_common.errors import MeshError
from fem_common.errors import SolverError
from .result_api import load_result
from fem_common import NoResultError
from fem_common import PersistenceError

__version__ = "1.1.2"
__all__ = ['ModeSolver1D', 'ModeSolver2D', 'Mode', 'ModeSet', 'SampledFields', 'BackendCapabilityError', 'ConfigurationError', 'FEMModeSolverError', 'GeometryError', 'MeshError', 'SolverError', 'load_result', 'NoResultError', 'PersistenceError']

# Shared construction tools are available directly from the solver family.
from fem_common import Material, GoodConductor, SurfaceImpedance, materials, shapes, EPSILON_0, MU_0, C_0
__all__ += ['Material', 'GoodConductor', 'SurfaceImpedance', 'materials', 'shapes', 'EPSILON_0', 'MU_0', 'C_0']

from fem_common.dispersion import plot_dispersion
__all__ += ["plot_dispersion"]
