"""Fem Periodic Modes. Public user API."""


from .solver_2d import PeriodicModeSolver2D
from .solver_3d import PeriodicModeSolver3D
from .results import PeriodicMode
from .results import PeriodicModeSet
from .results import PeriodicSampledFields
from .result_api import PeriodicSweepResult
from fem_common.errors import BackendCapabilityError
from fem_common.errors import ConfigurationError
from .exceptions import FEMPeriodicSolverError
from fem_common.errors import GeometryError
from fem_common.errors import MeshError
from fem_common import PersistenceError
from fem_common.errors import SolverError
from .result_api import load_result
from fem_common import NoResultError

from fem._version import __version__
__all__ = ['PeriodicModeSolver2D', 'PeriodicModeSolver3D', 'PeriodicMode', 'PeriodicModeSet', 'PeriodicSampledFields', 'PeriodicSweepResult', 'BackendCapabilityError', 'ConfigurationError', 'FEMPeriodicSolverError', 'GeometryError', 'MeshError', 'PersistenceError', 'SolverError', 'load_result', 'NoResultError']

# Shared construction tools are available directly from the solver family.
from fem_common import Material, GoodConductor, SurfaceImpedance, materials, shapes, EPSILON_0, MU_0, C_0
__all__ += ['Material', 'GoodConductor', 'SurfaceImpedance', 'materials', 'shapes', 'EPSILON_0', 'MU_0', 'C_0']

from .dispersion import plot_dispersion
__all__ += ["plot_dispersion"]
