"""Exception hierarchy for the standalone FEM mode solvers."""
from fem.common import errors as _common



class WaveguideModesError(_common.FEMError):
    """Base class for all public FEM mode-solver errors."""


class ConfigurationError(WaveguideModesError, _common.ConfigurationError):
    """The continuous model or solver configuration is invalid."""


class GeometryError(ConfigurationError, _common.GeometryError):
    """A geometry primitive or region operation is invalid."""


class MeshError(WaveguideModesError, _common.MeshError):
    """The continuous model could not be discretized."""


class NotDiscretizedError(WaveguideModesError):
    """A solve or plot was requested before discretization."""


class StaleDiscretizationError(WaveguideModesError):
    """The continuous model changed after it was discretized."""


class SolverError(WaveguideModesError, _common.SolverError):
    """The polynomial eigenproblem could not produce valid modes."""


class BackendCapabilityError(WaveguideModesError, _common.BackendCapabilityError):
    """A requested physical feature is not supported by this FEM backend."""


__all__ = [
    "BackendCapabilityError",
    "ConfigurationError",
    "WaveguideModesError",
    "GeometryError",
    "MeshError",
    "NotDiscretizedError",
    "SolverError",
    "StaleDiscretizationError",
]
