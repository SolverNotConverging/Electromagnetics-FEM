"""Public exception hierarchy for :mod:`fem.periodic`."""
from fem.common import errors as _common



class PeriodicModesError(_common.FEMError):
    """Base class for errors raised by the periodic FEM package."""


class ConfigurationError(PeriodicModesError, _common.ConfigurationError):
    """A solver option or material is invalid."""


class GeometryError(ConfigurationError, _common.GeometryError):
    """A geometry object cannot be represented by the periodic cell."""


class MeshError(PeriodicModesError, _common.MeshError):
    """Gmsh could not create a conforming periodic mesh."""


class NotDiscretizedError(PeriodicModesError):
    """An assembled operation was requested before :meth:`discretize`."""


class StaleDiscretizationError(PeriodicModesError):
    """The continuous geometry changed after discretization."""


class SolverError(PeriodicModesError, _common.SolverError):
    """The polynomial eigenproblem did not produce the requested modes."""


class BackendCapabilityError(PeriodicModesError, _common.BackendCapabilityError):
    """A requested feature is intentionally unavailable in this backend."""


from fem.common.errors import PersistenceError as _PersistenceError


class PersistenceError(PeriodicModesError, _PersistenceError):
    """A periodic-mode HDF5 archive is invalid or cannot be written."""


__all__ = [
    "BackendCapabilityError",
    "ConfigurationError",
    "PeriodicModesError",
    "GeometryError",
    "MeshError",
    "NotDiscretizedError",
    "PersistenceError",
    "SolverError",
    "StaleDiscretizationError",
]
