"""Exceptions raised by the finite-element electrostatic solver."""
from fem.common import errors as _common



class ElectrostaticsError(_common.FEMError, RuntimeError):
    """Base exception for this package."""


class GeometryError(ElectrostaticsError, _common.GeometryError):
    """The continuous geometry or material definition is invalid."""


class MeshError(ElectrostaticsError, _common.MeshError):
    """Gmsh could not produce a valid conforming mesh."""


class NotDiscretizedError(ElectrostaticsError):
    """An operation requires a current finite-element mesh."""


class SolverError(ElectrostaticsError, _common.SolverError):
    """The assembled electrostatic boundary-value problem cannot be solved."""
