"""Shared materials, shapes, constants, result contracts, errors and the viewer."""

from . import materials, shapes
from .constants import C_0, EPSILON_0, ETA_0, MU_0, TIME_CONVENTION
from .contracts import MeshSnapshot
from .errors import (
    BackendCapabilityError,
    ConfigurationError,
    FEMError,
    GeometryError,
    MeshError,
    NoResultError,
    PersistenceError,
    SolverError,
    ViewerError,
)
from .materials import GoodConductor, Material, SurfaceImpedance

__all__ = [
    "C_0",
    "EPSILON_0",
    "ETA_0",
    "MU_0",
    "TIME_CONVENTION",
    "BackendCapabilityError",
    "ConfigurationError",
    "FEMError",
    "GeometryError",
    "GoodConductor",
    "Material",
    "MeshError",
    "MeshSnapshot",
    "NoResultError",
    "PersistenceError",
    "SolverError",
    "SurfaceImpedance",
    "ViewerError",
    "materials",
    "shapes",
]
