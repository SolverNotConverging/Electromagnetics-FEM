"""Scattering, S-parameters and radiation of 2D waveguide discontinuities."""

from .exceptions import MaterialError, ModeProjectionError, ModeSolverError, WaveguideScatteringError
from .farfield import ClosedContourFields, FarFieldResult, LayeredExterior
from .incident import IncidentMode
from .modes import PortMode, PortModeSet
from .results import Diagnostic, DiagnosticReport, ScatteringResult
from .scattering import ScatteringSolver2D
from .sweep import ScatteringSweepResult

__all__ = [
    "ClosedContourFields",
    "Diagnostic",
    "DiagnosticReport",
    "FarFieldResult",
    "IncidentMode",
    "LayeredExterior",
    "MaterialError",
    "ModeProjectionError",
    "ModeSolverError",
    "PortMode",
    "PortModeSet",
    "ScatteringResult",
    "ScatteringSolver2D",
    "ScatteringSweepResult",
    "WaveguideScatteringError",
]
