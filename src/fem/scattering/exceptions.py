"""FEM Waveguide Scattering exception hierarchy."""

from __future__ import annotations

from fem.common import errors as _common




class WaveguideScatteringError(_common.FEMError):
    """Base class for actionable FEM Waveguide Scattering errors."""


class ConfigurationError(WaveguideScatteringError, _common.ConfigurationError):
    """The requested simulation configuration is incomplete or inconsistent."""


class MaterialError(ConfigurationError):
    """A material value or constitutive representation is invalid."""


class MeshError(WaveguideScatteringError, _common.MeshError):
    """Mesh generation, import, or physical-region tagging failed."""


class ModeSolverError(WaveguideScatteringError):
    """The guided-mode eigenproblem could not produce valid requested modes."""


class ModeProjectionError(WaveguideScatteringError):
    """Fields could not be projected reliably onto the requested lead modes."""


class SolverError(WaveguideScatteringError, _common.SolverError):
    """A finite-element linear or eigenvalue solve failed."""


class ViewerError(WaveguideScatteringError, _common.ViewerError):
    """The standalone native viewer could not be found or launched."""


__all__ = [
    "ConfigurationError",
    "MaterialError",
    "MeshError",
    "ModeProjectionError",
    "ModeSolverError",
    "SolverError",
    "ViewerError",
    "WaveguideScatteringError",
]
