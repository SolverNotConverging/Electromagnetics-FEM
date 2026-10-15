"""Finite-element electromagnetic solvers.

Every solver follows the same workflow, here for a copper microstrip on a lossy
substrate::

    import fem

    fr4 = fem.Material(name="FR-4", epsilon=4.4 * (1 - 0.02j))  # lossy dielectric, tan(delta) = 0.02
    copper = fem.materials.copper                                # surface-impedance conductor

    solver = fem.ModeSolver2D(frequency=10e9, x_range=(-6e-3, 6e-3), y_range=(-35e-6, 6e-3),
                              boundary=fem.materials.PEC)
    solver.add_rectangle(x_range=(-6e-3, 6e-3), y_range=(0, 1.6e-3), name="substrate", material=fr4)
    solver.add_rectangle(x_range=(-6e-3, 6e-3), y_range=(-35e-6, 0), name="ground", material=copper)
    solver.add_rectangle(x_range=(-1.5e-3, 1.5e-3), y_range=(1.6e-3, 1.635e-3), name="strip", material=copper)
    solver.mesh(max_element_size=0.5e-3)
    modes = solver.solve(num_modes=1)

    print(f"neff = {modes[0].neff:.4f}, loss = {modes[0].alpha * 8.686:.1f} dB/m")
    modes.save("microstrip.h5")
    modes.show()  # fields, mesh and material regions

``fem.load(path)`` reopens any saved result. Units are SI and phasors use
``exp(+i*omega*t)``; see ``docs/conventions.md``.
"""

from ._version import __version__
from .common import materials, shapes
from .common.constants import C_0, EPSILON_0, MU_0, TIME_CONVENTION
from .common.dispersion import plot_dispersion
from .common.errors import (
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
from .common.materials import GoodConductor, Material, SurfaceImpedance
from .common.persistence import load
from .common.viewer import show
from .electrostatics import ElectrostaticResult, ElectrostaticSolver
from .periodic import (
    PeriodicMode,
    PeriodicModeSet,
    PeriodicModeSolver2D,
    PeriodicModeSolver3D,
    PeriodicSweepResult,
)
from .scattering import ScatteringResult, ScatteringSolver2D, ScatteringSweepResult
from .waveguide import Mode, ModeSet, ModeSolver1D, ModeSolver2D

__all__ = [
    "C_0",
    "EPSILON_0",
    "MU_0",
    "TIME_CONVENTION",
    "BackendCapabilityError",
    "ConfigurationError",
    "ElectrostaticResult",
    "ElectrostaticSolver",
    "FEMError",
    "GeometryError",
    "GoodConductor",
    "Material",
    "MeshError",
    "Mode",
    "ModeSet",
    "ModeSolver1D",
    "ModeSolver2D",
    "NoResultError",
    "PeriodicMode",
    "PeriodicModeSet",
    "PeriodicModeSolver2D",
    "PeriodicModeSolver3D",
    "PeriodicSweepResult",
    "PersistenceError",
    "ScatteringResult",
    "ScatteringSolver2D",
    "ScatteringSweepResult",
    "SolverError",
    "SurfaceImpedance",
    "ViewerError",
    "__version__",
    "load",
    "materials",
    "plot_dispersion",
    "shapes",
    "show",
]
