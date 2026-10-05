"""Cross-family checks for the 1.0 material-first public contract."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from fem_common import Material, SurfaceImpedance, materials, shapes
from fem_common.errors import BackendCapabilityError, ConfigurationError, GeometryError


def test_materials_are_named_reusable_values_with_exp_plus_iwt_loss_sign() -> None:
    dielectric = Material(name="lossy dielectric", epsilon=2.5 - 0.03j, mu=1.0)

    assert dielectric.is_passive
    assert not Material(name="active", epsilon=2.5 + 0.03j).is_passive
    assert materials.vacuum.name == "vacuum"
    assert materials.air.is_passive
    assert materials.copper.at_frequency(frequency=10e9).real > 0.0
    assert materials.copper.at_frequency(frequency=10e9).imag > 0.0
    assert SurfaceImpedance(impedance=1.0 + 1.0j).at_frequency(frequency=10e9) == 1.0 + 1.0j

    with pytest.raises(ConfigurationError):
        Material(name="", epsilon=1.0)
    with pytest.raises(BackendCapabilityError, match="off-diagonal"):
        materials.bulk_values(
            Material(name="tensor", epsilon=((2.0, 0.1), (0.1, 2.0))),
            dimension=2,
        )


def test_shared_shapes_support_boolean_and_transformed_geometry() -> None:
    disk = shapes.Circle(center=(0.0, 0.0), radius=1.0)
    aperture = shapes.Rectangle(bounds=((-0.25, 0.25), (-2.0, 2.0)))
    split_disk = shapes.Difference(shape=disk, tool=aperture)
    moved = split_disk.translated(offset=(2.0, 0.0)).rotated(
        angle=90.0,
        center=(2.0, 0.0),
    )

    assert split_disk.contains(0.75, 0.0)
    assert not split_disk.contains(0.0, 0.0)
    assert moved.contains(2.0, 0.75)
    assert shapes.Annulus(center=(0.0, 0.0), inner_radius=0.5, outer_radius=1.0).contains(0.75, 0.0)
    assert shapes.Ellipsoid(center=(0.0, 0.0, 0.0), radii=(1.0, 2.0, 3.0)).dimension == 3

    with pytest.raises(GeometryError):
        shapes.Polygon(points=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))


@pytest.mark.gmsh
@pytest.mark.parametrize(
    "shape",
    (
        shapes.Ellipse(center=(0.5, 0.5), radii=(0.25, 0.15)),
        shapes.RoundedRectangle(bounds=((0.2, 0.8), (0.2, 0.8)), radius=0.1),
        shapes.Difference(
            shape=shapes.Circle(center=(0.5, 0.5), radius=0.3),
            tool=shapes.Circle(center=(0.5, 0.5), radius=0.12),
        ),
    ),
)
def test_shared_extended_shapes_mesh_in_fem(shape) -> None:
    from fem_waveguide_modes import ModeSolver2D

    solver = ModeSolver2D(frequency=299_792_458.0, x_range=1.0, y_range=1.0)
    solver.add_geometry(
        shape=shape,
        material=Material(name="inclusion", epsilon=2.25),
    )
    mesh = solver.mesh(max_element_size=0.25, material_aware=False)
    assert mesh.elements.size > 0


@pytest.mark.parametrize(
    ("package", "solver_name"),
    (
        ("fem_waveguide_modes", "ModeSolver1D"),
        ("fem_waveguide_modes", "ModeSolver2D"),
        ("fem_periodic_modes", "PeriodicModeSolver2D"),
        ("fem_periodic_modes", "PeriodicModeSolver3D"),
        ("fem_waveguide_scattering", "WaveguideScatteringSolver2D"),
        ("fem_electrostatics", "ElectrostaticSolver"),
    ),
)
def test_public_solver_configuration_is_keyword_only(package: str, solver_name: str) -> None:
    module = __import__(package, fromlist=(solver_name,))
    signature = inspect.signature(getattr(module, solver_name))
    assert all(
        parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
        for parameter in signature.parameters.values()
    )


def test_clean_break_removes_obsolete_solver_workflows() -> None:
    solver_types = []
    for package, names in {
        "fem_waveguide_modes": ("ModeSolver1D", "ModeSolver2D"),
        "fem_periodic_modes": ("PeriodicModeSolver2D", "PeriodicModeSolver3D"),
        "fem_waveguide_scattering": ("WaveguideScatteringSolver2D",),
        "fem_electrostatics": ("ElectrostaticSolver",),
    }.items():
        module = __import__(package, fromlist=names)
        solver_types.extend(getattr(module, name) for name in names)

    obsolete = ("add_pec", "add_pmc", "add_object", "add_region", "run", "visualize_with_gui")
    for solver_type in solver_types:
        assert not any(hasattr(solver_type, name) for name in obsolete)






def test_solver_families_export_shared_construction_tools():
    import importlib
    import fem_common
    for family in ('fem_electrostatics', 'fem_periodic_modes',
                   'fem_waveguide_modes', 'fem_waveguide_scattering'):
        solver = importlib.import_module(family)
        for name in ('Material', 'GoodConductor', 'SurfaceImpedance',
                     'materials', 'shapes', 'EPSILON_0', 'MU_0', 'C_0'):
            assert getattr(solver, name) is getattr(fem_common, name)
            assert name in solver.__all__
