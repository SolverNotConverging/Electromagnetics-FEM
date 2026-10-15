"""Shared scattering test devices."""
import pytest


MM = 1.0e-3


def build_grounded_slot(frequency: float = 20.0e9, *, matched_ports: bool = False):
    """The grounded-slab slot of examples/scattering/2d_grounded_slab_slot.py."""
    import fem

    solver = fem.ScatteringSolver2D(frequency=frequency, angle=0.0, x_range=(-20.0 * MM, 20.0 * MM),
                                    z_range=(-30.0 * MM, 30.0 * MM), background_material=fem.materials.air)
    solver.add_rectangle(x_range=(0.0, 1.27 * MM), z_range=solver.z_range, background=True,
                         name="dielectric_slab", material=fem.Material(name="grounded slab", epsilon=10.2))
    ground = solver.add_geometry(background=True, name="ground_plane", material=fem.materials.PEC,
                                 shape=fem.shapes.Segment(start=(0.0, solver.z_range[0]), end=(0.0, solver.z_range[1])))
    solver.add_slot(name="ground_slot", geometry=ground, z_range=(-1.0 * MM, 1.0 * MM))
    solver.add_pml(target_reflection=1e-08, thickness=4.0 * MM, direction="x")
    if matched_ports:
        solver.set_matched_ports()
    else:
        solver.add_pml(target_reflection=1e-08, thickness=6.0 * MM, direction="z")
    solver.set_monitors(left=-20.0 * MM, right=20.0 * MM)
    return solver


@pytest.fixture(scope="session")
def grounded_slot():
    """Builder for the grounded-slab slot device: ``grounded_slot(frequency, matched_ports=...)``."""
    return build_grounded_slot
