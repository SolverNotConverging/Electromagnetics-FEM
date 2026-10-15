"""The shared viewer shows mesh, material and E/H views for every family."""
import numpy as np
import pytest

import fem
from fem.common.viewer import ResultViewer, cell_materials


@pytest.fixture(scope="module")
def slab_modes():
    solver = fem.ModeSolver1D(frequency=200e12, x_range=(-1e-6, 1e-6),
                              background_material=fem.Material(name="cladding", epsilon=2.0))
    solver.add_layer(x_range=(-2.5e-7, 2.5e-7), name="core", material=fem.Material(name="core", epsilon=12.0))
    solver.mesh(max_element_size=1e-7)
    return solver.solve(num_modes=1, max_refinements=0, backend="cpu")


@pytest.fixture(scope="module")
def capacitor():
    solver = fem.ElectrostaticSolver(dim=1, x_range=(0.0, 1.0), outer_potential=None)
    solver.add_layer(x_range=(0.5, 1.0), name="dielectric", material=fem.Material(name="dielectric", epsilon=4.0))
    solver.set_potential(geometry="left", potential=0.0)
    solver.set_potential(geometry="right", potential=1.0)
    solver.mesh(max_element_size=0.1)
    return solver.solve(max_refinements=0)


def _views(viewer):
    """Visit every component with each mesh/material combination; return status messages."""
    messages = []
    for index in range(viewer._component.count()):
        viewer._component.setCurrentIndex(index)
        for mesh, material in ((False, False), (True, False), (False, True), (True, True)):
            viewer._mesh.setChecked(mesh)
            viewer._material.setChecked(material)
            messages.append(viewer._status.currentMessage())
    return messages


def test_cell_materials_follow_region_names_and_permittivity(slab_modes):
    index, labels = cell_materials(slab_modes.mesh_data)
    assert len(index) == len(slab_modes.mesh_data.elements)
    assert any(label.startswith("core (εr=12") for label in labels)
    assert any(label.startswith("background (εr=2") for label in labels)


def test_mode_viewer_shows_e_and_h_with_mesh_and_material(slab_modes):
    viewer = ResultViewer(slab_modes)
    components = [viewer._component.itemText(i) for i in range(viewer._component.count())]
    assert {"E", "H"} <= set(components)
    assert {"Ex", "Ey", "Hx", "Hy"} & set(components)
    assert not any(_views(viewer))


def test_electrostatic_viewer_shows_potential_e_d_mesh_and_material(capacitor):
    viewer = ResultViewer(capacitor)
    components = [viewer._component.itemText(i) for i in range(viewer._component.count())]
    assert components == ["potential", "Ex", "E", "Dx", "D"]
    assert not any(_views(viewer))
    assert "no H" in viewer._info.text()


def test_arrows_follow_the_selected_phasor_part():
    pv = pytest.importorskip("pyvista")
    from fem.common.viewer import add_arrows

    points = np.random.default_rng(0).random((50, 3))
    field = [np.full(50, 1.0 + 2.0j), np.zeros(50), np.zeros(50)]
    plotter = pv.Plotter(off_screen=True)
    real = add_arrows(plotter, points, field, "real", label="|E|")
    imag = add_arrows(plotter, points, field, "imag", label="|E|")
    real_arrows, imag_arrows = real, imag
    assert real_arrows.n_points and imag_arrows.n_points
    # Arrows lie along x; the imaginary part is twice as long, so both normalize to 8% of the span.
    np.testing.assert_allclose(np.ptp(real_arrows.points[:, 0]), np.ptp(imag_arrows.points[:, 0]), rtol=0.05)
    np.testing.assert_allclose(real_arrows["|E|"].max(), np.sqrt(5.0))
    plotter.close()


def test_saved_results_keep_mesh_and_material_for_the_viewer(slab_modes, capacitor, tmp_path):
    for name, result in (("modes", slab_modes), ("capacitor", capacitor)):
        loaded = fem.load(result.save(tmp_path / f"{name}.h5"))
        regions = cell_materials(loaded.mesh_data)
        assert regions is not None
        assert regions[1] == cell_materials(result.mesh_data)[1]
