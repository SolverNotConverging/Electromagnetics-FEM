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
    assert {"E", "H", "Et", "Ht"} <= set(components)
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


def _ring_modes():
    from fem.waveguide import SampledFields
    from fem.common import MeshSnapshot

    points = np.array([[-2., -2.], [2., -2.], [2., 2.], [-2., 2.],
                       [-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    cells = np.array([[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    samples = points[cells].mean(axis=1)
    fields = SampledFields(samples, {name: np.full(8, value) for name, value in
        dict(Ex=1+2j, Ey=-2+3j, Ez=1e6, Hx=4+5j, Hy=6-7j, Hz=2e6).items()},
        dimension=2, mesh_points=points, mesh_cells=cells,
        metadata={'sample_element_indices': np.arange(8)})
    result = fem.ModeSet([fem.Mode(neff=1., beta=2., fields=fields)], frequency=1e9)
    object.__setattr__(result, '_mesh_snapshot', MeshSnapshot(points, cells, ('x', 'y'), {},
        {'element_tags': np.ones(8, int), 'physical_names': {1: 'background'}}))
    return result


@pytest.mark.parametrize('component,expected', [('Et', (1, -2, 2, 3, np.sqrt(18))),
                                               ('Ht', (4, 6, 5, -7, np.sqrt(126)))])
def test_transverse_quiver_excludes_longitudinal_fields_and_preserves_holes(component, expected):
    from matplotlib.quiver import Quiver

    viewer = ResultViewer(_ring_modes())
    viewer._component.setCurrentText(component)
    assert [viewer._quantity.itemText(i) for i in range(viewer._quantity.count())] == ['real', 'imag', 'magnitude']
    for quantity, pair in [('real', expected[:2]), ('imag', expected[2:4])]:
        viewer._quantity.setCurrentText(quantity)
        assert viewer._status.currentMessage() == ''
        arrows = next(item for item in viewer._canvas.figure.axes[0].collections if isinstance(item, Quiver))
        np.testing.assert_allclose(arrows.U, pair[0])
        np.testing.assert_allclose(arrows.V, pair[1])
        np.testing.assert_allclose(arrows.get_array(), expected[4])
        assert not np.any((abs(arrows.X) < .5) & (abs(arrows.Y) < .5))
    for mesh, material in ((True, False), (False, True), (True, True)):
        viewer._mesh.setChecked(mesh)
        viewer._material.setChecked(material)
        assert viewer._status.currentMessage() == ''
        assert any(isinstance(item, Quiver) for item in viewer._canvas.figure.axes[0].collections)
    viewer._window.close()


def test_transverse_magnitude_uses_one_global_phase_and_loaded_results(tmp_path):
    from matplotlib.quiver import Quiver

    result = _ring_modes()
    loaded = fem.load(result.save(tmp_path/'arrows.h5'))
    viewer = ResultViewer(loaded)
    viewer._component.setCurrentText('Et')
    viewer._quantity.setCurrentText('magnitude')
    arrows = next(item for item in viewer._canvas.figure.axes[0].collections if isinstance(item, Quiver))
    # Ey has the largest phasor amplitude: make it real with one shared phase.
    phase = np.exp(-1j*np.angle(-2+3j))
    np.testing.assert_allclose(arrows.U, ((1+2j)*phase).real)
    np.testing.assert_allclose(arrows.V, ((-2+3j)*phase).real)
    np.testing.assert_allclose(arrows.get_array(), np.sqrt(18))
    viewer._window.close()


def test_periodic_2d_transverse_arrows_keep_y_direction_in_xyz_view():
    from fem.periodic import PeriodicMode, PeriodicModeSet, PeriodicSampledFields
    from fem.common import MeshSnapshot

    points = np.array([[0., 0.], [1., 0.], [0., 1.]])
    fields = PeriodicSampledFields([[.25, .25]], {'Ey': [2j], 'Hx': [3.]}, dimension=2,
        mesh_points=points, mesh_cells=[[0, 1, 2]], sample_element_indices=[0])
    mode = PeriodicMode(neff=1., k0=2., period=1., fields=fields, coefficients=[1.])
    result = PeriodicModeSet([mode], frequency=1e9, period=1., dimension=2)
    object.__setattr__(result, '_mesh_snapshot', MeshSnapshot(points, [[0, 1, 2]], ('x', 'z'), {},
        {'element_tags': [1], 'physical_names': {1: 'background'}}))
    viewer = ResultViewer(result)
    viewer._component.setCurrentText('Et')
    viewer._quantity.setCurrentText('imag')
    assert viewer._status.currentMessage() == ''
    ax = viewer._canvas.figure.axes[0]
    assert ax.get_ylabel() == 'y (m)' and ax.get_zlabel() == 'z (m)'
    segments = np.asarray(ax.collections[0]._segments3d)
    # The shaft is along y, at the physical x/z sample position.
    np.testing.assert_allclose(segments[0, :, 0], .25)
    np.testing.assert_allclose(segments[0, :, 2], .25)
    assert np.ptp(segments[0, :, 1]) > 0
    viewer._mesh.setChecked(True)
    viewer._material.setChecked(True)
    viewer._canvas.draw()
    assert viewer._status.currentMessage() == ''
    viewer._window.close()


@pytest.mark.parametrize('component', ['Et', 'Ht'])
def test_periodic_3d_transverse_arrows_zero_z(monkeypatch, component):
    from types import SimpleNamespace
    from fem.common import viewer as module
    from fem.periodic import PeriodicSampledFields

    fields = PeriodicSampledFields([[.2, .2, .2]],
        {name: [value] for name, value in dict(Ex=1+2j, Ey=3-4j, Ez=1e6,
                                              Hx=5+6j, Hy=7-8j, Hz=2e6).items()},
        dimension=3, mesh_points=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        mesh_cells=[[0, 1, 2, 3]], sample_element_indices=[0])
    captured = []
    monkeypatch.setattr(module, 'add_arrows', lambda *args, **kwargs: captured.append((args, kwargs)))
    plotter = SimpleNamespace(add_axes=lambda: None)
    assert module.populate_3d(plotter, SimpleNamespace(fields=fields), None, component, 'real') == ''
    components = captured[0][0][2]
    np.testing.assert_allclose(components[0], fields.component(f'{component[0]}x'))
    np.testing.assert_allclose(components[1], fields.component(f'{component[0]}y'))
    np.testing.assert_array_equal(components[2], 0)
    assert captured[0][1]['label'] == f'|{component}|'
