"""Regression tests for topology-aware 2D FEM field rendering."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg", force=True)

from fem.waveguide.results import Mode, SampledFields
from fem.common.constants import C_0
from fem.waveguide.solver_2d import ModeSolver2D
from fem.waveguide.visualization import visualize


def _mode_on_square_ring() -> Mode:
    # Eight native triangles cover a square annulus.  A global Delaunay of the
    # element samples would invent triangles across the central square hole.
    points = np.asarray(
        [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (0.8, 0.8),
            (1.2, 0.8),
            (1.2, 1.2),
            (0.8, 1.2),
        ]
    )
    cells = np.asarray(
        [
            (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6),
            (3, 0, 4), (3, 4, 7),
        ],
        dtype=np.int64,
    )
    centroids = points[cells].mean(axis=1)
    sample_points = np.repeat(centroids, 2, axis=0)
    sample_points[0::2, 0] -= 0.025
    sample_points[1::2, 0] += 0.025
    owners = np.repeat(np.arange(cells.shape[0]), 2)
    values = np.repeat(np.linspace(-1.0, 1.0, cells.shape[0]), 2)
    fields = SampledFields(
        sample_points,
        {"Ez": values.astype(np.complex128)},
        dimension=2,
        mesh_points=points,
        mesh_cells=cells,
        material=np.repeat(np.arange(cells.shape[0]) % 2 + 1.0, 2),
        metadata={
            "sampling": "element-quadrature",
            "sample_element_indices": owners,
        },
    )
    return Mode(neff=1.0, beta=1.0, fields=fields, index=1)


def test_quadrature_samples_render_on_native_cells_without_filling_hole() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    mode = _mode_on_square_ring()
    figure, axes = visualize(
        mode,
        component="Ez",
        quantity="real",
        material=False,
        show=False,
    )
    figure.canvas.draw()

    collections = [item for item in axes[0].collections if isinstance(item, PolyCollection)]
    assert collections
    field_collection = collections[0]
    assert len(field_collection.get_paths()) == mode.fields.mesh_cells.shape[0]

    # Every rendered polygon is one supplied mesh triangle; none has a
    # centroid in the intentionally empty central square.
    for path in field_collection.get_paths():
        centroid = path.vertices[:-1].mean(axis=0)
        assert not (0.8 < centroid[0] < 1.2 and 0.8 < centroid[1] < 1.2)

    plt.close(figure)


def test_material_mode_uses_native_interface_edges_without_field_layer() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    mode = _mode_on_square_ring()
    figure, axes = visualize(
        mode,
        component="Ez",
        material=True,
        show=False,
    )
    figure.canvas.draw()

    assert any(isinstance(item, LineCollection) for item in axes[0].collections)
    assert axes[0].get_title().startswith("Mode 0: material")
    plt.close(figure)


def test_2d_solver_records_element_ownership_for_every_sample() -> None:
    solver = ModeSolver2D(frequency=C_0, x_range=1.0, y_range=0.5)
    solver.mesh(resolution=(5, 3))
    fields = solver.solve(max_refinements=0, num_modes=1, neff_guess=np.sqrt(0.75))[0].fields

    owners = np.asarray(fields.metadata["sample_element_indices"], dtype=np.int64)
    assert owners.shape == fields.component("Ey").shape
    assert owners.min() == 0
    assert owners.max() == fields.mesh_cells.shape[0] - 1
    assert np.all(np.bincount(owners) > 0)


def test_viewer_mesh_and_material_toggles(monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from fem.common.viewer import ResultViewer

    # A bare mode carries no mesh snapshot: the viewer says so instead of failing.
    viewer = ResultViewer(_mode_on_square_ring())
    assert viewer._material.isEnabled()
    viewer._material.setChecked(True)
    assert "no" in viewer._status.currentMessage()
    viewer._material.setChecked(False)
    assert viewer._status.currentMessage() == ""
