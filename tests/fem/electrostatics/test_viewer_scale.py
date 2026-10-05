"""Mesh selection preserves the physical bounds and GUI drawing area."""
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fem_electrostatics.result_api import show_result


def test_mesh_and_fields_use_same_spatial_scale(monkeypatch):
    result = SimpleNamespace(
        coordinates=np.array([[0., 0.], [4e-3, 0.], [4e-3, 1e-3], [0., 1e-3]]),
        elements=np.array([[0, 1, 2], [0, 2, 3]]),
        potential=np.array([0., 1., 1., 0.]),
        element_electric_field=np.ones((2, 2)),
        element_displacement_field=np.ones((2, 2)),
    )
    monkeypatch.setattr(plt, "show", lambda **kwargs: None)
    figure = show_result(result, block=False)
    try:
        figure.canvas.draw()
        reference = figure.axes[1].get_position().bounds
        for index in (5, 1, 0):  # Mesh, Ex, potential.
            figure._cem_selector.set_active(index)
            figure.canvas.draw()
            axis = figure.axes[1]
            np.testing.assert_allclose(axis.get_position().bounds, reference)
            np.testing.assert_allclose(axis.get_xlim(), (0., 4e-3))
            np.testing.assert_allclose(axis.get_ylim(), (0., 1e-3))
            assert axis.get_aspect() == 1
            assert axis.get_xlabel() == "x (m)"
            assert axis.get_ylabel() == "y (m)"
    finally:
        plt.close(figure)
