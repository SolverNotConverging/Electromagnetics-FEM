from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import fem.scattering as wf
from fem.scattering.results import ScatteringResult
from fem.scattering.sweep import ScatteringSweepResult


def _result(*, frequency: float = 1.0e9) -> ScatteringResult:
    coordinates = np.asarray(
        ((0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)), dtype=float
    )
    incident = np.ones((3, 4), dtype=np.complex128)
    scattered = np.full((3, 4), 0.1j, dtype=np.complex128)
    return ScatteringResult(
        coordinates=coordinates,
        E_incident=incident,
        E_scattered=scattered,
        H_incident=incident / 377.0,
        H_scattered=scattered / 377.0,
        s_parameters={("left", 0, 0): 0.1j, ("right", 0, 0): 0.9},
        reflected_power=0.01,
        transmitted_power=0.81,
        radiated_power=0.09,
        absorbed_power=0.09,
        incident_power=1.0,
        ndofs=4,
        frequency=frequency,
    )


def test_plot_is_headless(monkeypatch):
    def unexpected_show(*args, **kwargs):
        raise AssertionError("plot() must never open a window")
    monkeypatch.setattr("matplotlib.pyplot.show", unexpected_show)
    figure = _result().plot(component="Ey", quantity="magnitude")
    assert figure.axes[0].get_title() == "total Ey: abs"
    sweep = ScatteringSweepResult(np.asarray((1e9, 2e9)), (_result(), _result(frequency=2e9)))
    figure = sweep.plot(quantity="db")
    assert [line.get_label() for line in figure.axes[0].lines] == ["S11", "S21"]
    assert figure.axes[0].get_xlabel() == "Frequency (Hz)"


def test_viewer_renders_fields_and_s_parameters(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from fem.common.viewer import ResultViewer

    single = ResultViewer(_result())
    assert single._info.text().startswith("S[left, 0←0]")
    assert single._status.currentMessage() == ""

    sweep = ScatteringSweepResult(np.asarray((1e9, 2e9)), (_result(), _result(frequency=2e9)))
    viewer = ResultViewer(sweep)
    assert viewer._case.itemText(0) == "S-parameters"
    assert [viewer._component.itemText(i) for i in range(viewer._component.count())] == ["S11", "S21"]
    viewer._case.setCurrentIndex(2)
    assert viewer._status.currentMessage() == ""


def test_show_routes_to_the_shared_viewer(monkeypatch):
    seen = []
    monkeypatch.setattr("fem.common.viewer.show", lambda result, block=True: seen.append((result, block)) or "window")
    result = _result()
    assert result.show(block=False) == "window"
    assert seen == [(result, False)]


def test_viewer_far_field_tab_uses_the_nf2ff_contour(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from fem.common import viewer as module

    class Field:
        def __init__(self, values):
            self.directivity = self.gain = self.realized_gain = self.power_density = values

    class Case:
        nf2ff = object()
        s_parameters = {}
        def far_field(self, theta):
            return Field(1.0 + np.cos(theta) ** 2)

    for kind in module._FARFIELD:
        for quantity in module._FARFIELD_QUANTITIES:
            figure = module.ResultViewer._farfield_figure(Case(), kind, quantity)
            assert len(figure.axes[0].lines) == 1
    assert set(module._FARFIELD) <= set(module._Source(Case()).components(Case()))
