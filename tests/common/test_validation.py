"""Reject malformed numeric inputs before they reach numerical kernels."""

import numpy as np
import pytest

from fem.common import ConfigurationError, MeshSnapshot
from fem.common.contracts import bounds
from fem.common.validation import finite_complex, positive_integer, positive_real
from fem.periodic import PeriodicModeSolver2D, PeriodicModeSolver3D
from fem.waveguide import ModeSolver1D, ModeSolver2D
from fem.scattering import ScatteringSolver2D


@pytest.mark.parametrize(
    "value", [True, np.bool_(False), "4", b"4", [4], np.array(4), None, np.nan, np.inf, 4j]
)
@pytest.mark.parametrize("validate", [positive_real, positive_integer])
def test_scalar_controls_reject_non_numeric_nonfinite_or_nonreal_inputs(value, validate):
    with pytest.raises(ConfigurationError):
        validate(value, "control")


@pytest.mark.parametrize("value", [1, 1.0, np.int64(1), np.float64(1)])
def test_python_and_numpy_scalars_are_accepted(value):
    assert positive_real(value, "control") == 1.0
    assert positive_integer(value, "control") == 1
    assert finite_complex(value, "control") == 1 + 0j


@pytest.mark.parametrize("value", [True, "1j", [1j], np.array(1j), complex(np.nan, 1), None])
def test_complex_controls_reject_invalid_inputs(value):
    with pytest.raises(ConfigurationError):
        finite_complex(value, "target")


@pytest.mark.parametrize(
    "cells",
    [
        [[0.2, 1.8]],
        [[True, False]],
        [["0", "1"]],
        [[0j, 1j]],
        [[0, np.nan]],
        [[0, np.inf]],
        [[-1, 0]],
        [[0, 2]],
        [[0, 2**64]],
        [0, 1],
        [[0], [0, 1]],
    ],
)
def test_mesh_snapshot_rejects_invalid_connectivity(cells):
    with pytest.raises(ConfigurationError):
        MeshSnapshot([[0.0], [1.0]], cells, ("x",), {}, {})


def test_mesh_snapshot_owns_immutable_arrays():
    points = np.array([[0.0], [1.0]])
    cells = np.array([[0.0, 1.0]])
    snapshot = MeshSnapshot(points, cells, ("x",), {}, {})
    points[0, 0] = 2
    cells[0, 0] = 1
    np.testing.assert_array_equal(snapshot.elements, [[0, 1]])
    np.testing.assert_array_equal(snapshot.coordinates, [[0], [1]])
    assert not snapshot.elements.flags.writeable
    assert not snapshot.coordinates.flags.writeable


def test_result_diagnostics_are_explicit_and_include_execution_details():
    from fem.common.contracts import ResultMixin

    class Result(ResultMixin):
        metadata = {"backend": "python", "method": "dense",
                    "adaptive_converged": True, "solver_notes": "unrelated"}

    assert dict(Result().solve_info) == {
        "backend": "python", "method": "dense", "adaptive_converged": True,
    }


@pytest.mark.parametrize("value", ["4", True, [0, "1"], [[0, 1]], [0j, 1j], [0, np.inf]])
def test_bounds_reject_invalid_values(value):
    with pytest.raises(ConfigurationError):
        bounds(value)


@pytest.mark.parametrize(
    "solver, ranges",
    [
        (ModeSolver1D, dict(x_range=0.02)),
        (ModeSolver2D, dict(x_range=0.02, y_range=0.01)),
        (PeriodicModeSolver2D, dict(x_range=0.02, z_range=0.01)),
        (PeriodicModeSolver3D, dict(x_range=0.02, y_range=0.01, z_range=0.01)),
        (ScatteringSolver2D, dict(x_range=0.02, z_range=0.01)),
    ],
)
@pytest.mark.parametrize("frequency", ["1e9", True, np.array(1e9), 1e9 + 0j, np.inf])
def test_em_solver_frequency_contract(solver, ranges, frequency):
    with pytest.raises(ConfigurationError):
        solver(frequency=frequency, **ranges)
