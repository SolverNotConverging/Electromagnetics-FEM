"""Architecture policy checks; these do not claim execution on simulated GPUs."""
import pytest
from fem.common.gpu import _check_architecture

@pytest.mark.parametrize("sm", [60, 61, 70, 72, 75, 80, 86, 89, 90])
def test_cuda12_architecture_policy(sm):
    _check_architecture(sm, 12090)

@pytest.mark.parametrize("sm", [75, 80, 86, 89, 90, 100, 120])
def test_cuda13_architecture_policy(sm):
    _check_architecture(sm, 13030)

@pytest.mark.parametrize("sm", [60, 61, 70, 72])
def test_pascal_volta_require_cuda12(sm):
    with pytest.raises(RuntimeError, match="cuda12"):
        _check_architecture(sm, 13000)

@pytest.mark.parametrize("runtime", [12090, 13030])
def test_maxwell_is_below_cudss_minimum(runtime):
    with pytest.raises(RuntimeError, match="Pascal"):
        _check_architecture(52, runtime)
