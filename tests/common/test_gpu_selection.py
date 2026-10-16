"""Automatic resident selection, including older optional CUDA configurations."""
from inspect import signature
from types import SimpleNamespace
import sys

import pytest

from fem.common import _gpu_arnoldi as arnoldi
from fem.common import _gpu_resident_arnoldi as resident
from fem.common.errors import BackendCapabilityError, SolverError


def test_no_additional_periodic_solve_argument():
    from fem.periodic import PeriodicModeSolver3D
    assert 'gpu_resident' not in signature(PeriodicModeSolver3D.solve).parameters


@pytest.mark.parametrize('has_eigvals,geev', [(False, False), (True, False), (True, True)])
def test_optional_cuda_ritz_availability(monkeypatch, has_eigvals, geev):
    linalg = SimpleNamespace(eigvals=object()) if has_eigvals else SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(linalg=linalg))
    monkeypatch.setitem(sys.modules, 'cupyx.cusolver',
                        SimpleNamespace(check_availability=lambda name: geev))
    assert resident.is_available() == (has_eigvals and geev)


def test_unsupported_cuda_geev_key_never_uses_hybrid(monkeypatch):
    from fem.common import gpu
    from fem.common.linalg import resolve_eigen_backend
    def unavailable(name):
        raise KeyError(name)
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace(linalg=SimpleNamespace(eigvals=object())))
    monkeypatch.setitem(sys.modules, 'cupyx.cusolver', SimpleNamespace(check_availability=unavailable))
    monkeypatch.setattr(gpu, 'is_available', lambda: True)
    monkeypatch.setattr(gpu, 'require_available', lambda: None)
    with pytest.warns(RuntimeWarning, match='resident CUDA eigensolver'):
        assert resolve_eigen_backend('auto') == 'cpu'
    with pytest.raises(BackendCapabilityError, match='CUDA geev'):
        resolve_eigen_backend('gpu')


def test_resident_numerical_failure_does_not_retry_hybrid(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cupy', SimpleNamespace())
    def fail(*args):
        raise SolverError('CUDA numerical failure')
    def forbidden(*args):
        pytest.fail('Numerical failures must not retry with the hybrid implementation')
    monkeypatch.setattr(resident, 'iterate_device', fail)
    monkeypatch.setattr(arnoldi, 'iterate_hybrid', forbidden)
    with pytest.raises(SolverError, match='CUDA numerical failure'):
        arnoldi.iterate(None, 0, 0, 1, 1e-8, None, None, 0)
