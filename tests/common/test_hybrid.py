"""Hybrid preference, explicit resident selection and numerical correctness."""
import os
import numpy as np
import pytest
from scipy.sparse import diags, eye

from fem.common import gpu, _gpu_arnoldi as hybrid, _gpu_resident_arnoldi as resident
from fem.common.errors import BackendCapabilityError, SolverError
from fem.common.linalg import resolve_eigen_backend, generalized_eigenpairs, quadratic_eigenpairs


def test_auto_never_tries_resident_when_hybrid_is_missing(monkeypatch):
    monkeypatch.setattr(gpu, 'is_available', lambda: True)
    monkeypatch.setattr(gpu, 'require_available', lambda: None)
    def missing():
        raise RuntimeError('hybrid kernel unavailable')
    def forbidden():
        pytest.fail('Auto must never try resident GPU')
    monkeypatch.setattr(hybrid, 'probe_hybrid_available', missing)
    monkeypatch.setattr(resident, 'probe_resident_available', forbidden)
    with pytest.warns(RuntimeWarning, match='hybrid kernel unavailable'):
        assert resolve_eigen_backend('auto') == 'cpu'
    with pytest.raises(BackendCapabilityError, match="backend='hybrid'"):
        resolve_eigen_backend('hybrid')
    monkeypatch.setattr(resident, 'probe_resident_available', lambda: None)
    assert resolve_eigen_backend('gpu') == 'gpu'


def test_hybrid_numerical_failure_never_retries_another_algorithm(monkeypatch):
    from fem.common.linalg import _iterate
    def failure(*args, **kwargs):
        raise SolverError('hybrid numerical failure')
    def forbidden(*args, **kwargs):
        pytest.fail('No retry is permitted after a numerical failure')
    monkeypatch.setattr(hybrid, 'iterate_hybrid', failure)
    monkeypatch.setattr(hybrid, 'iterate', forbidden)
    operator = gpu._DeviceOperator((20,20), None, None, backend='hybrid')
    with pytest.raises(SolverError, match='hybrid numerical failure'):
        _iterate(operator, 1., 1.1, 2, 1e-9, 20, None, 0)


@pytest.mark.skipif(os.environ.get('FEM_TEST_CUDA') != '1', reason='Set FEM_TEST_CUDA=1')
@pytest.mark.parametrize('backend', ['hybrid','auto'])
@pytest.mark.parametrize('quadratic', [False,True])
def test_hybrid_repeated_roots_without_cuda_ritz(monkeypatch, backend, quadratic):
    def forbidden(*args, **kwargs):
        pytest.fail('Hybrid must not use resident CUDA Ritz facilities')
    monkeypatch.setattr(resident, 'probe_resident_available', forbidden)
    monkeypatch.setattr(resident, 'refined_gpu', forbidden)
    original = hybrid.refined_coefficients
    calls = []
    def extract(hbar, *args):
        assert isinstance(hbar, np.ndarray)
        calls.append(hbar.shape)
        return original(hbar, *args)
    monkeypatch.setattr(hybrid, 'refined_coefficients', extract)
    roots=np.r_[1.,1.,np.arange(2.,20.)]
    if quadratic:
        values,vectors,method=quadratic_eigenpairs(diags(-roots**2),diags(np.zeros(roots.size)),
            eye(roots.size),target=1.1,count=2,backend=backend,tolerance=1e-9)
    else:
        mass=np.ones(roots.size); mass[-2:]=0.
        values,vectors,method=generalized_eigenpairs(diags(roots),diags(mass),target=1.1,
            count=2,backend=backend,tolerance=1e-9)
    np.testing.assert_allclose(values,[1.,1.],atol=1e-8)
    assert np.linalg.matrix_rank(vectors,tol=1e-5)==2
    assert method=='cudss-hybrid-shift-invert'
    assert any(shape[1]>2 for shape in calls)
