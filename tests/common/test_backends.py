"""Automatic selection and explicit backend behavior."""
import numpy as np
import pytest
from scipy.sparse import diags, eye
from fem.common import gpu
from fem.common.linalg import resolve_backend, resolve_eigen_backend, generalized_eigenpairs
from fem.common.errors import BackendCapabilityError, ConfigurationError

_REAL_IS_AVAILABLE = gpu.is_available

@pytest.mark.parametrize("available, expected", [(True, "gpu"), (False, "cpu")])
def test_auto_selects_runtime(monkeypatch, available, expected):
    monkeypatch.setattr(gpu, "is_available", lambda: available)
    if available:
        assert resolve_backend() == expected
    else:
        with pytest.warns(RuntimeWarning, match="falling back to CPU"):
            assert resolve_backend() == expected

def test_explicit_cpu_does_not_probe(monkeypatch):
    def unexpected():
        raise AssertionError("CPU must not probe CUDA")
    monkeypatch.setattr(gpu, "require_available", unexpected)
    assert resolve_backend("cpu") == "cpu"


def test_explicit_gpu_requires_runtime(monkeypatch):
    def missing():
        raise ImportError("cuDSS unavailable")
    monkeypatch.setattr(gpu, "require_available", missing)
    with pytest.raises(BackendCapabilityError, match="backend='gpu'.*cuDSS unavailable"):
        resolve_backend("gpu")


@pytest.mark.parametrize("backend", ["cuda", "python", "cython", None, "GPU"])
def test_removed_backend_names(backend):
    with pytest.raises(ConfigurationError):
        resolve_backend(backend)

def test_unavailable_runtime_falls_back_to_scipy(monkeypatch):
    def missing(*args):
        raise ImportError("cuDSS unavailable")
    monkeypatch.setattr(gpu, "solve_sparse", missing)
    monkeypatch.setattr(gpu, "is_available", _REAL_IS_AVAILABLE)
    assert not gpu.is_available()
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        values, vectors, method = generalized_eigenpairs(
            diags(np.arange(1., 21.)), eye(20), target=3.1, count=2)
    assert method == "sparse-shift-invert"
    np.testing.assert_allclose(np.sort(values.real), [3., 4.], atol=1e-9)


def test_auto_does_not_hide_numerical_failure(monkeypatch):
    from contextlib import contextmanager
    monkeypatch.setattr(gpu, "is_available", lambda: True)
    from fem.common import _gpu_resident_arnoldi as resident
    monkeypatch.setattr(resident, "probe_resident_available", lambda: None)
    @contextmanager
    def failed_factorization(*args):
        raise RuntimeError("Numerical factorization failed")
        yield
    monkeypatch.setattr(gpu, "generalized_operator", failed_factorization)
    with pytest.raises(RuntimeError, match="Numerical factorization failed"):
        generalized_eigenpairs(diags(np.arange(1., 21.)), eye(20), target=3.1, count=2)


@pytest.mark.parametrize('failure', ['CUDA geev unavailable', 'CUDA SVD unavailable', 'NVRTC compilation failed'])
def test_missing_resident_step_warns_for_auto_and_errors_for_gpu(monkeypatch, failure):
    from fem.common import _gpu_resident_arnoldi as resident
    monkeypatch.setattr(gpu, 'is_available', lambda: True)
    monkeypatch.setattr(gpu, 'require_available', lambda: None)
    def unavailable():
        raise RuntimeError(failure)
    monkeypatch.setattr(resident, 'probe_resident_available', unavailable)
    with pytest.warns(RuntimeWarning, match=failure):
        assert resolve_eigen_backend('auto') == 'cpu'
    with pytest.raises(BackendCapabilityError, match=failure):
        resolve_eigen_backend('gpu')


def test_direct_gpu_does_not_require_eigenvalue_libraries(monkeypatch):
    from fem.common import _gpu_resident_arnoldi as resident
    monkeypatch.setattr(gpu, 'require_available', lambda: None)
    def forbidden():
        pytest.fail('Direct solves must not require CUDA geev/SVD')
    monkeypatch.setattr(resident, 'probe_resident_available', forbidden)
    assert resolve_backend('gpu') == 'gpu'
