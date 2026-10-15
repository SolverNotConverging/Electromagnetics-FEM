"""Automatic selection and explicit backend behavior."""
import numpy as np
import pytest
from scipy.sparse import diags, eye
from fem_common import gpu
from fem_common.linalg import resolve_backend, generalized_eigenpairs
from fem_common.errors import ConfigurationError

_REAL_IS_AVAILABLE = gpu.is_available

@pytest.mark.parametrize("available, expected", [(True, "gpu"), (False, "cpu")])
def test_auto_selects_runtime(monkeypatch, available, expected):
    monkeypatch.setattr(gpu, "is_available", lambda: available)
    assert resolve_backend() == expected

@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_explicit_backend_does_not_probe(monkeypatch, backend):
    def unexpected():
        raise AssertionError("Explicit backend must not probe or fall back")
    monkeypatch.setattr(gpu, "is_available", unexpected)
    assert resolve_backend(backend) == backend

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
    values, vectors, method = generalized_eigenpairs(
        diags(np.arange(1., 21.)), eye(20), target=3.1, count=2)
    assert method == "sparse-shift-invert"
    np.testing.assert_allclose(np.sort(values.real), [3., 4.], atol=1e-9)


def test_auto_does_not_hide_numerical_failure(monkeypatch):
    from contextlib import contextmanager
    monkeypatch.setattr(gpu, "is_available", lambda: True)
    @contextmanager
    def failed_factorization(*args):
        raise RuntimeError("Numerical factorization failed")
        yield
    monkeypatch.setattr(gpu, "generalized_operator", failed_factorization)
    with pytest.raises(RuntimeError, match="Numerical factorization failed"):
        generalized_eigenpairs(diags(np.arange(1., 21.)), eye(20), target=3.1, count=2)
