"""Guard benchmark acceptance of symmetry partners and distinct roots."""
import numpy as np
import pytest

from benchmarks.cuda_eigen_3d.validate_magnitudes import magnitude_error


def test_magnitude_matching_accepts_permuted_forward_backward_partners():
    reference = np.array([.68, .78j, -.78j, 1.02j])
    candidate = np.array([-1.02j, -.68, -.78j, .78j])
    assert magnitude_error(reference, candidate) == pytest.approx(0.)


def test_magnitude_matching_rejects_different_spectrum_and_duplicate_roots():
    assert magnitude_error([1., 2., 3.], [-3., -1., -1.]) > 1e-5
    assert magnitude_error([1., 2.], [1., 2.01]) > 1e-5


def test_magnitude_matching_rejects_missing_and_nonfinite_values():
    assert magnitude_error([1., 2.], [1.]) is None
    assert magnitude_error([1., 2.], [1., np.nan]) is None
