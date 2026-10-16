"""Real CUDA checks for resident Ritz extraction and complete FEM numerics."""
import os

import numpy as np
import pytest
from scipy.sparse import diags, eye


@pytest.fixture(scope='module')
def cuda():
    if not os.environ.get('CUDSS_LIBRARY'):
        pytest.skip('Set CUDSS_LIBRARY to execute real CUDA checks')
    cp = pytest.importorskip('cupy')
    from cupyx.cusolver import check_availability
    if cp.cuda.runtime.getDeviceCount() == 0 or not check_availability('geev'):
        pytest.skip('CUDA geev is unavailable')
    return cp


@pytest.mark.parametrize('repeated', [False, True])
def test_resident_ritz_and_vectors_stay_on_device(cuda, monkeypatch, repeated):
    from fem.common.gpu import generalized_operator
    from fem.common._gpu_resident_arnoldi import iterate_device
    roots = np.r_[1., 1. if repeated else 1.5, np.arange(2., 31.)]
    target = 1.1
    downloads = []
    original = cuda.asnumpy

    def track(array, *args, **kwargs):
        downloads.append((array.shape, array.dtype))
        return original(array, *args, **kwargs)

    with generalized_operator(diags(roots), eye(len(roots)), target) as operator:
        with monkeypatch.context() as patch:
            patch.setattr(cuda, 'asnumpy', track)
            values, vectors, residuals, execution = iterate_device(
                operator, target, target, 2, 1e-9, len(roots), None, 0)
    assert isinstance(values, cuda.ndarray) and isinstance(vectors, cuda.ndarray)
    assert downloads and all(shape == (2,) and dtype == np.dtype(bool) for shape,dtype in downloads)
    assert execution['ritz_backend'] == 'cusolver-geev-svd'
    np.testing.assert_allclose(cuda.asnumpy(values), roots[:2], atol=1e-8)
    assert float(cuda.max(residuals).get()) < 1e-9
    assert np.linalg.matrix_rank(cuda.asnumpy(vectors), tol=1e-5) == 2


@pytest.mark.parametrize('lossy', [False, True])
@pytest.mark.parametrize('direction', ['forward', 'backward'])
def test_resident_periodic_fields_estimator_and_cache(cuda, monkeypatch, tmp_path, lossy, direction):
    import fem
    mm = 1e-3
    solver = fem.PeriodicModeSolver3D(frequency=12e9, x_range=22.86*mm,
        y_range=10.16*mm, z_range=8*mm,
        background_material=fem.Material(epsilon=2-.02j if lossy else 1.))
    for name, xr in [('left', (0., 4*mm)), ('right', (18.86*mm, 22.86*mm))]:
        solver.add_box(name=name, x_range=xr, y_range=(0., 10.16*mm),
                       z_range=(3.6*mm, 4.4*mm), material=fem.materials.PEC)
    solver.mesh(max_element_size=3*mm, material_aware=False)
    target = 1.2-.01j if lossy else .7
    options = dict(num_modes=1, neff_guess=target if direction == 'forward' else -target,
                   direction=direction,
                   eigensolver='sparse', ncv=80, eigensolver_tolerance=1e-8,
                   max_refinements=0)
    reference = solver.solve(backend='cpu', **options)
    actual = solver.solve(backend='gpu', **options)
    np.testing.assert_allclose(actual.neff, reference.neff, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(actual[0].coefficients, reference[0].coefficients, rtol=1e-7, atol=1e-8)
    for component in ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'):
        np.testing.assert_allclose(actual[0].fields.component(component),
                                   reference[0].fields.component(component), rtol=1e-7, atol=1e-8)
    assert actual[0].power == pytest.approx(reference[0].power, rel=1e-8)
    assert actual[0].polarization == reference[0].polarization
    assert actual.metadata['adaptive_residual'] == pytest.approx(
        reference.metadata['adaptive_residual'], rel=1e-10)
    assert actual[0].residual < 1e-8 and actual[0].gauss_residual < 1e-6
    assert actual.metadata['gpu_resident'] is True
    assert actual.metadata['gpu_execution']['ritz_backend'] == 'cusolver-geev-svd'
    cached = solver._gpu_resident_session
    from fem.common import gpu
    monkeypatch.setattr(gpu, 'is_available', lambda: True)
    again = solver.solve(backend='auto', **options)
    assert again.metadata['backend'] == 'hybrid'
    assert again.metadata['gpu_resident'] is False and again.metadata['gpu_postprocessing'] is True
    assert again.metadata['gpu_execution']['ritz_backend'] == 'scipy-eigvals-svd'
    np.testing.assert_allclose(again.neff, reference.neff, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(again[0].coefficients, reference[0].coefficients, rtol=1e-7, atol=1e-8)
    path = tmp_path/'hybrid.h5'
    again.save(path)
    loaded = fem.load(path)
    assert loaded.metadata['backend'] == 'hybrid'
    np.testing.assert_allclose(loaded.neff, again.neff)
    assert solver._gpu_resident_session is cached
    assert again.metadata['gpu_execution']['prepare_seconds'] == 0
    solver.set_boundary(material=fem.materials.PMC)
    assert solver._gpu_resident_session is None and solver._gpu_resident_estimate is None


def test_resident_pml_leaky_modes_and_saved_fields(cuda, tmp_path):
    import fem
    mm = 1e-3
    solver = fem.PeriodicModeSolver3D(frequency=20e9, x_range=(-9*mm, 9*mm),
        y_range=(-8*mm, 8*mm), z_range=(0., 10*mm))
    solver.add_box(name='rod', x_range=(-3*mm, 3*mm), y_range=(-2*mm, 2*mm),
                   z_range=(0., 10*mm), material=fem.Material(epsilon=6.))
    solver.add_box(name='patch', x_range=(-2*mm, 2*mm), y_range=(2*mm, 2.2*mm),
                   z_range=(3.5*mm, 6.5*mm), material=fem.materials.PEC)
    solver.add_pml(thickness=3*mm, direction='all')
    solver.mesh(max_element_size=4*mm, wavelength_elements=4)
    options = dict(num_modes=2, neff_guess=.5-.02j, ncv=96,
                   eigensolver_tolerance=1e-8, max_refinements=0)
    reference = solver.solve(backend='cpu', **options)
    result = solver.solve(backend='gpu', **options)
    np.testing.assert_allclose(result.neff, reference.neff, rtol=1e-8, atol=1e-9)
    assert result.metadata['adaptive_residual'] == pytest.approx(
        reference.metadata['adaptive_residual'], rel=1e-9)
    for actual, expected in zip(result, reference):
        assert actual.pml_fraction == pytest.approx(expected.pml_fraction, rel=1e-8)
        np.testing.assert_allclose(actual.coefficients, expected.coefficients, rtol=1e-7, atol=1e-8)
    path = tmp_path/'resident.h5'
    result.save(path)
    loaded = fem.load(path)
    np.testing.assert_array_equal(loaded.neff, result.neff)
    np.testing.assert_array_equal(loaded[0].fields.component('Ey'), result[0].fields.component('Ey'))


def test_resident_adaptive_remesh_rebuilds_device_geometry(cuda):
    import fem
    solver = fem.PeriodicModeSolver3D(frequency=12e9, x_range=.02286,
                                    y_range=.01016, z_range=.008)
    solver.mesh(max_element_size=.004)
    options = dict(num_modes=1, neff_guess=.84, ncv=80, eigensolver_tolerance=1e-8)
    solver.solve(backend='gpu', max_refinements=0, **options)
    old = solver._gpu_resident_session
    result = solver.solve(backend='gpu', max_refinements=1,
                          adaptive_tolerance=1e-12, **options)
    assert len(result.metadata['adaptive_history']) == 2
    assert solver._gpu_resident_session is not old
    assert solver._gpu_resident_session.system is solver.system
    assert result.metadata['gpu_execution']['prepare_seconds'] > 0


def test_shared_quadratic_exports_only_physical_vectors(cuda, monkeypatch):
    from fem.common.linalg import quadratic_eigenpairs
    # Readiness checks precede the numerical residency/export boundary.
    from fem.common import gpu, _gpu_resident_arnoldi as resident
    from fem.common.linalg import resolve_eigen_backend
    resolve_eigen_backend('gpu')
    monkeypatch.setattr(gpu, 'require_available', lambda: None)
    monkeypatch.setattr(resident, 'probe_resident_available', lambda: None)
    n = 20
    roots = np.arange(1., n+1)
    downloads = []
    original = cuda.asnumpy

    def track(array, *args, **kwargs):
        downloads.append((array.shape, array.dtype))
        return original(array, *args, **kwargs)

    monkeypatch.setattr(cuda, 'asnumpy', track)
    values, vectors, _ = quadratic_eigenpairs(
        diags(-roots**2), diags(np.zeros(n)), eye(n), target=3.1,
        count=2, tolerance=1e-9, backend='gpu')
    np.testing.assert_allclose(np.sort(values.real), [3., 4.], atol=1e-8)
    assert vectors.shape == (n, 2)
    assert downloads[-2:][0][0] == (2,)
    assert downloads[-1][0] == (n, 2)
    assert all(shape == (2,) and dtype == np.dtype(bool)
               for shape, dtype in downloads[:-2])
