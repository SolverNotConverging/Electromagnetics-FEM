"""Opt-in real-device CPU/CUDA agreement checks."""
import os
import numpy as np
import pytest
from scipy.constants import c

pytestmark = pytest.mark.skipif(os.environ.get("FEM_TEST_CUDA") != "1", reason="Set FEM_TEST_CUDA=1 on a cuDSS host")

@pytest.fixture
def cuda_ritz(monkeypatch):
    """Prove family GPU solves invoke the shared CUDA dense Ritz/SVD core."""
    import cupy as cp
    from fem.common import _gpu_resident_arnoldi as resident
    from fem.common import _gpu_arnoldi as arnoldi
    if not resident.is_available():
        pytest.skip('CUDA geev unavailable')
    original = resident.refined_gpu
    calls = []

    def track(hbar, shift, count):
        assert isinstance(hbar, cp.ndarray)
        result = original(hbar, shift, count)
        assert all(isinstance(array, cp.ndarray) for array in result)
        calls.append(hbar.shape)
        return result

    def forbidden(*args, **kwargs):
        pytest.fail('A supported resident solve must not invoke the older hybrid iteration')
    monkeypatch.setattr(resident, 'refined_gpu', track)
    monkeypatch.setattr(arnoldi, 'iterate_hybrid', forbidden)
    yield calls
    assert any(shape[1] > 2 for shape in calls), 'The numerical solve did not invoke CUDA Ritz extraction'


@pytest.mark.parametrize("dim", [1, 2])
def test_electrostatic_cuda(dim):
    from fem.electrostatics import ElectrostaticSolver
    solver = ElectrostaticSolver(dim=dim, outer_potential=None)
    solver.set_potential(potential=0., geometry="left")
    solver.set_potential(potential=1., geometry="right")
    solver.mesh(max_element_size=.15)
    cpu = solver.solve(max_refinements=0, backend="cpu")
    gpu = solver.solve(max_refinements=0, backend="gpu")
    np.testing.assert_allclose(gpu.potential, cpu.potential, atol=1e-9)
    np.testing.assert_allclose(gpu.energy, cpu.energy, rtol=1e-8)
    assert gpu.residual_norm < 1e-9

@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("backend", ["gpu", "auto"])
def test_waveguide_cuda(dim, backend, cuda_ritz):
    from fem.waveguide.solver_1d import ModeSolver1D
    from fem.waveguide.solver_2d import ModeSolver2D
    solver = ModeSolver1D(frequency=c, x_range=1.) if dim == 1 else ModeSolver2D(frequency=c, x_range=1., y_range=.5)
    solver.mesh(resolution=24 if dim == 1 else (5, 3))
    cpu = solver.solve(num_modes=1, neff_guess=.86, max_refinements=0, backend="cpu")
    gpu = solver.solve(num_modes=1, neff_guess=.86, max_refinements=0, backend=backend)
    np.testing.assert_allclose(gpu[0].neff, cpu[0].neff, rtol=1e-6, atol=1e-7)
    assert gpu[0].residual < 1e-7


@pytest.mark.parametrize("matched_ports", [False, True])
@pytest.mark.parametrize("backend", ["gpu", "auto"])
def test_scattering_cuda(matched_ports, backend, cuda_ritz):
    from fem.common import materials
    from fem.scattering import ScatteringSolver2D
    solver = ScatteringSolver2D(frequency=c / 1.55e-6,
        x_range=(0., 1e-6), z_range=(-3e-6, 3e-6), boundary=materials.PEC)
    solver.add_rectangle(x_range=solver.x_range, z_range=(-.3e-6, .3e-6),
        material=materials.Material(epsilon=1.1))
    if matched_ports:
        solver.set_matched_ports()
    else:
        solver.add_pml(thickness=.8e-6, direction="z", target_reflection=1e-7)
    solver.mesh(max_element_size=.25e-6, wavelength_elements=6)
    cpu_modes = solver.solve_modes(backend="cpu", num_modes=1, neff_guess=1.,
        num_elements=32, max_refinements=0)
    gpu_modes = solver.solve_modes(backend=backend, num_modes=1, neff_guess=1.,
        num_elements=32, max_refinements=0)
    np.testing.assert_allclose(gpu_modes[0].neff, cpu_modes[0].neff, rtol=1e-7)
    assert gpu_modes.solve_info["method"] == "cudss-shift-invert"
    assert gpu_modes[0].residual < 1e-8
    # Use identical incident fields to compare the direct solvers independently
    # of eigenvector phase conventions.
    solver.set_modes(cpu_modes)
    solver.set_incident_mode(0, amplitude=1e-13)
    cpu = solver.solve(backend="cpu", max_refinements=0)
    gpu = solver.solve(backend=backend, max_refinements=0)
    assert gpu.solve_info["method"] == "cudss-direct"
    assert gpu.solve_info["relative_residual"] < 1e-9
    assert np.linalg.norm(cpu.E_scattered) > 0
    np.testing.assert_allclose(gpu.E_scattered, cpu.E_scattered, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose([gpu.S11, gpu.S21], [cpu.S11, cpu.S21], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose([gpu.reflection, gpu.transmission, gpu.power_balance_error],
        [cpu.reflection, cpu.transmission, cpu.power_balance_error], atol=1e-8)


def test_scattering_cuda_sweep(cuda_ritz):
    from fem.common import materials
    from fem.scattering import ScatteringSolver2D
    solver = ScatteringSolver2D(frequency=c / 1.55e-6,
        x_range=(0., 1e-6), z_range=(-3e-6, 3e-6), boundary=materials.PEC)
    solver.add_pml(thickness=.8e-6, direction="z")
    result = solver.sweep([c / 1.55e-6, c / 1.5e-6], backend="gpu",
        max_refinements=0, mesh_options={"max_element_size": .25e-6},
        mode_options={"num_modes": 1, "num_elements": 24, "neff_guess": 1.})
    for point in result.results:
        assert point.solve_info["method"] == "cudss-direct"
        assert point.S21 == pytest.approx(1., abs=1e-8)


def test_scattering_cuda_prescribed_pec_values():
    from skfem import MeshTri
    from fem.scattering.fem import MaxwellParameters, assemble_mixed_system, solve_prescribed_pec
    mesh = MeshTri.init_tensor(np.linspace(0., 1., 5), np.linspace(0., 1., 5))
    system = assemble_mixed_system(mesh, MaxwellParameters(k0=1.3))
    rng = np.random.default_rng(43)
    exact = rng.standard_normal(system.ndofs) + 1j * rng.standard_normal(system.ndofs)
    prescribed = np.zeros(system.ndofs, dtype=complex)
    prescribed[system.pec_dofs] = exact[system.pec_dofs]
    load = system.matrix @ exact
    cpu = solve_prescribed_pec(system, load, boundary_values=prescribed, backend="cpu")
    gpu = solve_prescribed_pec(system, load, boundary_values=prescribed, backend="gpu")
    np.testing.assert_allclose(gpu.coefficients, exact, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(gpu.coefficients, cpu.coefficients, rtol=1e-9, atol=1e-10)
    assert gpu.solve_info["relative_residual"] < 1e-10


def test_scattering_cuda_oblique_mode_resolution(cuda_ritz):
    from fem.common import materials
    from fem.scattering import ScatteringSolver2D
    solver = ScatteringSolver2D(frequency=c / 1.55e-6, angle=10.,
        x_range=(0., 1e-6), z_range=(-3e-6, 3e-6), boundary=materials.PEC)
    solver.solve_modes(backend="gpu", num_modes=1, neff_guess=1.,
        num_elements=32, max_refinements=0)
    solver.set_incident_mode(0)
    assert solver.modes.solve_info["method"] == "cudss-shift-invert"
    assert solver.modes[0].neff == pytest.approx(np.cos(np.deg2rad(10.)), rel=1e-7)
    assert solver.modes[0].residual < 1e-8


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("backend", ["gpu", "auto"])
def test_periodic_cuda(dim, backend, cuda_ritz):
    from fem.common import Material
    from fem.periodic import PeriodicModeSolver2D, PeriodicModeSolver3D

    options = dict(frequency=10e9, x_range=.02, z_range=.005,
                   background_material=Material(epsilon=2.25))
    solver = (PeriodicModeSolver2D(**options, polarization="TE") if dim == 2
              else PeriodicModeSolver3D(**options, y_range=.01))
    solver.mesh(max_element_size=.006, wavelength_elements=8)
    cpu = solver.solve(num_modes=1, neff_guess=1.3, max_refinements=0,
                       eigensolver="dense", backend="cpu")
    gpu = solver.solve(num_modes=1, neff_guess=1.3, max_refinements=0,
                       eigensolver="sparse", backend=backend)
    np.testing.assert_allclose(gpu.neff, cpu.neff, rtol=1e-6, atol=1e-8)
    assert gpu.metadata["backend"] == "gpu"
    assert gpu[0].residual < 1e-8


def test_default_backend_uses_available_gpu(cuda_ritz):
    from fem.common.linalg import resolve_backend
    from fem.waveguide import ModeSolver1D
    assert resolve_backend() == "gpu"
    solver = ModeSolver1D(frequency=c, x_range=1.)
    solver.mesh(resolution=24)
    result = solver.solve(num_modes=1, neff_guess=.86, max_refinements=0)
    assert result.backend == "gpu"
    assert all(method == "cudss-shift-invert" for method in result.metadata["methods"].values())


@pytest.mark.parametrize('case', ['pec_lossy', 'pec_sibc', 'pec_sibc_lossy'])
def test_waveguide_material_combinations_use_resident_core(case, cuda_ritz):
    from benchmarks.gpu_waveguide_2d import make_solver, solve
    solver, target, _ = make_solver(case, 8, 1)
    reference = solve(solver, target, 'cpu', 1)
    actual = solve(solver, target, 'gpu', 1)
    np.testing.assert_allclose(actual.neff, reference.neff, rtol=1e-7, atol=1e-8)
    assert actual[0].residual < 1e-8
    assert actual[0].divergence_residual < 1e-6


@pytest.mark.parametrize('backend', ['gpu', 'auto'])
def test_periodic_tm_cuda(backend, cuda_ritz):
    from fem import Material, PeriodicModeSolver2D
    solver = PeriodicModeSolver2D(frequency=10e9, x_range=.004, z_range=.005,
        background_material=Material(epsilon=2.25), polarization='TM')
    solver.mesh(max_element_size=.001)
    expected = solver.solve(backend='cpu', num_modes=1, neff_guess=1.5,
                            max_refinements=0, eigensolver='dense')
    result = solver.solve(backend=backend, num_modes=1, neff_guess=1.5, max_refinements=0)
    assert result.metadata['backend'] == 'gpu'
    np.testing.assert_allclose(result.neff, expected.neff, atol=1e-8, rtol=1e-7)
