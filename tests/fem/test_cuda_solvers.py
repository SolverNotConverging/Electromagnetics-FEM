"""Opt-in real-device CPU/CUDA agreement checks."""
import os
import numpy as np
import pytest
from scipy.constants import c

pytestmark = pytest.mark.skipif(os.environ.get("FEM_TEST_CUDA") != "1", reason="Set FEM_TEST_CUDA=1 on a cuDSS host")

@pytest.mark.parametrize("dim", [1, 2])
def test_electrostatic_cuda(dim):
    from fem_electrostatics import ElectrostaticSolver
    solver = ElectrostaticSolver(dim=dim, outer_potential=None)
    solver.set_potential(potential=0., geometry="left")
    solver.set_potential(potential=1., geometry="right")
    solver.mesh(max_element_size=.15)
    cpu = solver.solve(max_refinements=0)
    gpu = solver.solve(max_refinements=0, backend="cuda")
    np.testing.assert_allclose(gpu.potential, cpu.potential, atol=1e-9)
    np.testing.assert_allclose(gpu.energy, cpu.energy, rtol=1e-8)
    assert gpu.residual_norm < 1e-9

@pytest.mark.parametrize("dim", [1, 2])
def test_waveguide_cuda(dim):
    from fem_waveguide_modes.solver_1d import ModeSolver1D
    from fem_waveguide_modes.solver_2d import ModeSolver2D
    solver = ModeSolver1D(frequency=c, x_range=1.) if dim == 1 else ModeSolver2D(frequency=c, x_range=1., y_range=.5)
    solver.mesh(resolution=24 if dim == 1 else (5, 3))
    cpu = solver.solve(num_modes=1, neff_guess=.86, max_refinements=0)
    gpu = solver.solve(num_modes=1, neff_guess=.86, max_refinements=0, backend="cuda")
    np.testing.assert_allclose(gpu[0].neff, cpu[0].neff, rtol=1e-6, atol=1e-7)
    assert gpu[0].residual < 1e-7


@pytest.mark.parametrize("matched_ports", [False, True])
def test_scattering_cuda(matched_ports):
    from fem_common import materials
    from fem_waveguide_scattering import WaveguideScatteringSolver2D
    solver = WaveguideScatteringSolver2D(frequency=c / 1.55e-6,
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
    gpu_modes = solver.solve_modes(backend="cuda", num_modes=1, neff_guess=1.,
        num_elements=32, max_refinements=0)
    np.testing.assert_allclose(gpu_modes[0].neff, cpu_modes[0].neff, rtol=1e-7)
    assert gpu_modes.solve_info["method"] == "cudss-shift-invert"
    assert gpu_modes[0].residual < 1e-8
    # Use identical incident fields to compare the direct solvers independently
    # of eigenvector phase conventions.
    solver.set_modes(cpu_modes)
    solver.set_incident_mode(0, amplitude=1e-13)
    cpu = solver.solve(backend="cpu", max_refinements=0)
    gpu = solver.solve(backend="cuda", max_refinements=0)
    assert gpu.solve_info["method"] == "cudss-direct"
    assert gpu.solve_info["relative_residual"] < 1e-9
    assert np.linalg.norm(cpu.E_scattered) > 0
    np.testing.assert_allclose(gpu.E_scattered, cpu.E_scattered, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose([gpu.S11, gpu.S21], [cpu.S11, cpu.S21], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose([gpu.reflection, gpu.transmission, gpu.power_balance_error],
        [cpu.reflection, cpu.transmission, cpu.power_balance_error], atol=1e-8)


def test_scattering_cuda_sweep():
    from fem_common import materials
    from fem_waveguide_scattering import WaveguideScatteringSolver2D
    solver = WaveguideScatteringSolver2D(frequency=c / 1.55e-6,
        x_range=(0., 1e-6), z_range=(-3e-6, 3e-6), boundary=materials.PEC)
    solver.add_pml(thickness=.8e-6, direction="z")
    result = solver.sweep([c / 1.55e-6, c / 1.5e-6], backend="cuda",
        max_refinements=0, mesh_options={"max_element_size": .25e-6},
        mode_options={"num_modes": 1, "num_elements": 24, "neff_guess": 1.})
    for point in result.results:
        assert point.solve_info["method"] == "cudss-direct"
        assert point.S21 == pytest.approx(1., abs=1e-8)


def test_scattering_cuda_prescribed_pec_values():
    from skfem import MeshTri
    from fem_waveguide_scattering.fem import MaxwellParameters, assemble_mixed_system, solve_prescribed_pec
    mesh = MeshTri.init_tensor(np.linspace(0., 1., 5), np.linspace(0., 1., 5))
    system = assemble_mixed_system(mesh, MaxwellParameters(k0=1.3))
    rng = np.random.default_rng(43)
    exact = rng.standard_normal(system.ndofs) + 1j * rng.standard_normal(system.ndofs)
    prescribed = np.zeros(system.ndofs, dtype=complex)
    prescribed[system.pec_dofs] = exact[system.pec_dofs]
    load = system.matrix @ exact
    cpu = solve_prescribed_pec(system, load, boundary_values=prescribed, backend="cpu")
    gpu = solve_prescribed_pec(system, load, boundary_values=prescribed, backend="cuda")
    np.testing.assert_allclose(gpu.coefficients, exact, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(gpu.coefficients, cpu.coefficients, rtol=1e-9, atol=1e-10)
    assert gpu.solve_info["relative_residual"] < 1e-10


def test_scattering_cuda_oblique_mode_resolution():
    from fem_common import materials
    from fem_waveguide_scattering import WaveguideScatteringSolver2D
    solver = WaveguideScatteringSolver2D(frequency=c / 1.55e-6, angle=10.,
        x_range=(0., 1e-6), z_range=(-3e-6, 3e-6), boundary=materials.PEC)
    solver.solve_modes(backend="cuda", num_modes=1, neff_guess=1.,
        num_elements=32, max_refinements=0)
    solver.set_incident_mode(0)
    assert solver.modes.solve_info["method"] == "cudss-shift-invert"
    assert solver.modes[0].neff == pytest.approx(np.cos(np.deg2rad(10.)), rel=1e-7)
    assert solver.modes[0].residual < 1e-8
