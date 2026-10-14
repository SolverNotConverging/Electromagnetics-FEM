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
