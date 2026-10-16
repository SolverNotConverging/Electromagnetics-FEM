"""Public eigenvalue APIs consistently warn on auto fallback and reject unavailable GPU."""
import numpy as np
import pytest
from scipy.constants import c

from fem.common import gpu, _gpu_arnoldi as hybrid
from fem.common.errors import BackendCapabilityError


FAMILIES = ['waveguide_1d', 'waveguide_2d', 'periodic_te', 'periodic_tm', 'periodic_3d', 'port_modes']


def make_solver(family, mesh):
    import fem
    options = dict(num_modes=1, neff_guess=np.sqrt(.75), max_refinements=0)
    if family == 'waveguide_1d':
        solver = fem.ModeSolver1D(frequency=c, x_range=1.)
        if mesh:
            solver.mesh(resolution=16)
    elif family == 'waveguide_2d':
        solver = fem.ModeSolver2D(frequency=c, x_range=1., y_range=.5)
        if mesh:
            solver.mesh(resolution=(5, 3))
    elif family in ('periodic_te', 'periodic_tm'):
        solver = fem.PeriodicModeSolver2D(frequency=c, x_range=1., z_range=.5,
                                         polarization='TE' if family == 'periodic_te' else 'TM')
        if family == 'periodic_tm':
            options['neff_guess'] = 1.
        if mesh:
            solver.mesh(max_element_size=.25)
    elif family == 'periodic_3d':
        solver = fem.PeriodicModeSolver3D(frequency=c, x_range=1., y_range=.5, z_range=.5)
        if mesh:
            solver.mesh(max_element_size=.3)
    else:
        solver = fem.ScatteringSolver2D(frequency=c, x_range=1., z_range=(-1., 1.), boundary=fem.materials.PEC)
        options['num_elements'] = 16
        return solver, solver.solve_modes, options
    return solver, solver.solve, options


def unavailable(monkeypatch, missing):
    def failure():
        raise ImportError(missing)
    monkeypatch.setattr(gpu, 'is_available', lambda: missing != 'cuDSS')
    monkeypatch.setattr(gpu, 'require_available', failure if missing == 'cuDSS' else lambda: None)
    monkeypatch.setattr(hybrid, 'probe_hybrid_available', failure)
    from fem.common import _gpu_resident_arnoldi as resident
    monkeypatch.setattr(resident, 'probe_resident_available', failure)


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('missing', ['cuDSS', 'kernel compilation'])
def test_explicit_gpu_fails_before_meshing_or_cpu_calculation(monkeypatch, family, missing):
    unavailable(monkeypatch, missing)
    solver, solve, options = make_solver(family, mesh=False)
    with pytest.raises(BackendCapabilityError, match="backend='gpu'"):
        solve(backend='gpu', **options)
    assert solver.mesh_data is None


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('missing', ['cuDSS', 'kernel compilation'])
def test_auto_warns_and_completes_cpu_eigenproblem(monkeypatch, family, missing):
    unavailable(monkeypatch, missing)
    solver, solve, options = make_solver(family, mesh=True)
    with pytest.warns(RuntimeWarning, match='falling back to CPU') as captured:
        result = solve(backend='auto', **options)
    assert len(captured) == 1
    selected = (result.solve_info['backend'] if family == 'port_modes' else
                result.metadata['backend'] if family.startswith('periodic') else result.backend)
    assert selected == 'cpu'
    assert len(result) == 1 and np.isfinite(result[0].neff)
    assert result[0].residual < 1e-8
