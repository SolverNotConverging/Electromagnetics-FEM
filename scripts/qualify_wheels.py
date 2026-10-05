"""Install the complete wheel outside the checkout and exercise solvers and apps."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "fem_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from pathlib import Path
import argparse
import json
import os
import platform
import subprocess
import sys
import tomllib
import tempfile
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
SMOKE = r'''
from importlib.metadata import version
from importlib import import_module
from pathlib import Path
import os
import sys
import numpy as np
import scipy.sparse as sp
from fem_common import Material, materials, shapes

packages = ('fem_common', 'fem_adaptivity', 'periodic_eigensolver',
    'fem_waveguide_modes', 'fem_periodic_modes', 'fem_waveguide_scattering', 'fem_electrostatics')
for name in packages:
    module = import_module(name)
    assert module.__version__ == version('electromagnetics-fem'), name
    assert Path(module.__file__).is_relative_to(Path(sys.prefix)), module.__file__

from periodic_eigensolver import native_backend_available, solve_generalized
assert native_backend_available(), 'The release wheel must activate the native eigensolver.'
result = solve_generalized(sp.diags(np.arange(1., 21.), format='csc'), sp.eye(20, format='csc'),
    sigma=3.1, num_modes=2, backend='cython')
assert np.max(result.residuals) < 1e-8

import importlib.util
assert importlib.util.find_spec('cem_common') is None

from fem_waveguide_modes import ModeSolver1D, ModeSolver2D, load_result
for solver in (ModeSolver1D(frequency=10e9, x_range=.02),
               ModeSolver2D(frequency=10e9, x_range=.02, y_range=.01)):
    solver.mesh(max_element_size=.004)
    result = solver.solve(num_modes=1, neff_guess=.66, max_refinements=0)
    result.save('modes.h5')
    loaded = load_result('modes.h5')
    np.testing.assert_allclose(loaded.neff, result.neff)
    loaded.plot(component='Ey').savefig('modes.png')

from fem_electrostatics import ElectrostaticSolver, load_result
for dimension in (1, 2):
    solver = ElectrostaticSolver(dim=dimension, x_range=1., outer_potential=None)
    solver.set_potential(geometry='left', potential=0.)
    solver.set_potential(geometry='right', potential=1.)
    result = solver.solve(max_refinements=0)
    np.testing.assert_allclose(result.potential, result.coordinates[:, 0], atol=1e-12)
    result.save('static.h5')
    load_result('static.h5').plot().savefig('static.png')

from fem_periodic_modes import PeriodicModeSolver2D, PeriodicModeSolver3D, load_result
periodic_cases = (
    (PeriodicModeSolver2D(frequency=10e9, x_range=.02, z_range=.005),
     dict(max_element_size=.006), .66),
    (PeriodicModeSolver3D(frequency=10e9, x_range=.02, y_range=.01, z_range=.005,
                          background_material=Material(name='fill', epsilon=2.25)),
     dict(max_element_size=.006, wavelength_elements=8), 1.3),
)
for solver, mesh_settings, neff_guess in periodic_cases:
    solver.mesh(**mesh_settings)
    result = solver.solve(num_modes=1, neff_guess=neff_guess, max_refinements=0, eigensolver='dense')
    result.save('periodic.h5')
    np.testing.assert_array_equal(load_result('periodic.h5').neff, result.neff)

from fem_waveguide_scattering import WaveguideScatteringSolver2D, load_result
solver = WaveguideScatteringSolver2D(frequency=299792458., x_range=.5, z_range=(-2., 2.), boundary=materials.PEC)
solver.add_pml(thickness=.5, direction='z')
solver.mesh(max_element_size=.1)
solver.solve_modes(num_modes=1, neff_guess=1., max_refinements=0)
solver.set_incident_mode(0)
result = solver.solve(max_refinements=0)
assert abs(result.S21-1) < 1e-8 and abs(result.S11) < 1e-8
result.save('scattering.h5')
np.testing.assert_array_equal(load_result('scattering.h5').E_total, result.E_total)

print('Installed distributions, native eigensolver, all solver families, physics, and archives: PASS')

import fem
from importlib.metadata import distribution, version
assert Path(fem.__file__).is_relative_to(Path(sys.prefix))
installed = distribution('electromagnetics-fem')
assert not any(requirement.startswith(('fem-common', 'fem-', 'fdfd-', 'periodic-eigensolver')) for requirement in installed.requires)
from fem_common._native import bundled_executable
from fem_waveguide_scattering.viewer import find_viewer_executable
from fem_periodic_modes.persistence import _viewer_candidates
assert find_viewer_executable() == bundled_executable(
    'fem-waveguide-scattering-viewer').resolve()
viewer_name = 'fem-periodic-mode-viewer.exe' if os.name == 'nt' else 'fem-periodic-mode-viewer'
assert _viewer_candidates(viewer_name)[0].resolve() == bundled_executable('fem-periodic-mode-viewer').resolve()
'''

NATIVE_SMOKE = r'''
import os
import subprocess
from pathlib import Path
import sys
from fem_common._native import bundled_executable, bundled_environment

periodic = Path.cwd() / 'periodic.h5'
scattering = Path.cwd() / 'scattering.h5'
for name, arguments in (
    ('transmission-line-calculator', ['--calculate-smoke-test']),
    ('transmission-line-calculator-cli', ['--smoke-test']),
    ('fem-periodic-mode-viewer', ['--smoke-test', str(periodic)]),
    ('fem-waveguide-scattering-viewer', ['--smoke-test', str(scattering)]),
    ('fem-periodic-mode-inspect', [str(periodic)]),
    ('fem-waveguide-scattering-viewer-inspect', [str(scattering)]),
):
    exe = bundled_executable(name)
    env = bundled_environment(exe) or dict(os.environ)
    if os.name == 'nt':
        env['PATH'] = str(exe.parent) + os.pathsep + str(Path(os.environ['SystemRoot']) / 'System32')
        creationflags = subprocess.CREATE_NO_WINDOW
    else:
        env['PATH'] = str(exe.parent) + os.pathsep + '/usr/bin:/bin'
        creationflags = 0
    env['QT_QPA_PLATFORM'] = 'minimal' if os.name == 'nt' else 'offscreen'
    subprocess.run([str(exe), *arguments], env=env, check=True, timeout=90,
                   creationflags=creationflags)
subprocess.run([sys.executable, '-I', '-m', 'fem', 'info'], check=True)
scripts = Path(sys.prefix) / ('Scripts' if os.name == 'nt' else 'bin')
command = scripts / ('transmission-line-calculator-cli.exe'
                     if os.name == 'nt' else 'transmission-line-calculator-cli')
subprocess.run([str(command), '--smoke-test'], check=True)
print('Bundled native applications and installed command entry points: PASS')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', type=Path, default=ROOT/'build/dist')
    parser.add_argument('--fresh', action='store_true', help='Download dependencies into a clean environment.')
    args = parser.parse_args()
    wheels = sorted(args.dist.resolve().glob('*.whl'))
    if sys.platform == 'win32' and platform.machine().lower() in ('amd64', 'x86_64'):
        expected = f'electromagnetics_fem-{VERSION}-cp312-cp312-win_amd64.whl'
    elif sys.platform == 'darwin' and platform.machine() == 'arm64':
        expected = f'electromagnetics_fem-{VERSION}-cp312-cp312-macosx_15_0_arm64.whl'
    else:
        raise SystemExit('Wheel qualification supports Windows x64 and macOS Apple silicon.')
    if len(wheels) != 1 or wheels[0].name != expected:
        raise SystemExit(f'Expected the complete {expected} wheel, found {wheels}.')
    native = wheels[0]
    with zipfile.ZipFile(native) as archive:
        if not any(name.endswith(('.pyd', '.so')) for name in archive.namelist()):
            raise SystemExit('The periodic eigensolver wheel lacks its compiled extension.')
    with tempfile.TemporaryDirectory(prefix='fem-wheel-qualification-') as temporary:
        work = Path(temporary)
        venv.EnvBuilder(with_pip=True, system_site_packages=not args.fresh).create(work/'env')
        python = work/'env'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        for wheel in wheels:
            subprocess.run([str(python), '-I', '-m', 'pip', 'install',
                            *([] if args.fresh else ['--no-index', '--no-deps']), str(wheel)], cwd=work, check=True)
        subprocess.run([str(python), '-I', '-m', 'pip', 'check'], cwd=work, check=True)
        subprocess.run([str(python), '-I', '-c', SMOKE], cwd=work, check=True)
        subprocess.run([str(python), '-I', '-c', NATIVE_SMOKE], cwd=work, check=True)
        source_smoke = (
            f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
            "import periodic_eigensolver; "
            "assert periodic_eigensolver.native_backend_available(); "
            "from fem_common._native import bundled_executable; "
            "assert bundled_executable('fem-periodic-mode-viewer').is_file(); "
            "print('Direct source examples find installed native runtime and kernel: PASS')"
        )
        subprocess.run([str(python), '-I', '-c', source_smoke], cwd=work, check=True)

    print('Wheel qualification passed outside the checkout.')


if __name__=='__main__':main()
