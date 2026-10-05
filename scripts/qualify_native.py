"""Check Python-written archives with native inspectors and offscreen viewers."""

# Run directly from the checkout without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "fem_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from fem_common import Material, SurfaceImpedance, materials, shapes
import argparse
import os
from pathlib import Path
import subprocess
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,default=ROOT/'build/native-release')
    parser.add_argument('--output',type=Path,default=ROOT/'build/native-qualification')
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    def run(executable,*arguments,cwd=None):
        name=executable+('.exe' if os.name=='nt' else '')
        matches=[path for path in args.build.resolve().rglob(name) if path.is_file()]
        if len(matches)!=1:raise RuntimeError(f'Expected one {name} in {args.build}; found {len(matches)}.')
        from fem_periodic_modes.persistence import _build_runtime_environment
        environment=dict(_build_runtime_environment(matches[0]) or os.environ, QT_QPA_PLATFORM='minimal' if os.name=='nt' else 'offscreen')
        subprocess.run([str(matches[0]),*map(str,arguments)],cwd=cwd,env=environment,check=True,timeout=45,
            **({'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}))
    from fem_periodic_modes import PeriodicModeSolver2D,PeriodicModeSolver3D,PeriodicSweepResult
    periodic=[]
    for cls,ranges in ((PeriodicModeSolver2D,dict(x_range=.02,z_range=.005)),
                       (PeriodicModeSolver3D,dict(x_range=.02,y_range=.01,z_range=.005))):
        solver=cls(frequency=10e9,**ranges)
        solver.mesh(max_element_size=.006)
        result=solver.solve(num_modes=1,neff_guess=.66,max_refinements=0,eigensolver='dense')
        periodic.append(result)
        path=output/f'periodic-{result.dimension}d.h5'
        result.save(path)
        run('fem-periodic-mode-inspect',path,'0','0','--coefficients')
        run('fem-periodic-mode-viewer','--smoke-test',path)
    path=output/'periodic-sweep.h5'
    PeriodicSweepResult.from_results(periodic).save(path)
    run('fem-periodic-mode-inspect',path,'1','0','--coefficients')
    from fem_waveguide_scattering import WaveguideScatteringSolver2D
    import numpy as np
    solver=WaveguideScatteringSolver2D(frequency=299792458.0, x_range=0.5, z_range=(-2.0, 2.0), boundary=materials.PEC)
    solver.add_pml(thickness=.5,direction='z')
    solver.mesh(max_element_size=.1)
    result=solver.solve(max_refinements=0)
    path=output/'scattering.h5'
    result.save(path)
    run('fem-waveguide-scattering-viewer-inspect',path)
    run('fem-waveguide-scattering-viewer','--smoke-test',path)
    sweep=solver.sweep(np.asarray((299792458.,310e6)),max_refinements=0,
        mesh_options={'max_element_size':.1},mode_options={'num_modes':1,'max_refinements':0})
    path=output/'scattering-sweep.h5'
    sweep.save(path)
    run('fem-waveguide-scattering-viewer-inspect',path,'1')
    run('fem-waveguide-scattering-viewer','--smoke-test',path)
    # Launch with no file argument from isolated solver-output layouts.
    auto_root = output / 'auto-load'
    for family, executable, sample in (
        ('fem_periodic_modes', 'fem-periodic-mode-viewer', 'periodic-2d.h5'),
        ('fem_waveguide_scattering', 'fem-waveguide-scattering-viewer', 'scattering.h5'),
    ):
        case = auto_root / family / 'outputs' / '2d_example'
        case.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / sample, case / 'results.h5')
        run(executable, '--smoke-test', cwd=auto_root / family)
    print('Native readers, recursive output discovery, and offscreen viewers: PASS.')


if __name__=='__main__':main()
