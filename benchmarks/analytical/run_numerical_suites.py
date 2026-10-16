"""Run numerical regression coverage; deselect every FDFD periodic 3D case."""
from pathlib import Path
import json
import os
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=sys.argv[1]
REPO=ROOT if PACKAGE=='FEM' else ROOT.parent/'FDFD'
OUT=ROOT/'outputs/benchmarks/cross_solver_analytical'
sys.path.insert(0,str(REPO/'src'))
os.chdir(REPO)
os.environ['FEM_TEST_CUDA']='0'
os.environ['FDFD_TEST_CUDA']='1'


class Evidence:
    def __init__(self):
        self.results=[]
        self.excluded=[]

    def pytest_collection_modifyitems(self,config,items):
        kept,excluded=[],[]
        for item in items:
            params=getattr(getattr(item,'callspec',None),'params',{})
            skip=PACKAGE=='FDFD' and ('3d' in item.originalname.lower() or params.get('dimension')==3)
            # Viewer contracts do not validate the numerical solver.
            name=item.originalname.lower()
            skip=skip or 'gui' in name.split('_') or any(word in name for word in
                                                        ('viewer','visualiz','palette','colors'))
            (excluded if skip else kept).append(item)
        self.excluded=[item.nodeid for item in excluded]
        items[:]=kept
        config.hook.pytest_deselected(items=excluded)

    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed or report.skipped:
            self.results.append(dict(test=report.nodeid,stage=report.when,outcome=report.outcome,
                                     duration=report.duration,detail=str(report.longrepr) if report.failed else None))

    def pytest_sessionfinish(self,session,exitstatus):
        (OUT/f'{PACKAGE.lower()}_suite.json').write_text(json.dumps(dict(exitstatus=int(exitstatus),
            results=self.results,excluded=self.excluded),indent=2))


if PACKAGE=='FEM':
    files=['tests/waveguide/test_solver_1d.py','tests/waveguide/test_solver_2d.py',
           'tests/waveguide/test_coaxial_benchmark.py','tests/waveguide/test_assembly_impedance.py',
           'tests/waveguide/test_boolean_impedance_frame.py',
           'tests/periodic/test_solver_2d.py','tests/periodic/test_solver_3d.py',
           'tests/periodic/test_periodic_constraints.py','tests/electrostatics/test_fem_solver.py',
           'tests/electrostatics/test_adaptive_solve.py','tests/scattering/test_matched_ports.py',
           'tests/scattering/test_uniform_scattering.py','tests/scattering/test_energy_conservation.py',
           'tests/scattering/test_pml_convergence.py','tests/scattering/test_scattering_validations.py',
           'tests/scattering/test_dielectric_perturbation.py','tests/scattering/test_inserted_pec_scattering.py',
           'tests/scattering/test_mode_pml.py','tests/scattering/test_pml.py',
           'tests/common/test_eigenvalue_regressions.py']
else:
    files=['tests/waveguide','tests/periodic','tests/bands','tests/scattering','tests/tracking',
           'tests/core/test_refined.py','tests/test_regression.py','tests/test_cuda.py',
           'tests/test_cuda_policy.py','tests/test_cuda_residency.py']
raise SystemExit(pytest.main([*files,'--capture=sys','-q','-p','no:cacheprovider'],plugins=[Evidence()]))
