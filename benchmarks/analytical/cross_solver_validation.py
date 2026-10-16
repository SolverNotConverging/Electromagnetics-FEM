"""Independent analytical validation of sibling FEM/FDFD repositories.

Run with FEM's Python and the sibling FDFD source available. This exercises
public numerical APIs; FDFD periodic 3D is deliberately excluded. Results
are checkpointed so a failed case never hides already completed evidence.
"""
from pathlib import Path
import json
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT.parent / 'FDFD' / 'src'))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
from scipy.constants import c, epsilon_0, mu_0
from scipy.special import h2vp, hankel2, jv, jvp
from scipy.optimize import brentq
import fem
import fdfd

OUT = ROOT / 'outputs/benchmarks/cross_solver_analytical'
ROWS = []
FAILURES = []


def record(case, package, backend, level, actual, exact, tolerance, *, metric='relative', **extra):
    actual, exact = np.asarray(actual, complex), np.asarray(exact, complex)
    error = float(np.linalg.norm(actual-exact))
    if metric == 'relative':
        error /= max(float(np.linalg.norm(exact)), 1e-300)
    row = dict(case=case, package=package, backend=backend, level=level,
               actual=[[float(v.real), float(v.imag)] for v in actual.ravel()],
               exact=[[float(v.real), float(v.imag)] for v in exact.ravel()],
               error=error, metric=metric, tolerance=tolerance,
               passed=bool(np.isfinite(error) and error <= tolerance), **extra)
    ROWS.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(dict(rows=ROWS, failures=FAILURES), indent=2))
    print(f"{case}: {package} {backend} level={level}: {error:.4g} <= {tolerance:g} {row['passed']}", flush=True)


def run(name, function):
    try:
        function()
    except Exception:
        FAILURES.append(dict(group=name, traceback=traceback.format_exc()))
        print(FAILURES[-1]['traceback'], flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT/'results.json').write_text(json.dumps(dict(rows=ROWS, failures=FAILURES), indent=2))


def verify_summary(rows):
    """Gate physics, convergence, CPU/GPU agreement and requested coverage."""
    groups = {}
    checks = []
    for row in rows:
        groups.setdefault((row['case'], row['package'], row['level']), {})[row['backend']] = row
        for key in ('residual', 'divergence_residual', 'gauss_residual'):
            if key in row:
                limit = 1e-8 if key == 'residual' else 1e-6
                checks.append(dict(check=key,case=row['case'],passed=bool(row[key]<=limit),error=row[key]))
        if row['case'].startswith(('waveguide','periodic')):
            actual, exact = complex(*row['actual'][0]), complex(*row['exact'][0])
            if abs(exact.imag)>1e-12:
                error = abs(actual.imag/exact.imag-1)
                checks.append(dict(check='attenuation',case=row['case'],passed=error<.01,error=error))
    for (case,package,level), backends in groups.items():
        if set(backends) != {'cpu','gpu'}:
            checks.append(dict(check='both_backends',case=case,passed=False))
            continue
        arrays = [np.asarray([complex(*v) for v in backends[b]['actual']]) for b in ('cpu','gpu')]
        error = float(np.linalg.norm(arrays[0]-arrays[1])/max(np.linalg.norm(arrays[0]),1e-300))
        checks.append(dict(check='cpu_gpu_agreement',case=case,package=package,level=level,
                           passed=error<1e-6,error=error))
    for package,prefixes in [
        ('FEM',('electrostatic_1d','electrostatic_2d','waveguide_1d','waveguide_2d','periodic_2d',
                'periodic_3d','scattering_slab','sibc_2d','dielectric_slab')),
        ('FDFD',('waveguide_1d','waveguide_2d','periodic_2d','bands_2d','scattering_cylinder',
                 'tracking_1d','tracking_2d','sibc_2d','dielectric_slab'))]:
        for prefix in prefixes:
            checks.append(dict(check='family_coverage',case=f'{package} {prefix}',
                               passed=any(r['package']==package and r['case'].startswith(prefix) for r in rows)))
    for case,package in {(r['case'],r['package']) for r in rows if r['backend']=='cpu'}:
        levels=sorted([r for r in rows if r['case']==case and r['package']==package and r['backend']=='cpu'],
                      key=lambda r:r['level'])
        if len(levels)>1 and levels[0]['error']>1e-8 and not case.startswith('sibc'):
            checks.append(dict(check='mesh_or_order_convergence',case=case,package=package,
                               passed=levels[-1]['error']<levels[0]['error'],
                               error=levels[-1]['error']/levels[0]['error']))
    return checks


def fdfd_plate(n, eps=1., mu=1., *, frequency=30e9, width=.01, wall=None):
    dx = width/n
    solver = fdfd.ModeSolver1D(frequency=frequency, x_range=(-dx,width+dx),
        background_material=fdfd.Material(name='fill', epsilon=eps, mu=mu))
    wall = fdfd.materials.PEC if wall is None else wall
    solver.add_layer(x_range=(-dx,0.), material=wall)
    solver.add_layer(x_range=(width,width+dx), material=wall)
    solver.mesh(resolution=n+2, subpixels=1)
    return solver


def fdfd_rectangle(n, eps=1., *, frequency=10e9, width=.02286, height=.01016, wall=None):
    ny=max(6,n//2)
    dx,dy=width/n,height/ny
    outer=((-dx,width+dx),(-dy,height+dy))
    solver=fdfd.ModeSolver2D(frequency=frequency,x_range=outer[0],y_range=outer[1],
        background_material=fdfd.Material(name='fill',epsilon=eps))
    solver.add_geometry(shape=fdfd.shapes.Difference(shape=fdfd.shapes.Rectangle(bounds=outer),
        tool=fdfd.shapes.Rectangle(bounds=((0.,width),(0.,height)))),
        material=fdfd.materials.PEC if wall is None else wall)
    solver.mesh(resolution=(n+2,ny+2),subpixels=1)
    return solver


def electrostatics():
    for dim in (1,2):
        for n in (8,16):
            solver=fem.ElectrostaticSolver(dim=dim,x_range=(0.,1.),
                **({'y_range':(0.,.5)} if dim==2 else {}), outer_potential=None,
                background_material=fem.Material(epsilon=3.))
            solver.set_potential(potential=0.,geometry='left')
            solver.set_potential(potential=2.,geometry='right')
            solver.mesh(max_element_size=1/n)
            for backend in ('cpu','gpu'):
                result=solver.solve(backend=backend,max_refinements=0)
                record(f'electrostatic_{dim}d_potential','FEM',backend,n,result.potential,
                       2*result.coordinates[:,0],1e-9)
                exact_field=np.zeros_like(result.electric_field)
                exact_field[:,0]=-2.
                record(f'electrostatic_{dim}d_field','FEM',backend,n,result.electric_field,
                       exact_field,1e-9)
                record(f'electrostatic_{dim}d_energy','FEM',backend,n,result.energy,
                       .5*epsilon_0*3*4*(1 if dim==1 else .5),1e-8)
    # Piecewise P1 interpolation of the quadratic Poisson solution: actual
    # between-node error is measured, avoiding exact nodal superconvergence.
    for n in (8,16,32):
        solver=fem.ElectrostaticSolver(dim=1,x_range=(0.,1.),outer_potential=0.)
        solver.add_charge_density(density=1e-12,geometry=fem.shapes.Interval(bounds=(0.,1.)))
        solver.mesh(max_element_size=1/n)
        for backend in ('cpu','gpu'):
            r=solver.solve(backend=backend,max_refinements=0)
            x=np.linspace(0,1,2001)
            order=np.argsort(r.coordinates[:,0])
            value=np.interp(x,r.coordinates[order,0],r.potential[order])
            record('poisson_1d_interpolated_potential','FEM',backend,n,value,
                   1e-12*x*(1-x)/(2*epsilon_0),.025)


def waveguides():
    f,a=30e9,.01
    for label,eps,mu,pol in [('pec_TE1',1.,1.,'TE'),('lossy_TE1',2.25-.09j,1.,'TE'),
                             ('lossy_TEM',2.25-.09j,1.,'TM'),
                             ('anisotropic_TEM',(4.,7.,11.),(2.,3.,5.),'TM')]:
        exact=np.sqrt((eps[0]*mu[1]) if isinstance(eps,tuple) else eps*mu-(c/(2*f*a))**2*(pol=='TE'))
        for n in (40,80):
            fe=fem.ModeSolver1D(frequency=f,x_range=a,background_material=fem.Material(epsilon=eps,mu=mu))
            fe.mesh(resolution=n)
            fd=fdfd_plate(n,eps,mu)
            for backend in ('cpu','gpu'):
                r=fe.solve(num_modes=4,neff_guess=exact,max_refinements=0,backend=backend)
                mode=min((m for m in r if m.polarization==pol),key=lambda m:abs(m.neff-exact))
                record('waveguide_1d_'+label,'FEM',backend,n,mode.neff,exact,3e-4,
                       residual=float(mode.residual))
                r=fd.solve(num_modes=1,polarization=pol,guess=exact,tol=1e-10,
                           device='cpu' if backend=='cpu' else 'cuda')
                record('waveguide_1d_'+label,'FDFD',backend,n,r.neff[0],exact,3e-4,
                       residual=float(r.info.residuals[0]))
    f,a,b=10e9,.02286,.01016
    for label,eps in [('pec',1.),('lossy',2.25-.09j)]:
        exact=np.sqrt(eps-(c/(2*f*a))**2)
        for n in (12,24):
            fe=fem.ModeSolver2D(frequency=f,x_range=a,y_range=b,
                background_material=fem.Material(epsilon=eps),boundary=fem.materials.PEC)
            fe.mesh(resolution=(n,n//2),element_order=1)
            fd=fdfd_rectangle(2*n,eps)
            for backend in ('cpu','gpu'):
                r=fe.solve(num_modes=1,neff_guess=exact,max_refinements=0,backend=backend,
                           dense_linearization_limit=4)
                record('waveguide_2d_TE10_'+label,'FEM',backend,n,r[0].neff,exact,.004,
                       residual=float(r[0].residual),divergence_residual=float(r[0].divergence_residual))
                r=fd.solve(num_modes=1,guess=exact,tol=1e-10,device='cpu' if backend=='cpu' else 'cuda')
                record('waveguide_2d_TE10_'+label,'FDFD',backend,2*n,r.neff[0],exact,.004,
                       residual=float(r.info.residuals[0]))


def periodic():
    f,a,L=10e9,.02,.005
    for pol,eps in [('TE',1.),('TE',2.25-.09j),('TM',2.25-.09j)]:
        exact=np.sqrt(eps-(c/(2*f*a))**2*(pol=='TE'))
        for n in (12,24):
            fe=fem.PeriodicModeSolver2D(frequency=f,x_range=a,z_range=L,polarization=pol,
                background_material=fem.Material(epsilon=eps),boundary=fem.materials.PEC)
            fe.mesh(max_element_size=a/n)
            fd=fdfd.PeriodicModeSolver2D(frequency=f,x_range=(0.,a),z_range=(0.,L),
                background_material=fdfd.Material(name='fill',epsilon=eps),boundary=fdfd.materials.PEC)
            fd.mesh(resolution=(n,4),subpixels=1)
            for backend in ('cpu','gpu'):
                r=fe.solve(backend=backend,num_modes=1,neff_guess=exact,max_refinements=0,
                           direction='forward',eigensolver='sparse')
                record(f'periodic_2d_{pol}_{eps}','FEM',backend,n,r[0].neff,exact,.007,
                       residual=float(r[0].residual))
                r=fd.solve(num_modes=1,polarization=pol,guess=exact,tol=1e-10,ncv=40,
                           device='cpu' if backend=='cpu' else 'cuda')
                record(f'periodic_2d_{pol}_{eps}','FDFD',backend,n,r.neff[0],exact,.007,
                       residual=float(r.info.residuals[0]))
    # FEM 3D remains included; small conforming tetrahedral cell, TE10.
    exact=np.sqrt(2.25-(c/(2*f*a))**2)
    fe=fem.PeriodicModeSolver3D(frequency=f,x_range=a,y_range=.01,z_range=L,
                              background_material=fem.Material(epsilon=2.25))
    fe.mesh(max_element_size=.006,wavelength_elements=8)
    for backend in ('cpu','gpu'):
        r=fe.solve(backend=backend,num_modes=1,neff_guess=exact,max_refinements=0,
                   eigensolver='sparse')
        record('periodic_3d_TE10','FEM',backend,1,r[0].neff,exact,.01,
               residual=float(r[0].residual),gauss_residual=float(r[0].gauss_residual))


def bands():
    eps,mu=2.25-.09j,1.3-.02j
    beta=np.array([[.7],[.4]])
    exact=c*np.linalg.norm(beta)/(2*np.pi*np.sqrt(eps*mu))
    for n in (12,24):
        s=fdfd.BandStructureSolver2D(x_range=(0.,1.),y_range=(0.,1.2),
            background_material=fdfd.Material(name='fill',epsilon=eps,mu=mu))
        s.mesh(resolution=(n,n))
        for backend in ('cpu','gpu'):
            r=s.solve(beta_path=beta,num_modes=1,polarization='both',progress=False,
                      tol=1e-10,device='cpu' if backend=='cpu' else 'cuda')
            for pol in ('TE','TM'):
                record('bands_2d_lossy_'+pol,'FDFD',backend,n,r.frequencies[pol][0,0],exact,.001)


def scattering():
    # Exact Fabry-Perot slab coefficients, with the public port reference phases.
    f,width,d=10e9,.005,.006
    k=2*np.pi*f/c
    r0=(1-2)/(1+2)
    p=np.exp(-1j*2*k*d)
    reflection=r0*(1-p*p)/(1-r0*r0*p*p)
    transmission=(1-r0*r0)*p/(1-r0*r0*p*p)*np.exp(1j*k*d)
    for n in (1,2):
        fe=fem.ScatteringSolver2D(frequency=f,x_range=(0.,width),z_range=(-.025,.025),boundary=fem.materials.PEC)
        fe.add_rectangle(x_range=fe.x_range,z_range=(-d/2,d/2),material=fem.Material(epsilon=4.))
        fe.set_matched_ports()
        fe.mesh(max_element_size=.0009,element_order=n)
        for backend in ('cpu','gpu'):
            fe.solve_modes(num_modes=1,neff_guess=1.,num_elements=40,max_refinements=0,backend=backend)
            fe.set_incident_mode(0,reference_plane=-d/2)
            r=fe.solve(backend=backend,max_refinements=0)
            record('scattering_slab_S11_S21','FEM',backend,n,[r.S11,r.S21],
                   [reflection,transmission],.025 if n==1 else .004,metric='absolute')
            record('scattering_slab_power','FEM',backend,n,r.reflection+r.transmission,1.,.01,metric='absolute')
    # Independent Bessel/Hankel series: physical field error outside a cylinder.
    for pol in ('TE','TM'):
        for obj in ('PEC','dielectric'):
            for n in (120,240):
                fd=fdfd.ScatteringSolver2D(frequency=3e9,x_range=(-.15,.15),y_range=(-.15,.15))
                fd.add_circle(center=(0.,0.),radius=.025,material=fdfd.materials.PEC if obj=='PEC' else fdfd.Material(name='rod',epsilon=4.))
                fd.add_pml(thickness=.05)
                fd.mesh(resolution=(n,n))
                fd.set_source(inset=.075)
                for backend in ('cpu','gpu'):
                    r=fd.solve(polarization=pol,device='cpu' if backend=='cpu' else 'cuda')
                    component='Ez' if pol=='TE' else 'Hz'
                    xx,yy=np.meshgrid(*r.field_coordinates[component],indexing='ij')
                    rad,angle=np.hypot(xx,yy),np.arctan2(yy,xx)
                    selected=(rad>.035)&(rad<.065)
                    rad,angle=rad[selected],angle[selected]
                    k,a=fd._kernel.k0,.025
                    exact=np.exp(-1j*k*rad*np.cos(angle))
                    for order in range(-12,13):
                        if obj=='PEC':
                            coef=-jv(order,k*a)/hankel2(order,k*a) if pol=='TE' else -jvp(order,k*a)/h2vp(order,k*a)
                        else:
                            q=2. if pol=='TE' else .5
                            coef=(q*jv(order,k*a)*jvp(order,2*k*a)-jvp(order,k*a)*jv(order,2*k*a))/(h2vp(order,k*a)*jv(order,2*k*a)-q*hankel2(order,k*a)*jvp(order,2*k*a))
                        exact+=(-1j)**order*coef*hankel2(order,k*rad)*np.exp(1j*order*angle)
                    record('scattering_cylinder_'+obj+'_'+pol,'FDFD',backend,n,
                           r.fields[component][...,0][selected],exact,.08 if n==120 else .03)
                fd._kernel.close()


def tracking():
    width,height=.02286,.01016
    fc=c/(2*width)
    frequencies=np.array([.8,.9])*fc
    for dim in (1,2):
        n=80 if dim==1 else 32
        dx,dy=width/n,height/16
        outer=((-dx,width+dx),(-dy,height+dy))
        cls=fdfd.ModeTracker1D if dim==1 else fdfd.ModeTracker2D
        s=cls(frequencies=frequencies,x_range=outer[0],
              **({'y_range':outer[1]} if dim==2 else {}),port=fdfd.PortSpec(boundary='enclosed'))
        if dim==1:
            s.add_layer(x_range=(-dx,0.),material=fdfd.materials.PEC)
            s.add_layer(x_range=(width,width+dx),material=fdfd.materials.PEC)
        else:
            s.add_geometry(shape=fdfd.shapes.Difference(shape=fdfd.shapes.Rectangle(bounds=outer),
                tool=fdfd.shapes.Rectangle(bounds=((0.,width),(0.,height)))),material=fdfd.materials.PEC)
        s.mesh(resolution=n+2 if dim==1 else (n+2,18),subpixels=1)
        for backend in ('cpu','gpu'):
            r=s.solve(num_modes=1,**({'polarization':'TE'} if dim==1 else {}),
                guess=-.7j,progress=False,device='cpu' if backend=='cpu' else 'cuda',
                config=fdfd.TrackingConfig(verification_beta_tolerance=.006,verification_overlap=.995))
            modes=r.export()
            actual=[mode.beta/(2*np.pi*mode.frequency/c) for mode in modes]
            exact=[-1j*np.sqrt((fc/mode.frequency)**2-1) for mode in modes]
            record(f'tracking_{dim}d_evanescent','FDFD',backend,n,actual,exact,.004)


def sibc():
    # First-order wall-loss perturbation of a rectangular TE10 guide.
    f,a,b=10e9,.02286,.01016
    k=2*np.pi*f/c
    neff=np.sqrt(1-(np.pi/(k*a))**2)
    eta=np.sqrt(mu_0/epsilon_0)
    rs=np.sqrt(np.pi*f*mu_0/fdfd.materials.copper.conductivity)
    exact=rs/(b*eta*neff)*(1+2*b/a*(np.pi/(k*a))**2)
    for n in (24,48):
        dx,dy=a/n,b/(n//2)
        bounds=((-dx,a+dx),(-dy,b+dy))
        fe=fem.ModeSolver2D(frequency=f,x_range=bounds[0],y_range=bounds[1])
        fe.add_geometry(shape=fem.shapes.Difference(shape=fem.shapes.Rectangle(bounds=bounds),
            tool=fem.shapes.Rectangle(bounds=((0.,a),(0.,b)))), material=fem.materials.copper)
        fe.mesh(resolution=(n,n//2),element_order=2,material_aware=False,
                interface_refinement=None,boundary_refinement=None)
        fd=fdfd_rectangle(n,wall=fdfd.materials.copper)
        for backend in ('cpu','gpu'):
            r=fe.solve(backend=backend,num_modes=1,neff_guess=neff,max_refinements=0,
                       dense_linearization_limit=4)
            record('sibc_2d_TE10_attenuation','FEM',backend,n,-k*r[0].neff.imag,exact,.03)
            r=fd.solve(num_modes=1,guess=neff,tol=1e-11,device='cpu' if backend=='cpu' else 'cuda')
            record('sibc_2d_TE10_attenuation','FDFD',backend,n,r.attenuation_constant[0],exact,.03)


def dielectric_slab():
    # Even TE0 of a symmetric step-index slab: u*tan(u)=sqrt(V^2-u^2).
    f,half,L=10e9,.005,.06
    k=2*np.pi*f/c
    V=k*half*np.sqrt(4.-1.)
    u=brentq(lambda u:u*np.tan(u)-np.sqrt(V*V-u*u),1e-9,min(V,np.pi/2)-1e-9)
    exact=np.sqrt(4.-(u/(k*half))**2)
    decay=k*np.sqrt(exact*exact-1.)
    for n in (120,240):
        fe=fem.ModeSolver1D(frequency=f,x_range=L)
        fe.add_layer(x_range=(L/2-half,L/2+half),material=fem.Material(epsilon=4.))
        fe.mesh(resolution=n)
        fd=fdfd_plate(n,frequency=f,width=L)
        fd.add_layer(x_range=(L/2-half,L/2+half),material=fdfd.Material(name='core',epsilon=4.))
        fd.mesh(resolution=n+2,subpixels=1)
        for backend in ('cpu','gpu'):
            modes=fe.solve(backend=backend,num_modes=4,neff_guess=exact,max_refinements=0)
            mode=min((m for m in modes if m.polarization=='TE'),key=lambda m:abs(m.neff-exact))
            result=fd.solve(num_modes=1,polarization='TE',guess=exact,tol=1e-10,
                            device='cpu' if backend=='cpu' else 'cuda')
            for package,value,x,field in [
                ('FEM',mode.neff,mode.fields.coordinates[0],mode.component('Ey')),
                ('FDFD',result.neff[0],result.field_coordinates['Ey'][0],result.fields['Ey'][:,0])]:
                record('dielectric_slab_TE0_index',package,backend,n,value,exact,.002)
                distance=abs(x-L/2)
                profile=np.where(distance<=half,np.cos(u*distance/half),np.cos(u)*np.exp(-decay*(distance-half)))
                coefficient=np.vdot(profile,field)/np.vdot(profile,profile)
                record('dielectric_slab_TE0_field',package,backend,n,field/coefficient,profile,.015)


def main():
    for name, function in [('electrostatics',electrostatics),('waveguides',waveguides),
                           ('periodic',periodic),('bands',bands),('scattering',scattering),('tracking',tracking),
                           ('sibc',sibc),('dielectric_slab',dielectric_slab)]:
        run(name,function)
    bad=[row for row in ROWS if not row['passed']]
    checks=verify_summary(ROWS)
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2))
    print(f'Completed {len(ROWS)} analytical measurements; {len(bad)} tolerance failures; {len(FAILURES)} exceptions.',flush=True)
    raise SystemExit(bool(bad or FAILURES or any(not check['passed'] for check in checks)))


if __name__=='__main__':
    main()
