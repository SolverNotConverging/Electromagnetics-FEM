"""Four periodic z unit cells; SI units internally, mesh sizes in millimetres.

These are eigenproblem benchmark fixtures, not optimized filter/antenna designs.
The iris and rod reproduce the existing examples. SIW and perforated guide are
explicit new fixtures using the same PeriodicModeSolver3D assembly path.
"""
from __future__ import annotations

from pathlib import Path
import runpy

from fem import PeriodicModeSolver3D, Material, materials, shapes


MM = 1e-3
ROOT = Path(__file__).resolve().parents[2]


def add_box(solver, *, name, x_range, y_range, z_range, material):
    return solver.add_geometry(name=name, material=material,
        shape=shapes.Box(bounds=(x_range, y_range, z_range)))


def vertical_cylinder(x, z, radius, y_range):
    """Circular cylinder along y, specified in millimetres."""
    return shapes.Cylinder(center=(0., 0.), radius=radius*MM,
        z_range=(y_range[0]*MM, y_range[1]*MM)).rotated(
            angle=-90., axis=(1., 0., 0.)).translated(offset=(x*MM, 0., z*MM))


def iris():
    solver = PeriodicModeSolver3D(frequency=12e9, x_range=(0., 22.86*MM),
        y_range=(0., 10.16*MM), z_range=(0., 8.*MM), boundary=materials.PEC)
    for name, x_range in (("left_iris", (0., 4.*MM)), ("right_iris", (18.86*MM, 22.86*MM))):
        add_box(solver, name=name, x_range=x_range, y_range=(0., 10.16*MM),
            z_range=(3.6*MM, 4.4*MM), material=materials.PEC)
    return solver, .7+0j, dict(frequency_hz=12e9, period_mm=8.,
        description="Existing WR-90 cell, 0.8 mm thick opposing 4 mm irises; PEC walls.")


def siw():
    solver = PeriodicModeSolver3D(frequency=20e9, x_range=(-8.*MM, 8.*MM),
        y_range=(-.8*MM, .8*MM), z_range=(0., 8.*MM),
        background_material=Material(name="SIW substrate", epsilon=2.2), boundary=materials.PEC)
    # Via fences join the y PEC ground planes. A pair of larger interior vias
    # loads each cell. Lateral PML absorbs fields beyond the fences.
    for x in (-5., 5.):
        for z in (1., 3., 5., 7.):
            solver.add_geometry(name=f"fence_{x}_{z}", material=materials.PEC,
                shape=vertical_cylinder(x, z, .5, (-.8, .8)))
    for x in (-2.7, 2.7):
        solver.add_geometry(name=f"loading_via_{x}", material=materials.PEC,
            shape=vertical_cylinder(x, 4., .7, (-.8, .8)))
    solver.add_pml(thickness=1.5*MM, direction="x")
    return solver, 1.1-.005j, dict(frequency_hz=20e9, period_mm=8.,
        description="Via-loaded SIW: eps_r=2.2, thickness 1.6 mm, fence spacing 10 mm, via diameter 1 mm/pitch 2 mm, loading vias diameter 1.4 mm at x=+-2.7 mm, lateral PML 1.5 mm.")


def perforated_waveguide():
    solver = PeriodicModeSolver3D(frequency=20e9, x_range=(-17.43*MM, 17.43*MM),
        y_range=(-5.58*MM, 13.58*MM), z_range=(0., 15.*MM), boundary=materials.PEC)
    # PEC body with a WR-90 air cavity and an off-centre hole in its top wall.
    # Air above the wall terminates in transverse PML; z is periodic.
    body = shapes.Box(bounds=((-17.43*MM, 17.43*MM), (-5.58*MM, 5.58*MM), (0., 15.*MM)))
    cavity = shapes.Box(bounds=((-11.43*MM, 11.43*MM), (-5.08*MM, 5.08*MM), (0., 15.*MM)))
    aperture = vertical_cylinder(4., 7.5, 2., (5.0, 5.7))
    conductor = shapes.Difference(shape=shapes.Difference(shape=body, tool=cavity), tool=aperture)
    solver.add_geometry(name="perforated_pec_waveguide", shape=conductor, material=materials.PEC)
    solver.add_pml(thickness=3.*MM, direction="x")
    solver.add_pml(thickness=3.*MM, direction="y+")
    return solver, .9-.01j, dict(frequency_hz=20e9, period_mm=15.,
        description="WR-90 cavity under a 0.5 mm PEC top wall with one 4 mm diameter hole per 15 mm cell, offset x=4 mm. Air above the wall; x and y+ PML 3 mm.")


def dielectric_rod():
    solver = PeriodicModeSolver3D(frequency=20e9, x_range=(-9*MM, 9*MM),
        y_range=(-8*MM, 8*MM), z_range=(0., 10*MM), boundary=materials.PEC)
    add_box(solver, name="dielectric_rod", x_range=(-3*MM, 3*MM),
        y_range=(-2*MM, 2*MM), z_range=(0., 10*MM), material=Material(epsilon=6.))
    add_box(solver, name="periodic_metal_patch", x_range=(-2*MM, 2*MM),
        y_range=(2*MM, 2.2*MM), z_range=(3.5*MM, 6.5*MM), material=materials.PEC)
    solver.add_pml(thickness=3*MM, direction="all")
    return solver, .5-.02j, dict(frequency_hz=20e9, period_mm=10.,
        description="6x4 mm eps_r=6 rod, 4x3x0.2 mm PEC patch per 10 mm cell; 3 mm transverse PML.")


BUILDERS = {"iris": iris, "siw": siw, "perforated_waveguide": perforated_waveguide,
            "dielectric_rod": dielectric_rod}


def assemble(name, mesh_mm):
    solver, sigma, info = BUILDERS[name]()
    solver.mesh(max_element_size=mesh_mm*MM, wavelength_elements=4)
    info.update(field_dofs=solver.system.ndofs, mesh_mm=mesh_mm,
        mesh_nodes=int(solver.mesh_data.mesh.nvertices),
        mesh_tetrahedra=int(solver.mesh_data.mesh.nelements))
    return solver, sigma, info
