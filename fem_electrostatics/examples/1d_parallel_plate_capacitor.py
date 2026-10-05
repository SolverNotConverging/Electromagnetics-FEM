"""Layered 1D FEM capacitor with geometry created before the mesh."""

# Run directly from the downloaded repository without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from fem_electrostatics import Material, shapes

from fem_electrostatics import ElectrostaticSolver


OUTPUT_DIR = _Path(__file__).resolve().parents[1] / "outputs" / _Path(__file__).stem


def main():
    # Define geometry and relative permittivity before creating the mesh.
    dielectric = Material(name="dielectric slab", epsilon=6.0)
    solver = ElectrostaticSolver(dim=1, outer_potential=None, x_range=(0.0, 0.01))
    solver.add_geometry(name='dielectric', material=dielectric, shape=shapes.Interval(bounds=(0.004, 0.007)))
    solver.set_potential(potential=0.0, name='ground', geometry='left')
    solver.set_potential(potential=10.0, name='drive', geometry='right')

    # Potentials are in volts; max_element_size is in metres.
    solver.mesh(max_element_size=0.0004)
    result = solver.solve(max_refinements=0)
    print(f"nodes={len(result.mesh_data.coordinates)}, energy={result.energy:.6e} J/m2")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    solver.result.save(OUTPUT_DIR / "results.h5")
    solver.show()
    return result


if __name__ == "__main__":
    main()
