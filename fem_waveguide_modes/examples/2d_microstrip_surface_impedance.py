"""Solve a 10 GHz copper microstrip using surface-impedance conductors.

The 35 micrometre copper bodies are represented as opaque holes.  Their
dielectric-facing walls receive the copper surface impedance; no volume mesh
is generated inside the metal.
"""

from __future__ import annotations

# Run directly from the downloaded repository without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from fem_waveguide_modes import Material, materials, shapes

from fem_waveguide_modes import ModeSolver2D

FREQUENCY = 10.0e9
DOMAIN_X = (-6.0e-3, 6.0e-3)
DOMAIN_Y = (-35.0e-6, 6.0e-3)
SUBSTRATE_HEIGHT = 1.524e-3
COPPER_THICKNESS = 35.0e-6
STRIP_WIDTH = 3.0e-3

# A representative low-loss microwave laminate near 10 GHz.
SUBSTRATE_EPSILON = 3.55 * (1.0 - 1j * 0.0027)

MESH_OPTIONS = {
    "max_element_size": 0.60e-3,
    "wavelength_elements": 10,
    "material_aware": True,
    # boundary_refinement intentionally uses the solver default (0.5).
}


def build_solver() -> ModeSolver2D:
    """Return the continuous microstrip model, ready for discretization."""

    substrate = Material(name="microwave laminate", epsilon=SUBSTRATE_EPSILON)
    solver = ModeSolver2D(frequency=FREQUENCY, x_range=DOMAIN_X, y_range=DOMAIN_Y, boundary=materials.PEC, background_material=materials.air)
    solver.add_rectangle(x_range=DOMAIN_X, y_range=(0.0, SUBSTRATE_HEIGHT), name='substrate', material=substrate)
    solver.add_geometry(name='copper_ground', shape=shapes.Rectangle(bounds=(DOMAIN_X, (-COPPER_THICKNESS, 0.0))), material=materials.copper)
    solver.add_geometry(name='copper_strip', shape=shapes.Rectangle(bounds=((-0.5 * STRIP_WIDTH, 0.5 * STRIP_WIDTH), (SUBSTRATE_HEIGHT, SUBSTRATE_HEIGHT + COPPER_THICKNESS))), material=materials.copper)

    return solver


OUTPUT_DIR = _Path(__file__).resolve().parents[1] / "outputs" / _Path(__file__).stem


def main() -> None:
    solver = build_solver()
    mesh = solver.mesh(**MESH_OPTIONS)
    print(
        f"mesh: {mesh.info.nodes} nodes, {mesh.info.elements} triangles, "
        f"h=[{mesh.info.minimum_edge:.3g}, {mesh.info.maximum_edge:.3g}] m"
    )

    modes = solver.solve(max_refinements=0, num_modes=1)
    mode = modes[0]
    print(
        f"microstrip mode: neff={mode.neff:.9g}, "
        f"alpha={mode.alpha:.4g} 1/m, residual={mode.residual:.3e}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    solver.result.save(OUTPUT_DIR / "results.h5")
    solver.show()


if __name__ == "__main__":
    main()
