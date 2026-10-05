"""Leaky-wave antenna formed by a finite slot in a grounded dielectric slab.

The background lead contains a zero-thickness PEC ground plane invariant in
z.  The actual device releases a finite part of that plane, so the aperture is
a boundary perturbation rather than a permittivity perturbation.  The default
run saves a complete HDF5 result, opens it in ``fem-waveguide-scattering-viewer``, and displays
the electric-field magnitude with Matplotlib.
"""

from __future__ import annotations

# Run directly from the downloaded repository without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "cem_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from cem_common import Material, materials, shapes

from pathlib import Path

import fem_waveguide_scattering as scattering

OUTPUT_DIR = _ROOT / "fem_waveguide_scattering/outputs" / Path(
    __file__,
).stem


MM = 1.0e-3
DESIGN_FREQUENCY_HZ = 20.0e9
CORE_EPS_R = 10.2


def build_simulation(frequency_hz: float = DESIGN_FREQUENCY_HZ, *, matched_ports: bool = False) -> scattering.WaveguideScatteringSolver2D:
    """Return the grounded-slab slot configuration at one frequency."""

    substrate = Material(name="grounded slab", epsilon=CORE_EPS_R)
    simulation = scattering.WaveguideScatteringSolver2D(frequency=frequency_hz, angle=0.0, x_range=(-20.0 * MM, 20.0 * MM), z_range=(-30.0 * MM, 30.0 * MM), background_material=materials.air)
    simulation.add_rectangle(x_range=(0.0, 1.27 * MM), z_range=simulation.z_range, background=True, name='dielectric_slab', material=substrate)
    ground = simulation.add_geometry(background=True, name='ground_plane', shape=shapes.Segment(start=(0.0, simulation.z_range[0]), end=(0.0, simulation.z_range[1])), material=materials.PEC)
    simulation.add_slot(name='ground_slot', geometry=ground, z_range=(-1.0 * MM, 1.0 * MM))
    simulation.add_pml(target_reflection=1e-08, thickness=4.0 * MM, direction='x')
    if matched_ports:
        simulation.set_matched_ports()
    else:
        simulation.add_pml(target_reflection=1e-08, thickness=6.0 * MM, direction='z')
    simulation.set_monitors(left=-20.0 * MM, right=20.0 * MM)
    return simulation


def solve_single(output: Path) -> scattering.ScatteringResult:
    """Run the design-frequency solve and save its full HDF5 record."""

    simulation = build_simulation()
    mesh = simulation.mesh(max_element_size=2.0 * MM, wavelength_elements=10)
    modes = simulation.solve_modes(
        max_refinements=0,
        num_modes=1,
        neff_guess=1.8,
        num_elements=96,
    )
    simulation.set_incident_mode(modes[0])
    result = simulation.solve(max_refinements=0)
    output_path = result.save(output)

    print("selected maximum edge (mm) =", mesh.info.requested_maximum_edge / MM)
    print("surface-mode effective index =", modes[0].neff)
    print("S11 =", result.S11)
    print("S21 =", result.S21)
    print("R, T =", result.reflection, result.transmission)
    print("radiated, absorbed power (W/m) =", result.radiated_power, result.absorbed_power)
    print("power-balance error =", result.power_balance_error)
    print("released PEC facets =", result.solve_info["released_pec_facet_count"])
    print("HDF5 result =", output_path)
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = solve_single(OUTPUT_DIR / 'results.h5')
    result.show()


if __name__ == "__main__":
    main()
