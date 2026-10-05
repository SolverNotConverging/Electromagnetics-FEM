"""Fixed-mesh full-vector lead modes and second-order uniform scattering."""

# Run directly from the downloaded repository without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from fem_waveguide_scattering import materials

import fem_waveguide_scattering as scattering


OUTPUT_DIR = _Path(__file__).resolve().parents[1] / "outputs" / _Path(__file__).stem


def main():
    # A 2D x-z mesh represents full-vector fields in a uniform waveguide.
    simulation = scattering.WaveguideScatteringSolver2D(frequency=299792458.0 / 1.0, x_range=(0.0, 0.5), z_range=(-2.0, 2.0), boundary=materials.PEC)
    simulation.add_pml(thickness=0.5, direction='z')
    simulation.mesh(max_element_size=0.15, element_order=2)
    # First solve the lead modes, then choose the zero-based incident mode.
    modes = simulation.solve_modes(num_modes=1, neff_guess=1., num_elements=16, max_refinements=0)
    simulation.set_incident_mode(0)
    # The scattering solve returns power-normalized reflection and transmission.
    result = simulation.solve(max_refinements=0)
    print("TEM effective index (exact 1):", modes[0].neff)
    print("Reflection (exact 0):", result.reflection)
    print("Transmission (exact 1):", result.transmission)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT_DIR / "results.h5")
    result.show()
    return modes, result


if __name__ == "__main__":
    main()
