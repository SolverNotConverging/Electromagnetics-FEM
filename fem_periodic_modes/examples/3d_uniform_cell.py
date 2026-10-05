"""Uniform 3D periodic cell compared with the analytic TE10 effective index."""

# Run directly from the downloaded repository without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))


import numpy as np
from fem_periodic_modes import PeriodicModeSolver3D
from fem_periodic_modes import Material


OUTPUT_DIR = _Path(__file__).resolve().parents[1] / "outputs" / _Path(__file__).stem


def main():
    dielectric = Material(name="uniform dielectric", epsilon=2.25)
    common = dict(frequency=10e9, x_range=.02, z_range=.005,
                  background_material=dielectric)
    vector = PeriodicModeSolver3D(**common, y_range=0.01)
    # A fixed 3D mesh must resolve the Gauss filter without adaptive retries.
    vector.mesh(max_element_size=0.006, wavelength_elements=8)
    te10 = vector.solve(num_modes=1, max_refinements=0, neff_guess=1.3)
    expected = np.sqrt(2.25 - (np.pi / (.02 * vector.k0))**2)
    print("3D TE10 effective index:", te10[0].neff)
    print("Analytic TE10 effective index:", expected)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vector.result.save(OUTPUT_DIR / "results.h5")
    vector.show()
    return te10


if __name__ == "__main__":
    main()
