"""Parallel-plate 1D modes compared with analytic cutoffs."""

# Run directly from the downloaded repository without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = next(parent for parent in _Path(__file__).resolve().parents
             if (parent / "cem_common" / "__init__.py").is_file())
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))


import numpy as np
from fem_waveguide_modes import ModeSolver1D


OUTPUT_DIR = _Path(__file__).resolve().parents[1] / "outputs" / _Path(__file__).stem


def main():
    frequency = 299_792_458.0  # vacuum wavelength = 1 metre
    expected = np.sqrt(.75)
    line = ModeSolver1D(frequency=frequency, x_range=1.0)
    line.mesh(resolution=48)
    line_modes = line.solve(max_refinements=0, num_modes=3, neff_guess=expected)
    print("1D effective indices (TE/TM cutoff pair and TEM):", line_modes.neff)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    line.result.save(OUTPUT_DIR / "results.h5")
    line.show()
    return line_modes


if __name__ == "__main__":
    main()
