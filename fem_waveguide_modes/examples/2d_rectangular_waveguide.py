"""Second-order rectangular waveguide modes compared with the analytic TE10 cutoff."""

# Run directly from the downloaded repository without installing solver packages.
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))


import numpy as np
from fem_waveguide_modes import ModeSolver2D


OUTPUT_DIR = _Path(__file__).resolve().parents[1] / "outputs" / _Path(__file__).stem


def main():
    frequency = 299_792_458.0
    expected = np.sqrt(.75)
    guide = ModeSolver2D(frequency=frequency, x_range=1.0, y_range=0.5)
    guide.mesh(resolution=(5, 3), element_order=2)
    vector_modes = guide.solve(max_refinements=0, num_modes=1)
    print("2D TE10 effective index:", vector_modes[0].neff)
    print("Analytic TE10 effective index:", expected)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    guide.result.save(OUTPUT_DIR / "results.h5")
    guide.show()
    return vector_modes


if __name__ == "__main__":
    main()
