"""Plot saved 2d_ridge_waveguide_dispersion results without running the solver again."""
from pathlib import Path

import argparse
import csv
import fem

DEFAULT_INPUT = Path(__file__).parents[1] / "outputs" / "2d_ridge_waveguide_dispersion/dispersion.csv"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_INPUT,
                    help="Dispersion CSV from any 2D sweep in this solver family.")
args = parser.parse_args()
if not args.path.is_file():
    parser.error(f"{args.path} does not exist. Run a dispersion example first or supply its CSV path.")
with args.path.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
frequencies = sorted({float(row["frequency"]) for row in rows})
modes = sorted({int(row["mode"]) for row in rows})
values = {(float(row["frequency"]), int(row["mode"])):
          complex(float(row["neff_real"]), float(row["neff_imag"])) for row in rows}
neff = [[values[frequency, mode] for mode in modes] for frequency in frequencies]
figure = fem.plot_dispersion(frequencies, neff, show=False)
output = args.path.with_suffix(".png")
figure.savefig(output, dpi=160)
print(output)
