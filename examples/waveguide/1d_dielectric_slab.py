"""Solve and visualize the leading modes of a dielectric slab."""

from pathlib import Path

import fem

OUTPUT_DIR = Path(__file__).parent / "outputs" / Path(__file__).stem

wavelength = 1.55e-6
frequency = 299_792_458.0 / wavelength
cladding = fem.Material(name="silica cladding", epsilon=1.44**2)
silicon = fem.Material(name="silicon core", epsilon=3.45**2)
solver = fem.ModeSolver1D(
    frequency=frequency,
    x_range=(-3e-6, 3e-6),
    background_material=cladding,
)
solver.add_layer(x_range=(-2.5e-7, 2.5e-7), name="core", material=silicon)

solver.mesh(max_element_size=6e-08)
modes = solver.solve(backend="auto", max_refinements=0, num_modes=4)

for number, mode in enumerate(modes):
    print(
        f"mode {number}: polarization={mode.polarization}, "
        f"neff={mode.neff:.9g}, alpha={mode.alpha:.4g} 1/m, "
        f"residual={mode.residual:.3e}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
modes.save(OUTPUT_DIR / "results.h5")
modes.show()
