"""Solve and display 24 full-vector modes of a square PEC waveguide."""
import fem

solver = fem.ModeSolver2D(
    frequency=60e9,
    x_range=(0., 20e-3),
    y_range=(0., 20e-3),
)
solver.mesh(max_element_size=1e-3, element_order=2)
modes = solver.solve(
    num_modes=24,
    track_modes=True,
    max_refinements=0,
    backend="cpu",
)

for mode in modes:
    print(f"Mode {mode.index}: neff={mode.neff:.9g}, {mode.polarization}, "
          f"irrep={mode.metadata['symmetry_irrep']}, "
          f"partner={mode.metadata['symmetry_partner']}")

# Select each of the 24 modes and its field components in the viewer.
modes.show()
