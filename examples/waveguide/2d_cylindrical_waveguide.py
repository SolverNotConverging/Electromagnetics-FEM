"""Solve and display 24 full-vector modes of a circular PEC waveguide."""
import fem

radius = 10e-3
solver = fem.ModeSolver2D(
    frequency=60e9,
    x_range=(-1.1*radius, 1.1*radius),
    y_range=(-1.1*radius, 1.1*radius),
)
# Fill the rectangular model corners with PEC, leaving a circular vacuum guide.
solver.add_geometry(
    name="circular_wall",
    shape=fem.shapes.Annulus(center=(0., 0.), inner_radius=radius, outer_radius=2*radius),
    material=fem.materials.PEC,
    clip=True,
)
solver.mesh(max_element_size=1e-3, element_order=2)
modes = solver.solve(
    num_modes=24,
    track_modes=True,
    track_options={"degeneracy_tolerance": 1e-3},
    max_refinements=0,
    backend="cpu",
)

for mode in modes:
    print(f"Mode {mode.index}: neff={mode.neff:.9g}, {mode.polarization}, "
          f"irrep={mode.metadata['symmetry_irrep']}, "
          f"partner={mode.metadata['symmetry_partner']}")

# Select each of the 24 modes and its field components in the viewer.
modes.show()
