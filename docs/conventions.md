# Conventions

## Solver families

| Family | Domain | Materials | Guide |
|---|---|---|---|
| Electrostatics | 1D or 2D | Positive scalar, diagonal, or symmetric positive-definite permittivity | [Electrostatics](solvers/electrostatics.rst) |
| Waveguide modes | 1D or 2D cross-section | Scalar or diagonal permittivity/permeability | [Waveguide modes](solvers/waveguide.rst) |
| Periodic modes | 2D or 3D periodic cell | Scalar or diagonal permittivity/permeability | [Periodic modes](solvers/periodic.rst) |
| Scattering | 2D x/z mesh, full-vector fields with invariant y dependence | Scalar permittivity/permeability | [Scattering](solvers/scattering.rst) |

Boundary, PML, element-order and backend support depend on the family and
dimension; see the family guide.

## Workflow

1. Construct a solver with keyword arguments and SI quantities.
2. Define reusable `fem.Material` and `fem.shapes` values and assign them to geometry.
3. Configure boundary conditions and, for scattering, excitation and monitors.
4. Call `mesh()` to select the initial discretization.
5. Call `solve()` and inspect `solve_info`, including adaptive stopping status.
6. Call `plot()`, `show()` or `save(path)`; `fem.load(path)` reopens a result.

Geometry edits invalidate the mesh and result. Remeshing invalidates the result.
Solvers retain explicit mesh settings for automatic remeshing. Frequency is fixed
on a solver; use a supported sweep or construct another solver.

## Units and signs

Lengths are in metres, frequencies in hertz and constitutive values are relative.
Fields use `exp(+i*omega*t)` and guided propagation `exp(-i*beta*z)`. Passive
materials have non-positive imaginary parts and forward attenuation is
`alpha = -Im(beta)`. Periodic results store field envelopes with the Bloch phase
recorded separately.

## Vocabulary

The same names mean the same thing in every family:

| Name | Meaning |
|---|---|
| `frequency`, `frequencies` | Frequency in hertz (one value, or a sweep) |
| `neff` | Complex effective index |
| `beta`, `alpha` | Propagation constant (rad/m) and attenuation `-Im(beta)` (1/m) |
| `num_modes`, `neff_guess` | Number of modes requested and the shift-invert target |
| `max_element_size`, `element_order` | Mesh size in metres and finite-element order |
| `max_refinements`, `adaptive_tolerance` | Adaptive budget; `0` performs a fixed-mesh solve |
| `backend` | `"auto"`, `"cpu"` or `"gpu"` |
| `solve_info`, `mesh_data` | Solver diagnostics and the mesh snapshot of a result |

Python indices (modes, cases) are zero-based. Scalar controls accept Python and
NumPy numbers, including integral floats for counts; booleans, numeric strings
and arrays are rejected. Errors derive from `fem.FEMError`.
