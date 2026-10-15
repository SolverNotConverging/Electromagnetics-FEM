# FEM documentation

Start with [installation](installation.md) and the guide for your physics.
For an existing checkout, read the [2.0 migration notes](migration-2.0.md).

| Family | Domain | Materials | Results and guide |
|---|---|---|---|
| Electrostatics | 1D or 2D | Positive scalar, diagonal, or symmetric positive-definite permittivity | [Potential, fields, charge, energy](../fem_electrostatics/docs/guide.rst) |
| Waveguide modes | 1D or 2D cross-section | Scalar or diagonal permittivity/permeability | [Propagation modes and dispersion](../fem_waveguide_modes/docs/guide.rst) |
| Periodic modes | 2D or 3D periodic cell | Scalar or diagonal permittivity/permeability | [Bloch modes and periodic field envelopes](../fem_periodic_modes/docs/guide.rst) |
| Waveguide scattering | 2D x/z mesh, full-vector fields with invariant y dependence | Scalar permittivity/permeability | [Ports, S-parameters, fields, radiation](../fem_waveguide_scattering/docs/guide.rst) |

Each guide links to its API reference and runnable examples. Boundary, PML,
element-order, and backend support depend on the family and dimension; use the
family guide when configuring these features.

## Common workflow

1. Construct a solver with keyword arguments and SI quantities.
2. Define reusable `Material` and `shapes` values and assign them to geometry.
3. Configure boundary conditions and, for scattering, excitation and monitors.
4. Call `mesh()` to select the initial discretization.
5. Call `solve()` and inspect `solve_info`, including adaptive stopping status.
6. Call `plot()`, `show()`, or `save(path)`. Use `load_result(path)` for inspection.

Geometry edits invalidate the mesh and result. Remeshing invalidates the result.
Solvers retain explicit mesh settings for automatic remeshing. Frequency is fixed
on a solver; use a supported sweep or construct another solver.

EM fields use `exp(+i*omega*t)` and guided propagation `exp(-i*beta*z)`.
Passive constitutive values have nonpositive imaginary parts. Periodic results
store field envelopes with Bloch phase described separately. Python indices are
zero-based. Scalar controls accept Python and NumPy numbers, including integral
floats for counts; booleans, numeric strings, and arrays are rejected.

`max_refinements=0` performs a fixed-mesh solve. Adaptive convergence measures
discretization error separately from algebraic residuals. The 2D waveguide
backend uses global remeshing; check its guide before interpreting refinement
as local element marking. Scattering has its own port-mode implementation.

## Libraries and development

- [Shared materials, shapes, and result contracts](../fem_common/docs/guide.rst)
- [Adaptivity](../fem_adaptivity/docs/guide.rst)
- [Analytical benchmarks](../benchmarks/README.md)
- [Contributing](../CONTRIBUTING.md)
- [Tests](../tests/README.rst)

- [CUDA architecture compatibility and device qualification](cuda-compatibility.md)
