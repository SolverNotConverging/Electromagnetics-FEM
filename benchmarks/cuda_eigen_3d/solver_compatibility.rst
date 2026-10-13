CUDA eigensolver compatibility in this checkout
===============================================

Inspected source base: dev commit 9d3dbad3fa1fe5ec269c686c7a7d324cbf5af94e.

Periodic modes: already integrated
----------------------------------

``PeriodicModeSolver2D`` and ``PeriodicModeSolver3D`` accept
``eigensolver="refined", arnoldi_backend="cuda"``. The explicit CUDA backend
also makes ``eigensolver="auto"`` select refined Arnoldi. Their quadratic
neff pencils are linearized to twice the reduced FEM dimension, then passed
to ``periodic_eigensolver.solve_generalized``. The 3D wrapper exposes ncv,
max_restarts and random_seed; the 2D wrapper currently uses library defaults
for these controls. Dense QZ is a CPU method.

Standalone 1D waveguide modes: compatible, not integrated
--------------------------------------------------------

``fem_waveguide_modes/src/solver_1d.py::_eigenpairs`` solves a generalized
TE/TM pencil with eigenvalue ``neff**2``. Its assembled A and B can be passed
to the existing CUDA eigensolver with a shift near ``neff_guess**2``. An
adapter must preserve the small imaginary shift perturbation, candidate
oversampling, square-root branch selection, residual checks and polarization
handling. The public solve API currently has no CUDA/backend option. Typical
small 1D pencils may be dominated by GPU setup cost.

Standalone 2D waveguide modes: compatible, with an efficiency tradeoff
--------------------------------------------------------------------

``fem_waveguide_modes/src/assembly.py::linearized_pencil`` already returns
the generalized companion pencil required by the CUDA library. An adapter
could pass it to ``solve_generalized`` and extract the first n components of
the returned 2n-component eigenvectors, retaining the existing QEP/Gauss,
direction and physical-mode checks.

The current CPU sparse path in ``solve_qep_candidates`` exploits block
elimination: it factors only the n-by-n polynomial Q(sigma). The generic
CUDA implementation factors the complete 2n-by-2n shifted pencil. Therefore
compatibility does not establish a performance improvement. Preserving this
block elimination in a CUDA operator would require extending the GPU solver's
internal operator/factor interface. No CUDA option is exposed by the current
ModeSolver2D public API.

Scattering port modes: compatible, not integrated
-------------------------------------------------

``fem_waveguide_scattering/src/modes.py::_linearized_candidates`` has the
same quadratic-pencil structure and an n-by-n block-eliminated CPU solve.
Its port eigenmode stage could use a CUDA adapter subject to the same
efficiency and filtering considerations. The driven scattering system itself
is a linear-system solve, so it cannot be replaced with an eigensolver call.

Electrostatics
--------------

The electrostatic potential solve is a sparse linear system, not a generalized
eigenproblem. CUDA acceleration there would need a separate linear-solver
integration, potentially using cuDSS directly.

No production solver changes were made for this compatibility inspection.
These conclusions follow the current assembly and dispatch code; only the
already-integrated periodic backend is measured by the accompanying suite.
