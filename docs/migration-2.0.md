# Migrating to 2.0

This is a breaking source and validation update for the shared team repository.
Update scripts and environments together.

- Python modules now live directly inside each package. Replace private imports
  such as `fem_waveguide_modes.src.solver_2d` with
  `fem_waveguide_modes.solver_2d`. Prefer public imports such as
  `from fem_waveguide_modes import ModeSolver2D`.
- `fem_common.grid` and `fem_common._yee_scene` have been removed. The repository
  contains FEM implementations; those unused grid adapters have no replacement.
- Scalar numeric controls reject numeric strings, booleans, complex numbers for
  real quantities, and arrays. Pass `4` or `4.0` for a count, not `"4"` or `[4]`.
  Mesh connectivity must contain exact, in-range integer indices; fractional
  values are rejected rather than truncated.
- The custom refined Arnoldi implementation, `periodic_eigensolver` package,
  Cython extension, CUDA Arnoldi kernel and their benchmarks are removed.
  Use `eigensolver="sparse"` for standard SciPy/ARPACK shift-invert iteration.
  Replace `max_restarts` with the optional ARPACK `max_iterations` limit.
- Backends are `"auto"` (default), `"cpu"`, and `"gpu"`. Auto tries the GPU
  runtime first and falls back to SciPy CPU when unavailable. Replace old
  `"python"`/`"cython"` with `"cpu"` and `"cuda"` with `"gpu"`.
  Shared GPU implementation now lives in `fem_common`.
- Solver-only installations no longer need a compiler. The obsolete
  `setup_cython.py` script and `FEM_BUILD_PYTHON` CMake option are removed.
- Modal `solve_info` now selects explicit diagnostic keys and includes recorded
  backend/method details. Custom metadata names no longer enter the report by prefix.
- `pyproject.toml` is authoritative for version and runtime requirements.
  Run `python scripts/sync_metadata.py` after changing either.
- Generated API references are reproducible. Edit the generator or source
  signatures, then run `python scripts/generate_api_reference.py`.
- Use `FEM_BUILD_NATIVE_APPS=OFF` for a solver-only installation. See
  [installation](installation.md) for commands and compiler requirements.

Public solver-family imports and the HDF5 schema remain supported. Existing
valid `cem-fem-results` schema 1.0 files retain their interpretation. Invalid
connectivity previously accepted through truncation now fails validation.

Build this checkout or install a matching locally built 2.0 wheel.
Older release wheels do not contain this update.
