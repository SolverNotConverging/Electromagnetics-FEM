# Working on FEM

Use Python 3.12 and run `uv sync --locked` from the repository root. This installs
development dependencies; native applications are built separately. See
[installation](docs/installation.md) for source and solver-only builds.

## Layout and ownership

Each solver package holds its Python modules, guide, API reference, examples,
and ignored output directory. `fem_common` owns materials, shapes, input
validation, geometry handles, result contracts, and archive envelopes.
`fem_adaptivity` owns the solve/estimate/refine policy. Numerical assembly,
physics-specific diagnostics, and field recovery stay in each solver.

`apps/` contains native viewers and the calculator; `cmake/` holds shared build
support. `tests/` holds regression tests and the public API inventory.
`benchmarks/` holds analytical comparisons and their reference data.

## Before submitting a change

```sh
python -m ruff check .
python scripts/sync_metadata.py --check
python scripts/generate_api_reference.py --check
python scripts/check_documentation.py
python -m pytest
```

Use the interpreter from `.venv`. On Windows it is `.venv/Scripts/python.exe`.
See [test selection](tests/README.rst) for faster checks and runtime prerequisites.
Run analytical benchmarks when changing assembly or convergence behavior.

Version and dependencies are maintained in `pyproject.toml`. Regenerate metadata
with `scripts/sync_metadata.py`; update `uv.lock` with `uv lock` when dependencies
change. Regenerate API references after changing public signatures. The generator
requires an explicit description for every documented parameter.

## Adding or extending a solver

1. Define its physical coordinate axes explicitly and use the shared material,
   shape, and numeric validation contracts.
2. Implement geometry-change invalidation and mesh/result lifecycle. Keep
   numerical operators specific to the formulation.
3. Return a typed result with mesh, diagnostics, plotting, and atomic persistence.
   Add a distinct archive family/representation where physics requires it.
4. Add public exports to `tests/public_api.json`, then generate the API reference.
5. Add lifecycle, malformed-input, save/load, and analytical regression cases.
6. Add a small example with an explicit backend, a guide, and an entry in the
   [solver table](docs/README.md).
7. Register a new package in the wheel configuration and verify imports from an
   installed wheel outside the checkout.

A new backend must report unsupported features explicitly. Share infrastructure
when behavior is the same; document physical differences in the family guide.

Ruff checks syntax and high-confidence correctness rules across the repository.
Use `python -m ruff format <changed-files>` for new modules; keep large numerical
formatting-only changes separate from changes to numerical behavior.
