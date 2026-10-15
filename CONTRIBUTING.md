# Working on FEM

Use Python 3.12 and run `uv sync --locked` from the repository root. This installs
development dependencies and FEM in editable mode. See [getting started](docs/getting_started.md).

## Layout and ownership

Code lives in one package, `src/fem/`, with one subpackage per solver family
(`electrostatics`, `waveguide`, `periodic`, `scattering`). `fem.common` owns materials, shapes, input
validation, geometry handles, result contracts, and archive envelopes.
`fem.adaptivity` owns the solve/estimate/refine policy. Numerical assembly,
physics-specific diagnostics, and field recovery stay in each solver.

`fem/__init__.py` defines the public API and `fem/__main__.py` the `fem` command.
`fem/common/viewer.py` is the shared PySide6/PyVista viewer. `docs/` holds the
guides (`docs/solvers/`) and generated API pages (`docs/api/`); `examples/<family>/`
holds runnable scripts that use only the public API. `tests/` mirrors the
package layout and holds the public API inventory.
`benchmarks/` holds analytical comparisons and their reference data.

## Before submitting a change

```sh
python -m ruff check .
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

## Vocabulary

Use the same names in every solver family: `frequency` and `frequencies` (hertz),
`neff` (complex effective index), `beta` (rad/m), `alpha` (`-Im(beta)`, 1/m),
`num_modes`, `max_element_size`, `element_order`, `max_refinements`, `backend`,
`solve_info`, `mesh_data`. Results offer `plot()`, `show()` and `save()`; errors
derive from `FEMError`. Do not add aliases: keep one public name per concept.

## Writing examples

Examples are plain scripts that read top to bottom: build the solver, mesh,
solve, print a few numbers, save to `outputs/<example>/` next to the script and
call `show()`. Use only the public `fem` API, no `main()` function or
`if __name__ == "__main__"` guard, and no helper functions unless a loop needs
one. Tests never import examples; a test that needs the same device builds it
itself. `tests/test_examples.py` checks these rules and, with
`FEM_RUN_EXAMPLES=1`, runs every example and the README snippet.

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
   [solver table](docs/conventions.md).
7. Register a new package in the wheel configuration and verify imports from an
   installed wheel outside the checkout.

A new backend must report unsupported features explicitly. Share infrastructure
when behavior is the same; document physical differences in the family guide.

Ruff checks syntax and high-confidence correctness rules across the repository.
Use `python -m ruff format <changed-files>` for new modules; keep large numerical
formatting-only changes separate from changes to numerical behavior.
