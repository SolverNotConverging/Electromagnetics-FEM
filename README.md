# FEM

Electromagnetic solvers. Each root solver folder contains its own `src/`, `docs/`, and `examples/`.

| Solver | Import | Documentation | Examples |
|---|---|---|---|
| electrostatics | `fem_electrostatics` | [Guide](fem_electrostatics/docs/guide.rst) | [Examples](fem_electrostatics/examples/README.rst) |
| periodic modes | `fem_periodic_modes` | [Guide](fem_periodic_modes/docs/guide.rst) | [Examples](fem_periodic_modes/examples/README.rst) |
| waveguide modes | `fem_waveguide_modes` | [Guide](fem_waveguide_modes/docs/guide.rst) | [Examples](fem_waveguide_modes/examples/README.rst) |
| waveguide scattering | `fem_waveguide_scattering` | [Guide](fem_waveguide_scattering/docs/guide.rst) | [Examples](fem_waveguide_scattering/examples/README.rst) |

## Run from the checkout

Install the Python dependencies, then run an example. No solver package installation is needed.

```sh
python -m pip install -r requirements.txt
python fem_waveguide_modes/examples/parallel_plate_waveguide_1d.py
```

You can also run examples as modules from the root:

```sh
python -m fem_waveguide_modes.examples.parallel_plate_waveguide_1d
```

Shared materials and geometry live in `cem_common/`; `periodic_eigensolver/` provides
the NumPy eigensolver and an optional Cython kernel. To compile the kernel in place:

```sh
python -m pip install "Cython>=3,<4" "setuptools>=77,<83"
python setup_cython.py build_ext --inplace
```

Without the compiled kernel, the default backend uses NumPy. A C compiler is needed only for this optional build.

Native FEM applications live under `apps/`; their README files describe standalone builds.

## Checks

```sh
python -m pip install -r requirements-dev.txt
python -m pytest
python scripts/check_documentation.py
python scripts/qualify_examples.py --import-only
```

`uv sync` can also create a dependency environment without installing the solvers.
Generated results are saved under ignored `outputs/`. Source is [MIT licensed](LICENSE).
