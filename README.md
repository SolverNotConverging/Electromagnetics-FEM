# FEM

Finite-element electromagnetic solvers for electrostatics, guided waves, periodic
structures and waveguide scattering. FEM uses conforming meshes and supports
material interfaces, full-vector fields and adaptive refinement. It is a
pure-Python package with shared CPU/GPU sparse solvers and an interactive
PySide6/PyVista result viewer.

```python
import fem

fr4 = fem.Material(name="FR-4", epsilon=4.4 * (1 - 0.02j))  # lossy dielectric, tan(delta) = 0.02
copper = fem.materials.copper                                # surface-impedance conductor

solver = fem.ModeSolver2D(frequency=10e9, x_range=(-6e-3, 6e-3), y_range=(-35e-6, 6e-3),
                          boundary=fem.materials.PEC)
solver.add_rectangle(x_range=(-6e-3, 6e-3), y_range=(0, 1.6e-3), name="substrate", material=fr4)
solver.add_rectangle(x_range=(-6e-3, 6e-3), y_range=(-35e-6, 0), name="ground", material=copper)
solver.add_rectangle(x_range=(-1.5e-3, 1.5e-3), y_range=(1.6e-3, 1.635e-3), name="strip", material=copper)
solver.mesh(max_element_size=0.5e-3)
modes = solver.solve(num_modes=1)

print(f"neff = {modes[0].neff:.4f}, loss = {modes[0].alpha * 8.686:.1f} dB/m")
modes.save("microstrip.h5")
modes.show()  # fields, mesh and material regions
```

This quasi-TEM microstrip mode has `neff ≈ 1.887 - 0.020j` and about 36 dB/m of
dielectric plus copper loss.

## Solvers

| Class | Problem | Guide | Examples |
|---|---|---|---|
| `fem.ElectrostaticSolver` | Potentials, fields, charges and energies with dielectric interfaces and charge densities | [Guide](docs/solvers/electrostatics.rst) | [Examples](examples/electrostatics/README.rst) |
| `fem.ModeSolver1D`, `fem.ModeSolver2D` | Guided modes and complex effective indices of slabs and cross-sections | [Guide](docs/solvers/waveguide.rst) | [Examples](examples/waveguide/README.rst) |
| `fem.PeriodicModeSolver2D`, `fem.PeriodicModeSolver3D` | Bloch modes of 2D/3D periodic cells, including leaky structures | [Guide](docs/solvers/periodic.rst) | [Examples](examples/periodic/README.rst) |
| `fem.ScatteringSolver2D` | Reflection, transmission, fields and radiation of 2D waveguide discontinuities | [Guide](docs/solvers/scattering.rst) | [Examples](examples/scattering/README.rst) |

## Install

```sh
python -m pip install -e .            # Python 3.11-3.13, Windows/Linux/macOS
python -m pip install -e ".[cuda12]"  # optional: GPU factorization with NVIDIA cuDSS (or cuda13)
```

Then run an example: `python examples/waveguide/1d_parallel_plate_waveguide.py`.

## Viewing results

`result.show()` opens one window for every solver: case, mode, component and
quantity selectors with **Mesh** and **Material** toggles. E and H components are
shown for every electromagnetic solver (potential, E and D for electrostatics);
3D periodic cells draw E and H as arrows with PyVista. Sweeps add S-parameter
spectra and scattering with an NF2FF contour adds far-field patterns. From a
terminal: `fem view result.h5`, `fem inspect result.h5`, `fem info`.

## Documentation

- [Getting started](docs/getting_started.md) and [conventions](docs/conventions.md)
- [Materials and shapes](docs/materials_and_shapes.rst), [adaptivity](docs/adaptivity.rst), [CPU/GPU acceleration](docs/acceleration.md)
- API reference in [docs/api](docs/api/), [changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md), [tests](tests/README.rst), [analytical benchmarks](benchmarks/README.md)

Build the HTML docs with `uv run --group docs sphinx-build -W docs docs/_build/html`.

## License

FEM is [MIT licensed](LICENSE). Dependencies retain their own licenses; Gmsh is
GPL-2.0-or-later and PySide6 is LGPL-3.0.
