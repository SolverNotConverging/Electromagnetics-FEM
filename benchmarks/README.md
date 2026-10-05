# FEM benchmarks

Run analytical benchmarks after installing the project:

```sh
python benchmarks/analytical/coaxial_waveguide_adaptivity.py --check
python benchmarks/analytical/parallel_plate_electrostatics.py --check
python benchmarks/analytical/rectangular_waveguide_modes.py --check
python benchmarks/analytical/uniform_periodic_medium.py --check
```

Reports are written to ignored `outputs/benchmarks/analytical/`.
Shared periodic eigensolver performance benchmarks are in `periodic_eigensolver/`.
