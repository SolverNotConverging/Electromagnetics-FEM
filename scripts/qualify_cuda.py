"""Verify installed GPU operators on every visible device; write hardware evidence.

Run from a checkout with python -m scripts.qualify_cuda, or run this file with
an installed FEM environment. No checkout directory is added to sys.path.
"""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

from fem import __version__
from fem.common.gpu import solve_sparse
from fem.common.linalg import generalized_eigenpairs, quadratic_eigenpairs, resolve_backend


def qualify():
    import cupy as cp
    count = cp.cuda.runtime.getDeviceCount()
    if count == 0:
        raise RuntimeError("CUDA qualification requires at least one visible GPU.")
    devices = []
    for index in range(count):
        with cp.cuda.Device(index):
            assert resolve_backend() == "hybrid", "Automatic GPU readiness probe failed"
            n = 32
            a = diags((-.2j, np.arange(1., n+1)+1j, .1), (-1, 0, 1), shape=(n, n), format="csc")
            rhs = np.arange(n)+.5j
            actual = solve_sparse(a, rhs)
            np.testing.assert_allclose(actual, spsolve(a, rhs), rtol=1e-10, atol=1e-10)
            roots = np.arange(1., n+1)
            mass = np.ones(n)
            mass[-3:] = 0.
            values, vectors, method = generalized_eigenpairs(
                diags(roots), diags(mass), target=3.1, count=2, backend="gpu")
            np.testing.assert_allclose(np.sort(values.real), [3., 4.], atol=1e-8)
            generalized_residual = max(np.linalg.norm(roots*v-value*mass*v)
                                       for value, v in zip(values, vectors.T))
            values, vectors, method = quadratic_eigenpairs(
                diags(-roots**2), diags(np.zeros(n)), eye(n),
                target=3.1, count=2, backend="gpu")
            np.testing.assert_allclose(np.sort(values.real), [3., 4.], atol=1e-8)
            quadratic_residual = max(np.linalg.norm((-roots**2+value**2)*v)
                                     for value, v in zip(values, vectors.T))
            assert generalized_residual < 1e-8 and quadratic_residual < 1e-8
            properties = cp.cuda.runtime.getDeviceProperties(index)
            name = properties["name"]
            if isinstance(name, bytes):
                name = name.decode()
            devices.append(dict(device=index, name=name,
                compute_capability=cp.cuda.Device().compute_capability,
                generalized_residual=float(generalized_residual),
                quadratic_residual=float(quadratic_residual), status="passed"))
    return dict(fem=__version__, python=platform.python_version(), platform=platform.platform(),
                cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                cuda_driver=cp.cuda.runtime.driverGetVersion(), devices=devices,
                scope="Execution verified only on listed physical devices; architecture-policy tests are separate.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(qualify(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report+"\n", encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
