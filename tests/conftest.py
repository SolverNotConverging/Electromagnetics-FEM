"""Run tests headless and on the SciPy CPU backend unless FEM_TEST_CUDA=1."""
import os

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True, scope="session")
def cpu_regression_environment():
    from fem.common import gpu
    with pytest.MonkeyPatch.context() as patch:
        if os.environ.get("FEM_TEST_CUDA") != "1":
            patch.setattr(gpu, "is_available", lambda: False)
        yield
