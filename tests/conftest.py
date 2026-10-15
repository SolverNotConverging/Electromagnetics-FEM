"""Keep ordinary numerical regression tests independent of optional GPU installs."""
import os
import pytest

@pytest.fixture(autouse=True, scope="session")
def cpu_regression_environment():
    from fem_common import gpu
    with pytest.MonkeyPatch.context() as patch:
        if os.environ.get("FEM_TEST_CUDA") != "1":
            patch.setattr(gpu, "is_available", lambda: False)
        yield
