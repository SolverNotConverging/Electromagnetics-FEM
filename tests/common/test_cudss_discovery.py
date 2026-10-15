"""cuDSS discovery works without mutating the process environment."""
import os
import pytest
from fem.common import _cudss


def test_explicit_library_wins(monkeypatch):
    monkeypatch.setenv("CUDSS_LIBRARY", "chosen-library")
    monkeypatch.setattr(_cudss.ctypes.util, "find_library", lambda name: "system-library")
    assert _cudss.find_cudss_library() == "chosen-library"


def test_nvidia_wheel_discovery(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDSS_LIBRARY", raising=False)
    monkeypatch.setattr(_cudss.ctypes.util, "find_library", lambda name: None)
    monkeypatch.setattr(_cudss.os, "get_exec_path", lambda: [])
    monkeypatch.setattr(_cudss.sys, "path", [str(tmp_path / "site-packages")])
    filename = "cudss64_0.dll" if os.name == "nt" else "libcudss.so.0"
    library = tmp_path / "site-packages/nvidia/cu12/bin" / filename
    library.parent.mkdir(parents=True)
    library.touch()
    assert _cudss.find_cudss_library() == str(library.resolve())


def test_missing_library_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDSS_LIBRARY", raising=False)
    monkeypatch.setattr(_cudss.ctypes.util, "find_library", lambda name: None)
    monkeypatch.setattr(_cudss.os, "get_exec_path", lambda: [])
    monkeypatch.setattr(_cudss.sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(_cudss.sys, "path", [])
    with pytest.raises(ImportError, match="CUDSS_LIBRARY"):
        _cudss.find_cudss_library()
