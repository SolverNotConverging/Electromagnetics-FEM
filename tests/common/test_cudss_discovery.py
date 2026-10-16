"""cuDSS discovery works without mutating the process environment."""
import os
import sys
from types import SimpleNamespace
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


@pytest.mark.skipif(os.name != "nt", reason="Windows NVRTC DLL discovery")
def test_windows_nvidia_wheel_nvrtc_companion(tmp_path, monkeypatch):
    runtime = tmp_path / "nvidia/cu13"
    binary = runtime / "bin/x86_64"
    binary.mkdir(parents=True)
    builtin = binary / "nvrtc-builtins64_134.dll"
    builtin.touch()
    cudss = runtime / "bin/cudss64_0.dll"
    cudss.touch()
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        cuda=SimpleNamespace(get_cuda_path=lambda: str(runtime))))
    monkeypatch.setattr(_cudss, "find_cudss_library", lambda: str(cudss))
    monkeypatch.setattr(_cudss, "_dll_directories", {})
    monkeypatch.setattr(_cudss, "_compiler_libraries", {})
    registered, loaded = [], []
    monkeypatch.setattr(_cudss.os, "add_dll_directory", lambda path: registered.append(path))
    monkeypatch.setattr(_cudss.ct, "WinDLL", lambda path: loaded.append(path))
    _cudss.configure_dll_search()
    assert str(binary) in registered
    assert str(builtin) in loaded
