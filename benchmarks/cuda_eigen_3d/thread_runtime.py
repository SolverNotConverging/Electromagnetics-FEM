"""Read configured thread counts from already-loaded Linux OpenBLAS libraries."""
from __future__ import annotations

import ctypes
from pathlib import Path


def loaded_openblas():
    maps = Path('/proc/self/maps')
    if not maps.exists():
        return []
    paths = {line.split(maxsplit=5)[-1].strip() for line in maps.read_text().splitlines()
             if 'openblas' in line and len(line.split(maxsplit=5)) == 6}
    result = []
    for path in sorted(paths):
        library = ctypes.CDLL(path)
        for prefix, suffix in (('scipy_openblas_', '64_'), ('scipy_openblas_', ''),
                               ('openblas_', '64_'), ('openblas_', '')):
            try:
                getter = getattr(library, prefix + 'get_num_threads' + suffix)
                config = getattr(library, prefix + 'get_config' + suffix)
            except AttributeError:
                continue
            getter.restype = ctypes.c_int
            config.restype = ctypes.c_char_p
            result.append(dict(library=path, configured_threads=getter(),
                               config=config().decode()))
            break
        else:
            result.append(dict(library=path, configured_threads=None))
    return result
