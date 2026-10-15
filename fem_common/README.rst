FEM common
==========

Python modules live directly in this package directory. Install only the Python dependencies from the
`root README <../README.md>`_, then import this folder directly.

* `Guide <docs/guide.rst>`_
* `API reference <docs/API_REFERENCE.rst>`_

Shared solver backends
----------------------

``linalg.py`` selects ``auto`` (default), ``cpu``, or ``gpu`` and implements
standard ARPACK shift-invert iteration. ``gpu.py`` contains shared GPU operators
and sparse solves; ``_cudss.py`` binds the cuDSS runtime. All solver families use
these modules. Auto tries the GPU runtime and uses SciPy CPU when unavailable.
