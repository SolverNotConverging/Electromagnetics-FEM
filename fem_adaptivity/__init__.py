"""Shared solve/estimate/refine policy and Maxwell interface residuals."""
from pathlib import Path as _Path
__path__.append(str(_Path(__file__).parent / "src"))
from .policy import (bulk_mark, cell_geometry, maxwell_jump_residual,
                     mixed_mode_fields, run_adaptive, validate_controls, __version__)
