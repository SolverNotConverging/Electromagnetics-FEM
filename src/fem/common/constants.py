"""SI constants (CODATA via SciPy) and the time-harmonic convention.

Fields use ``exp(+i*omega*t)``; guided propagation is ``exp(-i*beta*z)``.
Passive materials therefore have non-positive imaginary permittivity and
permeability.
"""

from scipy.constants import c as _c
from scipy.constants import epsilon_0 as _epsilon_0
from scipy.constants import mu_0 as _mu_0

C_0 = float(_c)
"""Vacuum speed of light in metres per second."""

EPSILON_0 = float(_epsilon_0)
"""Vacuum permittivity in farads per metre."""

MU_0 = float(_mu_0)
"""Vacuum permeability in henries per metre."""

ETA_0 = (MU_0 / EPSILON_0) ** 0.5
"""Vacuum wave impedance in ohms."""

TIME_CONVENTION = "exp(+i*omega*t)"

__all__ = ["C_0", "EPSILON_0", "ETA_0", "MU_0", "TIME_CONVENTION"]
