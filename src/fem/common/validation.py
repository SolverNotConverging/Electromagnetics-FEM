"""Numeric input contracts shared by solver families.

Accept Python and NumPy numeric scalars. Reject booleans, strings, complex
values for real controls, and arrays (including zero-dimensional arrays).
"""

from numbers import Complex, Integral, Real

import numpy as np

from .errors import ConfigurationError


def finite_real(value, name):
    """Return a finite real scalar, without parsing text or discarding phase."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be a finite real scalar.")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be a finite real scalar.") from exc
    if not np.isfinite(result):
        raise ConfigurationError(f"{name} must be a finite real scalar.")
    return result


def positive_real(value, name):
    """Return a finite, strictly positive real scalar."""
    result = finite_real(value, name)
    if result <= 0:
        raise ConfigurationError(f"{name} must be finite and positive.")
    return result


def positive_integer(value, name, minimum=1):
    """Return an exact integer at least minimum; integral floats are accepted."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ConfigurationError(f"{name} must be an integer of at least {minimum}.")
    if isinstance(value, Integral):
        result = int(value)
    else:
        numeric = finite_real(value, name)
        if not numeric.is_integer():
            raise ConfigurationError(f"{name} must be an integer of at least {minimum}.")
        result = int(numeric)
    if result < minimum:
        raise ConfigurationError(f"{name} must be an integer of at least {minimum}.")
    return result


def finite_complex(value, name):
    """Return a finite numeric complex scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Complex):
        raise ConfigurationError(f"{name} must be a finite complex scalar.")
    try:
        result = complex(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must be a finite complex scalar.") from exc
    if not np.isfinite(result):
        raise ConfigurationError(f"{name} must be a finite complex scalar.")
    return result


def real_array(value, name):
    """Return an owned finite real array, rejecting text, bools and complex data."""
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError("expected real numeric data")
        result = np.array(raw, dtype=float, copy=True)
        if not np.isfinite(result).all():
            raise ValueError("non-finite data")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ConfigurationError(f"{name} must contain finite real numbers.") from exc
    return result


def connectivity(value, node_count):
    """Validate exact nonnegative node indices before converting to int64."""
    try:
        raw = np.asarray(value)
        if raw.ndim != 2 or raw.dtype.kind not in "iuf":
            raise ValueError("expected a two-dimensional numeric array")
        if (
            not np.isfinite(raw).all()
            or np.any(raw < 0)
            or np.any(raw >= node_count)
            or np.any(raw != np.floor(raw))
        ):
            raise ValueError("invalid node index")
        return np.array(raw, dtype=np.int64, copy=True)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ConfigurationError(
            "Invalid mesh connectivity: expected exact integer node indices."
        ) from exc
