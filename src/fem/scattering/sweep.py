"""Frequency-sweep scattering observables and lazy HDF5 persistence."""

from __future__ import annotations

from fem.common.contracts import ResultMixin

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


def _result_frequency_hz(value: object, index: int) -> float:
    """Validate optional per-result frequency metadata without truncation."""

    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"results[{index}].frequency must be a finite positive real scalar."
        ) from exc
    if (
        array.shape != ()
        or np.iscomplexobj(array)
        or isinstance(value, (bool, str, bytes))
    ):
        raise ValueError(
            f"results[{index}].frequency must be a finite positive real scalar."
        )
    try:
        frequency = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"results[{index}].frequency must be a finite positive real scalar."
        ) from exc
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError(
            f"results[{index}].frequency must be a finite positive real scalar."
        )
    return frequency


@dataclass(frozen=True, slots=True)
class ScatteringSweepResult(ResultMixin):
    """Ordered scattering results evaluated at increasing frequencies.

    Parameters
    ----------
    frequencies:
        Strictly increasing positive ordinary frequencies in hertz.
    results:
        One result per frequency.  Each result is expected to expose ``S`` and
        the scalar power-ratio properties used below.  If a result also exposes
        ``frequency``, that value is checked against the corresponding sweep
        frequency.
    h5_path:
        Optional path associated with an already persisted sweep.  Calling
        :meth:`save` returns the written path and does not mutate this frozen
        object.
    """

    frequencies: FloatArray
    results: tuple[Any, ...]
    h5_path: Path | None = None

    @property
    def metadata(self):
        from types import MappingProxyType
        return MappingProxyType({"time_convention": "exp(+i*omega*t)", "field_representation": "exp(-i*ky*y)", "result_kind": "frequency-sweep"})

    @property
    def solve_info(self):
        return tuple(case.solve_info for case in self.results)

    @property
    def mesh_data(self):
        return tuple(case.mesh_data for case in self.results)

    def plot(self, *, component: str | None = None, quantity: str = "magnitude"):
        """Return an S11/S21 spectrum figure; frequency is measured in hertz."""
        return self._result_api().plot_result(self, component=component, quantity=quantity)

    def __post_init__(self) -> None:
        from .result_api import _LazyScatteringCases
        if isinstance(self.results, _LazyScatteringCases):
            frequencies = np.array(self.frequencies, dtype=float, copy=True)
            if frequencies.ndim != 1 or not len(frequencies) or not np.isfinite(frequencies).all() or np.any(frequencies <= 0) or np.any(np.diff(frequencies) <= 0):
                raise ValueError("frequencies must be finite, positive, and increasing.")
            frequencies.flags.writeable = False
            object.__setattr__(self, "frequencies", frequencies)
            return
        try:
            raw_frequencies = np.asarray(self.frequencies)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "frequencies must contain finite positive real values."
            ) from exc
        if raw_frequencies.ndim != 1 or raw_frequencies.size == 0:
            raise ValueError("frequencies must be a nonempty one-dimensional array.")
        if np.iscomplexobj(raw_frequencies) or raw_frequencies.dtype.kind == "b":
            raise ValueError("frequencies must contain finite positive real values.")
        try:
            frequencies = np.array(raw_frequencies, dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "frequencies must contain finite positive real values."
            ) from exc
        if not np.isfinite(frequencies).all() or np.any(frequencies <= 0.0):
            raise ValueError("frequencies must contain finite positive real values.")
        if np.any(np.diff(frequencies) <= 0.0):
            raise ValueError("frequencies must be strictly increasing.")

        try:
            results = tuple(self.results)
        except TypeError as exc:
            raise ValueError(
                "results must be an iterable with one entry per frequency."
            ) from exc
        if len(results) != frequencies.size:
            raise ValueError(
                "results must contain exactly one entry for each frequency; "
                f"received {len(results)} result(s) for {frequencies.size} frequencies."
            )

        for index, (frequency, result) in enumerate(
            zip(frequencies, results, strict=True)
        ):
            if not hasattr(result, "frequency"):
                continue
            result_frequency = _result_frequency_hz(result.frequency, index)
            if not np.isclose(
                result_frequency,
                frequency,
                rtol=1e-12,
                atol=0.0,
            ):
                raise ValueError(
                    f"results[{index}].frequency={result_frequency:.16g} does not "
                    f"match frequencies[{index}]={frequency:.16g}."
                )

        frequencies.setflags(write=False)
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "results", results)
        if self.h5_path is not None:
            try:
                path = Path(self.h5_path)
            except TypeError as exc:
                raise ValueError("h5_path must be path-like or None.") from exc
            object.__setattr__(self, "h5_path", path)

    def S(
        self,
        side: str,
        *,
        out_mode: int = 0,
        in_mode: int = 0,
    ) -> ComplexArray:
        """Return one indexed modal amplitude across the sweep."""

        return np.asarray(
            [
                result.S(side, out_mode=out_mode, in_mode=in_mode)
                for result in self.results
            ],
            dtype=np.complex128,
        )

    @property
    def S11(self) -> ComplexArray:
        """Fundamental reflected modal amplitude at every frequency."""

        return self.S("left", out_mode=0, in_mode=0)

    @property
    def S21(self) -> ComplexArray:
        """Fundamental transmitted modal amplitude at every frequency."""

        return self.S("right", out_mode=0, in_mode=0)

    @property
    def reflection(self) -> FloatArray:
        """Total reflected-power ratio at every frequency."""

        return np.asarray(
            [result.reflection for result in self.results], dtype=np.float64
        )

    @property
    def transmission(self) -> FloatArray:
        """Total transmitted-power ratio at every frequency."""

        return np.asarray(
            [result.transmission for result in self.results], dtype=np.float64
        )

    @property
    def power_balance_error(self) -> FloatArray:
        """Dimensionless power-balance error at every frequency."""

        return np.asarray(
            [result.power_balance_error for result in self.results],
            dtype=np.float64,
        )

    @property
    def incident_power(self) -> FloatArray:
        """Incident modal power at every frequency."""

        return np.asarray(
            [result.incident_power for result in self.results], dtype=np.float64
        )

    @property
    def radiated_power(self) -> FloatArray:
        """Outward radiated power at every frequency."""

        return np.asarray(
            [result.radiated_power for result in self.results], dtype=np.float64
        )

    @property
    def absorbed_power(self) -> FloatArray:
        """Material-absorbed power at every frequency."""

        return np.asarray(
            [result.absorbed_power for result in self.results], dtype=np.float64
        )

    @property
    def power_balance(self) -> FloatArray:
        """Accounted output-power fraction at every frequency."""

        return np.asarray(
            [result.power_balance for result in self.results], dtype=np.float64
        )





__all__ = ["ScatteringSweepResult"]
