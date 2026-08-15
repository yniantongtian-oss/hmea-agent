"""Update multipliers applied to temporal-difference errors."""

from __future__ import annotations

import operator
from typing import Protocol, runtime_checkable

import numpy as np

from .homeostasis import HomeostaticState


def _seed_count(value: object) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError("n_seeds must be an integer") from exc
    if result < 1:
        raise ValueError("n_seeds must be >= 1")
    return result


@runtime_checkable
class Modulator(Protocol):
    """Interface implemented by update-multiplier policies."""

    name: str
    lag: int
    bounds: tuple[float, float]
    uses_current_td_error: bool

    def reset(self, n_seeds: int) -> None:
        """Reset state for ``n_seeds`` parallel runs."""
        ...

    def multiplier(self, td_error: np.ndarray, step: int) -> np.ndarray:
        """Return one multiplier per seed for the current update."""
        ...


class _BoundedModulationBase:
    """Parameter handling shared by bounded tanh-based modulators."""

    name = "base"
    lag = 0
    uses_current_td_error = True

    def __init__(self, beta: float, tau: float, bounds: tuple[float, float]):
        beta = float(beta)
        tau = float(tau)
        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be finite and > 0")
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau must be finite and > 0")
        if len(bounds) != 2:
            raise ValueError("bounds must contain exactly two values")
        lo, hi = float(bounds[0]), float(bounds[1])
        if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo <= hi):
            raise ValueError("bounds must satisfy 0 < lower <= upper < infinity")

        self.beta = beta
        self.tau = tau
        self.bounds = (lo, hi)
        self._inv_tau = 1.0 / tau
        self._lo = lo
        self._hi = hi
        self._out: np.ndarray | None = None

    def reset(self, n_seeds: int) -> None:
        n_seeds = _seed_count(n_seeds)
        self._out = np.empty(n_seeds, dtype=np.float64)

    def _buffer(self, n_seeds: int) -> np.ndarray:
        out = self._out
        if out is None or out.shape != (n_seeds,):
            self.reset(n_seeds)
            out = self._out
        assert out is not None
        return out

    def _transform(self, signal: np.ndarray) -> np.ndarray:
        out = self._buffer(signal.shape[0])
        np.multiply(signal, self._inv_tau, out=out)
        np.tanh(out, out=out)
        out *= self.beta
        out += 1.0
        np.clip(out, self._lo, self._hi, out=out)
        return out


class NoModulation:
    """Plain Q-learning: return a multiplier of one for every update."""

    name = "none"
    lag = 0
    bounds = (1.0, 1.0)
    uses_current_td_error = False

    def reset(self, n_seeds: int) -> None:
        n_seeds = _seed_count(n_seeds)
        self._ones = np.ones(n_seeds, dtype=np.float64)

    def multiplier(self, td_error: np.ndarray, step: int) -> np.ndarray:
        n_seeds = np.asarray(td_error).shape[0]
        ones = getattr(self, "_ones", None)
        if ones is None or ones.shape != (n_seeds,):
            self.reset(n_seeds)
            ones = self._ones
        return ones


class OnlineModulation(_BoundedModulationBase):
    """Compute the update multiplier from the current TD error."""

    name = "online"
    lag = 0
    uses_current_td_error = True

    def __init__(
        self,
        beta: float = 1.5,
        tau: float = 0.5,
        bounds: tuple[float, float] = (0.5, 2.0),
    ):
        super().__init__(beta, tau, bounds)

    def multiplier(self, td_error: np.ndarray, step: int) -> np.ndarray:
        return self._transform(td_error)


class ClippedModulation(_BoundedModulationBase):
    """Online modulation constrained to a narrower multiplier interval."""

    name = "clipped"
    lag = 0
    uses_current_td_error = True

    def __init__(
        self,
        beta: float = 0.4,
        tau: float = 0.5,
        bounds: tuple[float, float] = (0.8, 1.2),
    ):
        super().__init__(beta, tau, bounds)

    def multiplier(self, td_error: np.ndarray, step: int) -> np.ndarray:
        return self._transform(td_error)


class LaggedModulation(_BoundedModulationBase):
    """Compute the multiplier from the TD error observed ``lag`` steps earlier."""

    name = "lagged"
    uses_current_td_error = False

    def __init__(
        self,
        lag: int = 8,
        beta: float = 1.5,
        tau: float = 0.5,
        bounds: tuple[float, float] = (0.5, 2.0),
    ):
        if isinstance(lag, bool) or int(lag) != lag or int(lag) < 1:
            raise ValueError("lag must be an integer >= 1")
        super().__init__(beta, tau, bounds)
        self.lag = int(lag)
        self._history: list[np.ndarray] | None = None
        self._pointer = 0
        self._seen = 0

    def reset(self, n_seeds: int) -> None:
        n_seeds = _seed_count(n_seeds)
        super().reset(n_seeds)
        self._history = [np.zeros(n_seeds, dtype=np.float64) for _ in range(self.lag)]
        self._pointer = 0
        self._seen = 0

    def multiplier(self, td_error: np.ndarray, step: int) -> np.ndarray:
        history = self._history
        if history is None or history[0].shape != td_error.shape:
            self.reset(td_error.shape[0])
            history = self._history
        assert history is not None

        pointer = self._pointer
        if self._seen < self.lag:
            out = self._buffer(td_error.shape[0])
            out.fill(1.0)
        else:
            out = self._transform(history[pointer])

        np.copyto(history[pointer], td_error)
        self._pointer = (pointer + 1) % self.lag
        self._seen += 1
        return out


class HomeostaticModulation(_BoundedModulationBase):
    """Use a leaky TD-error state from the previous step as the multiplier signal."""

    name = "homeostatic"
    lag = 1
    uses_current_td_error = False

    def __init__(
        self,
        beta: float = 0.8,
        tau: float = 0.5,
        bounds: tuple[float, float] = (0.5, 1.5),
        setpoint: float = 0.0,
        eta: float = 0.05,
    ):
        super().__init__(beta, tau, bounds)
        self.state = HomeostaticState(setpoint=setpoint, eta=eta)

    def reset(self, n_seeds: int) -> None:
        super().reset(n_seeds)
        self.state.reset(n_seeds)

    def multiplier(self, td_error: np.ndarray, step: int) -> np.ndarray:
        try:
            signal = self.state.value
        except RuntimeError:
            self.reset(td_error.shape[0])
            signal = self.state.value
        if signal.shape != td_error.shape:
            raise ValueError("td_error shape does not match the initialized modulator")

        out = self._transform(signal)
        self.state.update(td_error, copy=False)
        return out
