"""A small leaky controller used by homeostatic modulation."""

from __future__ import annotations

import math
import operator

import numpy as np


class HomeostaticState:
    """Track TD-error deviation from a setpoint with an exponential moving average.

    For error ``delta_t``, setpoint ``c``, and response rate ``eta``:

    ``x_{t+1} = x_t + eta * ((delta_t - c) - x_t)``

    The state starts at zero, meaning no deviation from the setpoint has yet
    been observed.
    """

    def __init__(self, setpoint: float = 0.0, eta: float = 0.05):
        setpoint = float(setpoint)
        eta = float(eta)
        if not math.isfinite(setpoint):
            raise ValueError("setpoint must be finite")
        if not math.isfinite(eta) or not 0.0 < eta <= 1.0:
            raise ValueError("eta must satisfy 0 < eta <= 1")
        self.setpoint = setpoint
        self.eta = eta
        self._state: np.ndarray | None = None

    def reset(self, n_seeds: int) -> None:
        """Reset the controller for ``n_seeds`` parallel runs."""
        try:
            n_seeds = operator.index(n_seeds)
        except TypeError as exc:
            raise TypeError("n_seeds must be an integer") from exc
        if n_seeds < 1:
            raise ValueError("n_seeds must be >= 1")
        self._state = np.zeros(n_seeds, dtype=np.float64)

    @property
    def value(self) -> np.ndarray:
        """Return the live state array.

        Callers may read this array but must not modify it.
        """
        if self._state is None:
            raise RuntimeError("reset() or update() must be called before reading the state")
        return self._state

    def update(self, td_error: np.ndarray, *, copy: bool = True) -> np.ndarray:
        """Incorporate one vector of TD errors and return the updated state."""
        td_error = np.asarray(td_error, dtype=np.float64)
        if td_error.ndim != 1 or td_error.size == 0:
            raise ValueError("td_error must be a non-empty one-dimensional array")
        if not np.isfinite(td_error).all():
            raise ValueError("td_error must contain only finite values")
        if self._state is None:
            self.reset(td_error.size)
        elif self._state.shape != td_error.shape:
            raise ValueError("td_error shape does not match the initialized state")

        state = self._state
        assert state is not None
        state += self.eta * ((td_error - self.setpoint) - state)
        return state.copy() if copy else state
