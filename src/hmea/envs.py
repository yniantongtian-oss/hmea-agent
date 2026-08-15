"""A vectorized chain environment for controlled Q-learning experiments."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable

import numpy as np

_NOISE_BLOCK = 65_536


def _integer(name: str, value: object, *, minimum: int) -> int:
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


class ChainMDP:
    """A deterministic left/right chain with optional zero-mean reward noise.

    Entering the rightmost state yields a reward of one and ends the episode.
    Every other transition has a base reward of zero. ``sigma_r`` controls
    independent Gaussian noise on non-terminal rewards.
    """

    def __init__(
        self,
        n_states: int = 8,
        gamma: float = 0.99,
        sigma_r: float = 0.1,
        shaping: Callable[[int, int], float] | None = None,
    ):
        self.n_states = _integer("n_states", n_states, minimum=2)
        gamma = float(gamma)
        sigma_r = float(sigma_r)
        if not math.isfinite(gamma) or not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        if not math.isfinite(sigma_r) or sigma_r < 0.0:
            raise ValueError("sigma_r must be finite and >= 0")
        if shaping is not None and not callable(shaping):
            raise TypeError("shaping must be callable or None")

        self.gamma = gamma
        self.sigma_r = sigma_r
        self.shaping = shaping
        self._last = self.n_states - 1

        states = np.arange(self.n_states)
        right = np.minimum(states + 1, self._last)
        left = np.maximum(states - 1, 0)
        left[self._last] = self._last
        transitions = np.stack([left, right], axis=1)
        rewards = np.zeros((self.n_states, 2), dtype=np.float64)
        rewards[:-1, 1] = right[:-1] == self._last
        terminal = np.zeros((self.n_states, 2), dtype=bool)
        terminal[self._last, :] = True
        terminal[:-1, 1] |= right[:-1] == self._last

        self._trans_flat = transitions.ravel()
        self._rew_flat = rewards.ravel()
        self._term_flat = terminal.ravel()
        self.rng = np.random.default_rng()

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, value: np.random.Generator) -> None:
        if not isinstance(value, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        self._rng = value
        self._noise_buffer: np.ndarray | None = None
        self._noise_pointer = 0

    def reset(self, n_seeds: int) -> np.ndarray:
        """Return state zero for each parallel run."""
        return np.zeros(_integer("n_seeds", n_seeds, minimum=1), dtype=np.int64)

    def step(
        self, states: np.ndarray, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply one vectorized transition and return next state, reward, and done."""
        states = np.asarray(states)
        actions = np.asarray(actions)
        if states.ndim != 1 or actions.ndim != 1 or states.shape != actions.shape:
            raise ValueError("states and actions must be one-dimensional arrays of equal shape")
        if states.size == 0:
            raise ValueError("states and actions must not be empty")
        if not np.issubdtype(states.dtype, np.integer):
            raise TypeError("states must contain integers")
        if not np.issubdtype(actions.dtype, np.integer):
            raise TypeError("actions must contain integers")
        if np.any((states < 0) | (states >= self.n_states)):
            raise ValueError("states contain an out-of-range index")
        if np.any((actions < 0) | (actions > 1)):
            raise ValueError("actions must be 0 or 1")

        columns = states + states + actions
        next_states = self._trans_flat[columns]
        done = self._term_flat[columns]
        rewards = self._rew_flat[columns]
        if self.shaping is not None:
            shaped = np.fromiter(
                (self.shaping(int(a), int(b)) for a, b in zip(states, next_states, strict=True)),
                dtype=np.float64,
                count=states.size,
            )
            if not np.isfinite(shaped).all():
                raise ValueError("shaping must return finite values")
            rewards = rewards + shaped
        if self.sigma_r > 0.0:
            noise_buffer = self._noise_buffer
            if noise_buffer is None or self._noise_pointer + states.size > noise_buffer.size:
                noise_buffer = self.rng.normal(0.0, self.sigma_r, _NOISE_BLOCK)
                self._noise_buffer = noise_buffer
                self._noise_pointer = 0
            start = self._noise_pointer
            self._noise_pointer += states.size
            rewards += noise_buffer[start : start + states.size] * ~done
        return next_states, rewards, done

    def optimal_q(self) -> np.ndarray:
        """Return the expected optimal action values from value iteration.

        Reward noise is zero-mean. For episodic potential shaping, use a
        potential of zero at the terminal state.
        """
        n_states = self.n_states
        last = n_states - 1
        states = np.arange(n_states)
        right = np.minimum(states + 1, last)
        left = np.maximum(states - 1, 0)
        left[last] = last
        reward_right = ((right == last) & (states < last)).astype(np.float64)
        reward_left = ((left == last) & (states < last)).astype(np.float64)

        if self.shaping is not None:
            shape_right = np.fromiter(
                (self.shaping(int(a), int(b)) for a, b in zip(states, right, strict=True)),
                dtype=np.float64,
                count=n_states,
            )
            shape_left = np.fromiter(
                (self.shaping(int(a), int(b)) for a, b in zip(states, left, strict=True)),
                dtype=np.float64,
                count=n_states,
            )
            if not (np.isfinite(shape_right).all() and np.isfinite(shape_left).all()):
                raise ValueError("shaping must return finite values")
            reward_right += shape_right
            reward_left += shape_left

        values = np.zeros(n_states, dtype=np.float64)
        while True:
            q_values = np.stack(
                [
                    reward_left + self.gamma * values[left],
                    reward_right + self.gamma * values[right],
                ],
                axis=1,
            )
            updated = q_values.max(axis=1)
            updated[last] = 0.0
            if np.max(np.abs(updated - values)) < 1e-12:
                values = updated
                break
            values = updated

        result = np.stack(
            [
                reward_left + self.gamma * values[left],
                reward_right + self.gamma * values[right],
            ],
            axis=1,
        )
        result[last, :] = 0.0
        return result
