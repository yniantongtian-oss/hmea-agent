"""Vectorized tabular Q-learning with configurable update multipliers."""

from __future__ import annotations

import math
import operator
from typing import TypedDict

import numpy as np

from .envs import ChainMDP
from .modulation import Modulator
from .validation import ModulatorAudit, validate_modulator

_RNG_BLOCK = 8_192


class TrainLog(TypedDict):
    """Arrays returned by :meth:`QAgent.train`."""

    steps: np.ndarray
    bias: np.ndarray
    bias_by_seed: np.ndarray
    rmse: np.ndarray
    rmse_by_seed: np.ndarray
    q_start: np.ndarray
    q_start_by_seed: np.ndarray
    q_values: np.ndarray
    audit: ModulatorAudit


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be >= 1")
    return result


class QAgent:
    """Tabular Q-learning whose TD update is scaled by a modulator.

    ``Q(s, a) <- Q(s, a) + alpha * m_t * delta_t``
    """

    def __init__(
        self,
        env: ChainMDP,
        modulator: Modulator,
        alpha: float = 0.1,
        eps0: float = 0.3,
        eps_min: float = 0.02,
        eps_decay: float = 0.9995,
        seed: int = 0,
    ):
        if not isinstance(env, ChainMDP):
            raise TypeError("env must be a ChainMDP")
        validate_modulator(modulator)
        alpha = float(alpha)
        eps0 = float(eps0)
        eps_min = float(eps_min)
        eps_decay = float(eps_decay)
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be finite and > 0")
        if not math.isfinite(eps0) or not 0.0 <= eps0 <= 1.0:
            raise ValueError("eps0 must satisfy 0 <= eps0 <= 1")
        if not math.isfinite(eps_min) or not 0.0 <= eps_min <= eps0:
            raise ValueError("eps_min must satisfy 0 <= eps_min <= eps0")
        if not math.isfinite(eps_decay) or not 0.0 < eps_decay <= 1.0:
            raise ValueError("eps_decay must satisfy 0 < eps_decay <= 1")
        if isinstance(seed, (bool, np.bool_)):
            raise TypeError("seed must be an integer")
        try:
            seed = operator.index(seed)
        except TypeError as exc:
            raise TypeError("seed must be an integer") from exc
        if seed < 0:
            raise ValueError("seed must be >= 0")

        self.env = env
        self.modulator = modulator
        self.alpha = alpha
        self.eps0 = eps0
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.seed = seed

    def train(self, n_steps: int, n_seeds: int = 64, log_every: int = 500) -> TrainLog:
        """Train independent seeded runs in parallel and return aggregate diagnostics.

        The final step is always logged, even when ``n_steps`` is not divisible
        by ``log_every``. Per-seed arrays have shape ``(n_logs, n_seeds)``.
        ``multiplier`` must return one finite value per seed. If it returns the
        TD-error array itself, that array is copied before the in-place update.
        """
        n_steps = _positive_integer("n_steps", n_steps)
        n_seeds = _positive_integer("n_seeds", n_seeds)
        log_every = _positive_integer("log_every", log_every)
        report = validate_modulator(self.modulator)

        env = self.env
        n_states = env.n_states
        gamma = env.gamma
        q_star = env.optimal_q()

        q_values = np.zeros((n_seeds, n_states, 2), dtype=np.float64)
        q_flat = q_values.ravel()
        q_flat_shifted = q_flat[1:]

        children = np.random.SeedSequence(self.seed).spawn(n_seeds + 1)
        rngs = [np.random.default_rng(child) for child in children[:n_seeds]]
        env.rng = np.random.default_rng(children[n_seeds])
        self.modulator.reset(n_seeds)

        states = env.reset(n_seeds)
        epsilon = self.eps0
        index = np.arange(n_seeds)
        row_offset = index * (n_states * 2)
        base = row_offset + states + states

        n_logs = math.ceil(n_steps / log_every)
        bias_by_seed = np.empty((n_logs, n_seeds), dtype=np.float64)
        rmse_by_seed = np.empty((n_logs, n_seeds), dtype=np.float64)
        q_start_by_seed = np.empty((n_logs, n_seeds), dtype=np.float64)
        steps = np.empty(n_logs, dtype=np.int64)
        log_index = 0

        random_uniform = np.empty((_RNG_BLOCK, n_seeds))
        random_actions = np.empty((_RNG_BLOCK, n_seeds), dtype=np.int64)
        explore = np.empty((_RNG_BLOCK, n_seeds), dtype=bool)
        block_index = np.arange(_RNG_BLOCK)
        actions = np.empty(n_seeds, dtype=np.int64)

        transitions = env._trans_flat
        terminal = env._term_flat
        base_rewards = env._rew_flat
        shaping = env.shaping
        sigma_r = env.sigma_r
        multiplier = self.modulator.multiplier

        block_start = -_RNG_BLOCK
        until_log = log_every
        for step in range(n_steps):
            within_block = step - block_start
            if within_block == _RNG_BLOCK:
                for seed_index, generator in enumerate(rngs):
                    random_uniform[:, seed_index] = generator.random(_RNG_BLOCK)
                    random_actions[:, seed_index] = generator.integers(0, 2, _RNG_BLOCK)
                epsilon_block = epsilon * self.eps_decay**block_index
                np.maximum(epsilon_block, self.eps_min, out=epsilon_block)
                np.less(random_uniform, epsilon_block[:, None], out=explore)
                epsilon = max(epsilon_block[-1] * self.eps_decay, self.eps_min)
                block_start = step
                within_block = 0

            explore_now = explore[within_block]
            random_now = random_actions[within_block]
            greedy = q_flat_shifted[base] > q_flat[base]
            np.copyto(actions, greedy)
            np.copyto(actions, random_now, where=explore_now)

            selected = base + actions
            columns = selected - row_offset
            next_states = transitions[columns]
            done = terminal[columns]
            rewards = base_rewards[columns]
            if shaping is not None:
                current_states = (base - row_offset) // 2
                shaped = np.fromiter(
                    (
                        shaping(int(current), int(next_state))
                        for current, next_state in zip(current_states, next_states, strict=True)
                    ),
                    dtype=np.float64,
                    count=n_seeds,
                )
                if not np.isfinite(shaped).all():
                    raise ValueError("shaping must return finite values")
                rewards = rewards + shaped
            if sigma_r > 0.0:
                rewards += env._reward_noise(n_seeds) * ~done

            next_base = row_offset + next_states + next_states
            old_values = q_flat[selected]
            td_error = np.maximum(q_flat[next_base], q_flat_shifted[next_base])
            td_error[done] = 0.0
            td_error *= gamma
            td_error += rewards
            td_error -= old_values
            raw_multiplier = multiplier(td_error, step)
            aliased = raw_multiplier is td_error
            try:
                multiplier_values = np.asarray(raw_multiplier, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("multiplier must return one finite value per seed") from exc
            if aliased:
                multiplier_values = multiplier_values.copy()
            if multiplier_values.shape != (n_seeds,) or not np.isfinite(multiplier_values).all():
                raise ValueError("multiplier must return one finite value per seed")
            td_error *= self.alpha
            td_error *= multiplier_values
            td_error += old_values
            q_flat[selected] = td_error

            np.copyto(next_base, row_offset, where=done)
            base = next_base

            until_log -= 1
            if until_log == 0 or step + 1 == n_steps:
                error = q_values - q_star
                bias_by_seed[log_index] = error.mean(axis=(1, 2))
                rmse_by_seed[log_index] = np.sqrt(np.square(error).mean(axis=(1, 2)))
                q_start_by_seed[log_index] = q_values[:, 0, :].max(axis=1)
                steps[log_index] = step + 1
                log_index += 1
                until_log = log_every

        return {
            "steps": steps,
            "bias": bias_by_seed.mean(axis=1),
            "bias_by_seed": bias_by_seed,
            "rmse": rmse_by_seed.mean(axis=1),
            "rmse_by_seed": rmse_by_seed,
            "q_start": q_start_by_seed.mean(axis=1),
            "q_start_by_seed": q_start_by_seed,
            "q_values": q_values.copy(),
            "audit": report,
        }
