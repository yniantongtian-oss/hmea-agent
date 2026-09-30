"""Paired uncertainty estimates for independent seeded experiment replicas."""

from __future__ import annotations

import operator
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike


class PairedDifference(TypedDict):
    """Mean candidate-minus-reference difference and a percentile interval."""

    n_pairs: int
    mean_difference: float
    ci95_low: float
    ci95_high: float
    observed_zero_variance: bool
    per_seed_differences: list[float]


def _integer(value: int, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def paired_mean_difference(
    candidate: ArrayLike,
    reference: ArrayLike,
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> PairedDifference:
    """Estimate a raw mean difference with a paired 95% percentile bootstrap.

    Each position must identify the same independent replica in both inputs.
    Resampling complete pairs preserves their dependence; separate resampling
    of the two inputs would estimate a different quantity. The interval is
    exploratory, especially for small samples, and is not a simultaneous
    interval across multiple comparisons. Constant observed differences give
    a collapsed interval, not evidence of zero population uncertainty.
    """
    n_resamples = _integer(n_resamples, name="n_resamples", minimum=1)
    seed = _integer(seed, name="seed", minimum=0)
    candidate_values = np.asarray(candidate, dtype=float)
    reference_values = np.asarray(reference, dtype=float)
    if candidate_values.ndim != 1 or reference_values.ndim != 1:
        raise ValueError("candidate and reference must be one-dimensional")
    if candidate_values.size != reference_values.size:
        raise ValueError("candidate and reference must have equal lengths")
    if candidate_values.size < 2:
        raise ValueError("paired comparison requires at least 2 observations")
    if not (np.isfinite(candidate_values).all() and np.isfinite(reference_values).all()):
        raise ValueError("candidate and reference must contain only finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        differences = candidate_values - reference_values
    if not np.isfinite(differences).all():
        raise ValueError("candidate-minus-reference differences must be finite")

    zero_variance = bool(np.all(differences == differences[0]))
    if zero_variance:
        mean_difference = low = high = float(differences[0])
    else:
        # Normalize to avoid overflow in both means and percentile interpolation.
        scale = float(np.max(np.abs(differences)))
        normalized = differences / scale
        mean_difference = float(normalized.mean() * scale)
        rng = np.random.default_rng(seed)
        means = np.empty(n_resamples, dtype=float)
        batch_size = max(1, min(n_resamples, 1_000_000 // differences.size))
        for start in range(0, n_resamples, batch_size):
            stop = min(start + batch_size, n_resamples)
            indices = rng.integers(0, differences.size, size=(stop - start, differences.size))
            means[start:stop] = normalized[indices].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975]) * scale
    if not np.isfinite([mean_difference, low, high]).all():
        raise ValueError("paired summary must be finite; rescale the observations")
    return {
        "n_pairs": int(differences.size),
        "mean_difference": mean_difference,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "observed_zero_variance": zero_variance,
        "per_seed_differences": differences.tolist(),
    }
