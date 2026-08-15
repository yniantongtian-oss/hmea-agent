"""Structural checks for custom modulation policies."""

from __future__ import annotations

import operator
from typing import TypedDict

import numpy as np


class ModulatorAudit(TypedDict):
    """Summary returned by :func:`audit`."""

    name: str
    bounded: bool
    lag: int
    uses_current_td_error: bool
    current_error_independent: bool


def check_bounded(modulator: object) -> bool:
    """Return whether declared multiplier bounds are positive and finite."""
    try:
        bounds = modulator.bounds
        if len(bounds) != 2:
            return False
        lower, upper = float(bounds[0]), float(bounds[1])
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(np.isfinite(lower) and np.isfinite(upper) and 0.0 < lower <= upper)


def check_current_error_independence(modulator: object) -> bool:
    """Return whether the policy declares that it ignores the current TD error."""
    return getattr(modulator, "uses_current_td_error", None) is False


def audit(modulator: object) -> ModulatorAudit:
    """Describe a modulator's declared structure without making performance claims."""
    raw_lag = getattr(modulator, "lag", None)
    try:
        lag = operator.index(raw_lag) if not isinstance(raw_lag, bool) else -1
    except TypeError:
        lag = -1
    raw_current = getattr(modulator, "uses_current_td_error", None)
    uses_current = raw_current if isinstance(raw_current, bool) else True
    return {
        "name": str(getattr(modulator, "name", type(modulator).__name__)),
        "bounded": check_bounded(modulator),
        "lag": lag,
        "uses_current_td_error": uses_current,
        "current_error_independent": not uses_current,
    }


def validate_modulator(modulator: object) -> ModulatorAudit:
    """Validate the attributes required by :class:`hmea.Modulator`.

    This checks interface declarations. It cannot prove that a custom
    implementation always respects its declared bounds or signal timing.
    """
    report = audit(modulator)
    problems: list[str] = []
    if not callable(getattr(modulator, "reset", None)):
        problems.append("reset must be callable")
    if not callable(getattr(modulator, "multiplier", None)):
        problems.append("multiplier must be callable")
    if not isinstance(getattr(modulator, "name", None), str) or not modulator.name.strip():
        problems.append("name must be a non-empty string")
    if not isinstance(getattr(modulator, "uses_current_td_error", None), bool):
        problems.append("uses_current_td_error must be a boolean")
    if not report["bounded"]:
        problems.append("bounds must satisfy 0 < lower <= upper < infinity")
    if report["lag"] < 0:
        problems.append("lag must be a non-negative integer")
    if problems:
        raise ValueError("invalid modulator: " + "; ".join(problems))
    return report
