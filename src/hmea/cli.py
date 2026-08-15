"""Command-line comparison for the built-in update policies."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from typing import TypedDict

import numpy as np

from .agent import QAgent
from .envs import ChainMDP
from .modulation import (
    HomeostaticModulation,
    LaggedModulation,
    Modulator,
    NoModulation,
    OnlineModulation,
)

QUICK_STEPS = 5_000
QUICK_SEEDS = 16
QUICK_LOG_EVERY = 100
FULL_STEPS = 80_000
FULL_SEEDS = 128
FULL_LOG_EVERY = 400


class ComparisonResult(TypedDict):
    """One policy row returned by :func:`run_comparison`."""

    policy: str
    mean_bias: float
    bias_ci95: float
    mean_rmse: float
    rmse_ci95: float
    mean_start_value: float
    uses_current_td_error: bool


def _policy_factories() -> dict[str, tuple[str, Callable[[], Modulator]]]:
    return {
        "baseline": ("Baseline", NoModulation),
        "immediate": (
            "Immediate",
            lambda: OnlineModulation(beta=1.5, tau=0.5, bounds=(0.5, 2.0)),
        ),
        "fixed-lag": (
            "Fixed lag",
            lambda: LaggedModulation(lag=8, beta=1.5, tau=0.5, bounds=(0.5, 2.0)),
        ),
        "leaky-state": (
            "Leaky state",
            lambda: HomeostaticModulation(
                beta=0.8,
                tau=0.5,
                bounds=(0.5, 1.5),
                eta=0.05,
            ),
        ),
    }


def _ci95(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(1.96 * values.std(ddof=1) / math.sqrt(values.size))


def run_comparison(
    *,
    n_steps: int,
    n_seeds: int,
    log_every: int,
    seed: int = 0,
    policy: str = "all",
) -> list[ComparisonResult]:
    """Run one or all built-in policies with a shared experiment configuration."""
    factories = _policy_factories()
    selected = factories.items() if policy == "all" else [(policy, factories[policy])]
    rows: list[ComparisonResult] = []

    for _, (label, factory) in selected:
        agent = QAgent(
            ChainMDP(n_states=8, gamma=0.99, sigma_r=0.1),
            factory(),
            seed=seed,
        )
        log = agent.train(n_steps=n_steps, n_seeds=n_seeds, log_every=log_every)
        final_bias = log["bias_by_seed"][-1]
        final_rmse = log["rmse_by_seed"][-1]
        final_start = log["q_start_by_seed"][-1]
        rows.append(
            {
                "policy": label,
                "mean_bias": float(final_bias.mean()),
                "bias_ci95": _ci95(final_bias),
                "mean_rmse": float(final_rmse.mean()),
                "rmse_ci95": _ci95(final_rmse),
                "mean_start_value": float(final_start.mean()),
                "uses_current_td_error": log["audit"]["uses_current_td_error"],
            }
        )
    return rows


def _positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def _nonnegative_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return result


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="hmea-compare",
        description="Compare adaptive update multipliers in a small tabular Q-learning task.",
    )
    parser.add_argument(
        "--policy",
        choices=("all", *_policy_factories()),
        default="all",
        help="policy to run (default: all)",
    )
    parser.add_argument("--steps", type=_positive_integer, help="training steps per policy")
    parser.add_argument("--seeds", type=_positive_integer, help="seeded replicas per policy")
    parser.add_argument("--log-every", type=_positive_integer, help="metric logging interval")
    parser.add_argument("--seed", type=_nonnegative_integer, default=0, help="root random seed")
    parser.add_argument(
        "--full",
        action="store_true",
        help="use the reference configuration instead of the short default",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installed command-line interface."""
    args = build_parser().parse_args(argv)
    defaults = (
        (FULL_STEPS, FULL_SEEDS, FULL_LOG_EVERY)
        if args.full
        else (QUICK_STEPS, QUICK_SEEDS, QUICK_LOG_EVERY)
    )
    n_steps = args.steps or defaults[0]
    n_seeds = args.seeds or defaults[1]
    log_every = args.log_every or defaults[2]
    rows = run_comparison(
        n_steps=n_steps,
        n_seeds=n_seeds,
        log_every=log_every,
        seed=args.seed,
        policy=args.policy,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "configuration": {
                        "steps": n_steps,
                        "seeds": n_seeds,
                        "log_every": log_every,
                        "seed": args.seed,
                    },
                    "results": rows,
                },
                indent=2,
            )
        )
        return 0

    print(f"HMEA comparison: steps={n_steps:,}, seeds={n_seeds}, log_every={log_every}")
    print("Policy        Mean bias (95% CI)     Mean RMSE (95% CI)    Current TD error")
    for row in rows:
        current = "yes" if row["uses_current_td_error"] else "no"
        print(
            f"{row['policy']:<13} "
            f"{row['mean_bias']:+.4f} +/- {row['bias_ci95']:.4f}      "
            f"{row['mean_rmse']:.4f} +/- {row['rmse_ci95']:.4f}      "
            f"{current}"
        )
    print("Interpret rows only within this controlled environment and configuration.")
    return 0
