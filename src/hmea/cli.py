"""Command-line comparison for the built-in update policies."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np

from . import __version__
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
DEFAULT_STATES = 8
DEFAULT_GAMMA = 0.99
DEFAULT_REWARD_NOISE = 0.1


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
    n_states: int = DEFAULT_STATES,
    gamma: float = DEFAULT_GAMMA,
    sigma_r: float = DEFAULT_REWARD_NOISE,
) -> list[ComparisonResult]:
    """Run one or all built-in policies with a shared experiment configuration."""
    factories = _policy_factories()
    selected = factories.items() if policy == "all" else [(policy, factories[policy])]
    rows: list[ComparisonResult] = []

    for _, (label, factory) in selected:
        agent = QAgent(
            ChainMDP(n_states=n_states, gamma=gamma, sigma_r=sigma_r),
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


def _state_count(value: str) -> int:
    result = _positive_integer(value)
    if result < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return result


def _gamma(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise argparse.ArgumentTypeError("must satisfy 0 <= value < 1")
    return result


def _nonnegative_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be finite and at least 0")
    return result


def _configuration(
    args: argparse.Namespace, defaults: tuple[int, int, int]
) -> dict[str, int | float]:
    return {
        "steps": args.steps or defaults[0],
        "seeds": args.seeds or defaults[1],
        "log_every": args.log_every or defaults[2],
        "seed": args.seed,
        "states": args.states,
        "gamma": args.gamma,
        "reward_noise": args.reward_noise,
    }


def _render_table(configuration: dict[str, int | float], rows: list[ComparisonResult]) -> str:
    lines = [
        (
            "HMEA comparison: "
            f"steps={configuration['steps']:,}, seeds={configuration['seeds']}, "
            f"log_every={configuration['log_every']}, states={configuration['states']}, "
            f"gamma={configuration['gamma']}, reward_noise={configuration['reward_noise']}"
        ),
        "Policy        Mean bias (95% CI)     Mean RMSE (95% CI)    Current TD error",
    ]
    for row in rows:
        current = "yes" if row["uses_current_td_error"] else "no"
        lines.append(
            f"{row['policy']:<13} "
            f"{row['mean_bias']:+.4f} +/- {row['bias_ci95']:.4f}      "
            f"{row['mean_rmse']:.4f} +/- {row['rmse_ci95']:.4f}      "
            f"{current}"
        )
    lines.append("Interpret rows only within this controlled environment and configuration.")
    return "\n".join(lines) + "\n"


def _render_json(configuration: dict[str, int | float], rows: list[ComparisonResult]) -> str:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "software": {"hmea": __version__, "numpy": np.__version__},
                "configuration": configuration,
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )


def _render_csv(configuration: dict[str, int | float], rows: list[ComparisonResult]) -> str:
    fields = [
        "policy",
        "mean_bias",
        "bias_ci95",
        "mean_rmse",
        "rmse_ci95",
        "mean_start_value",
        "uses_current_td_error",
        "steps",
        "seeds",
        "log_every",
        "seed",
        "states",
        "gamma",
        "reward_noise",
        "hmea_version",
        "numpy_version",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                **row,
                **configuration,
                "hmea_version": __version__,
                "numpy_version": np.__version__,
            }
        )
    return stream.getvalue()


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
        "--states",
        type=_state_count,
        default=DEFAULT_STATES,
        help=f"number of chain states (default: {DEFAULT_STATES})",
    )
    parser.add_argument(
        "--gamma",
        type=_gamma,
        default=DEFAULT_GAMMA,
        help=f"discount factor in [0, 1) (default: {DEFAULT_GAMMA})",
    )
    parser.add_argument(
        "--reward-noise",
        type=_nonnegative_float,
        default=DEFAULT_REWARD_NOISE,
        help=f"reward-noise standard deviation (default: {DEFAULT_REWARD_NOISE})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="use the reference configuration instead of the short default",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("table", "json", "csv"),
        help="output format (default: table)",
    )
    parser.add_argument("--json", action="store_true", help="alias for --format json")
    parser.add_argument("-o", "--output", type=Path, help="write results to this file")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installed command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.json and args.output_format not in (None, "json"):
        parser.error("--json cannot be combined with a non-JSON --format")
    output_format = "json" if args.json else args.output_format or "table"
    defaults = (
        (FULL_STEPS, FULL_SEEDS, FULL_LOG_EVERY)
        if args.full
        else (QUICK_STEPS, QUICK_SEEDS, QUICK_LOG_EVERY)
    )
    configuration = _configuration(args, defaults)
    rows = run_comparison(
        n_steps=int(configuration["steps"]),
        n_seeds=int(configuration["seeds"]),
        log_every=int(configuration["log_every"]),
        seed=args.seed,
        policy=args.policy,
        n_states=args.states,
        gamma=args.gamma,
        sigma_r=args.reward_noise,
    )

    renderers = {"table": _render_table, "json": _render_json, "csv": _render_csv}
    rendered = renderers[output_format](configuration, rows)
    if args.output is None:
        print(rendered, end="")
        return 0

    try:
        args.output.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        parser.error(f"cannot write output file: {exc}")
    print(f"Wrote {output_format} results to {args.output}.")
    return 0
