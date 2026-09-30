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
from .statistics import PairedDifference, paired_mean_difference

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


class PairedComparisonResult(PairedDifference):
    """One candidate and metric compared with the reference policy."""

    policy: str
    reference_policy: str
    metric: str


class ComparisonMetadata(TypedDict):
    """The interpretation and reproducibility settings of paired intervals."""

    reference_policy: str
    difference_direction: str
    interval_method: str
    confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int


class PairedComparison(TypedDict):
    """Absolute summaries and paired differences from the same training runs."""

    results: list[ComparisonResult]
    paired_comparisons: list[PairedComparisonResult]
    comparison: ComparisonMetadata


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


def _run_policies(
    *,
    policies: Sequence[str],
    n_steps: int,
    n_seeds: int,
    log_every: int,
    seed: int = 0,
    n_states: int = DEFAULT_STATES,
    gamma: float = DEFAULT_GAMMA,
    sigma_r: float = DEFAULT_REWARD_NOISE,
) -> tuple[list[ComparisonResult], dict[str, dict[str, np.ndarray]]]:
    factories = _policy_factories()
    rows: list[ComparisonResult] = []
    observations: dict[str, dict[str, np.ndarray]] = {}

    for policy in policies:
        label, factory = factories[policy]
        agent = QAgent(
            ChainMDP(n_states=n_states, gamma=gamma, sigma_r=sigma_r),
            factory(),
            seed=seed,
        )
        log = agent.train(n_steps=n_steps, n_seeds=n_seeds, log_every=log_every)
        final_bias = log["bias_by_seed"][-1]
        final_rmse = log["rmse_by_seed"][-1]
        final_start = log["q_start_by_seed"][-1]
        observations[policy] = {"bias": final_bias.copy(), "rmse": final_rmse.copy()}
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
    return rows, observations


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
    rows, _ = _run_policies(
        policies=list(_policy_factories()) if policy == "all" else [policy],
        n_steps=n_steps,
        n_seeds=n_seeds,
        log_every=log_every,
        seed=seed,
        n_states=n_states,
        gamma=gamma,
        sigma_r=sigma_r,
    )
    return rows


def run_paired_comparison(
    *,
    n_steps: int,
    n_seeds: int,
    log_every: int,
    seed: int = 0,
    policy: str = "all",
    n_states: int = DEFAULT_STATES,
    gamma: float = DEFAULT_GAMMA,
    sigma_r: float = DEFAULT_REWARD_NOISE,
    reference_policy: str = "baseline",
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 0,
) -> PairedComparison:
    """Compare final per-seed metrics using candidate-minus-reference differences.

    Paired replicas share random streams, not necessarily state/action paths.
    Matching requires the same root seed, seed count and environment settings.
    Bootstrap randomness is separate from training randomness. Each policy is
    trained once, including a reference absent from the requested selection.
    """
    factories = _policy_factories()
    if reference_policy not in factories:
        raise ValueError(f"unknown reference policy: {reference_policy}")
    if policy != "all" and policy not in factories:
        raise ValueError(f"unknown policy: {policy}")
    if policy == reference_policy:
        raise ValueError("select a candidate policy different from the reference policy")
    if isinstance(n_seeds, (bool, np.bool_)) or not isinstance(n_seeds, (int, np.integer)):
        raise ValueError("paired comparison requires an integer n_seeds of at least 2")
    if n_seeds < 2:
        raise ValueError("paired comparison requires at least 2 seeds")
    # Validate bootstrap settings before spending time training any policies.
    paired_mean_difference([0, 0], [0, 0], n_resamples=bootstrap_resamples, seed=bootstrap_seed)
    policies = list(factories) if policy == "all" else [reference_policy, policy]
    rows, observations = _run_policies(
        policies=policies,
        n_steps=n_steps,
        n_seeds=n_seeds,
        log_every=log_every,
        seed=seed,
        n_states=n_states,
        gamma=gamma,
        sigma_r=sigma_r,
    )
    reference_label = factories[reference_policy][0]
    paired_rows: list[PairedComparisonResult] = []
    for candidate in policies:
        if candidate == reference_policy:
            continue
        for metric in ("rmse", "bias"):
            paired_rows.append(
                {
                    "policy": factories[candidate][0],
                    "reference_policy": reference_label,
                    "metric": metric,
                    **paired_mean_difference(
                        observations[candidate][metric],
                        observations[reference_policy][metric],
                        n_resamples=bootstrap_resamples,
                        seed=bootstrap_seed,
                    ),
                }
            )
    return {
        "results": rows,
        "paired_comparisons": paired_rows,
        "comparison": {
            "reference_policy": reference_label,
            "difference_direction": "candidate_minus_reference",
            "interval_method": "paired_percentile_bootstrap",
            "confidence_level": 0.95,
            "bootstrap_resamples": int(bootstrap_resamples),
            "bootstrap_seed": int(bootstrap_seed),
        },
    }


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


def _render_paired_table(configuration: dict[str, int | float], report: PairedComparison) -> str:
    lines = [
        _render_table(configuration, report["results"]).rstrip(),
        "",
        f"Paired differences: candidate - reference ({report['comparison']['reference_policy']})",
        "Policy        Metric   Mean difference    Paired 95% interval         Pairs",
    ]
    for row in report["paired_comparisons"]:
        lines.append(
            f"{row['policy']:<13} {row['metric']:<8} {row['mean_difference']:+.4f}"
            f"            [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}]"
            f"       {row['n_pairs']}"
        )
    lines.extend(
        [
            "Negative RMSE differences mean lower error; bias differences indicate direction only.",
            "Approximate paired percentile-bootstrap intervals; "
            "not simultaneous across comparisons.",
            "An interval crossing zero is inconclusive, not evidence of equivalence.",
        ]
    )
    if any(row["observed_zero_variance"] for row in report["paired_comparisons"]):
        lines.append(
            "Some differences have zero observed variance; "
            "collapsed intervals do not imply certainty."
        )
    return "\n".join(lines) + "\n"


def _render_paired_json(configuration: dict[str, int | float], report: PairedComparison) -> str:
    return (
        json.dumps(
            {
                "schema_version": 2,
                "software": {"hmea": __version__, "numpy": np.__version__},
                "configuration": configuration,
                **report,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _render_paired_csv(configuration: dict[str, int | float], report: PairedComparison) -> str:
    fields = [
        "schema_version",
        "policy",
        "reference_policy",
        "metric",
        "n_pairs",
        "mean_difference",
        "ci95_low",
        "ci95_high",
        "observed_zero_variance",
        "difference_direction",
        "interval_method",
        "confidence_level",
        "bootstrap_resamples",
        "bootstrap_seed",
        *configuration,
        "hmea_version",
        "numpy_version",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in report["paired_comparisons"]:
        summary = {key: value for key, value in row.items() if key != "per_seed_differences"}
        writer.writerow(
            {
                "schema_version": 2,
                **summary,
                **report["comparison"],
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
    parser.add_argument(
        "--compare-to",
        choices=tuple(_policy_factories()),
        help="add paired differences against this reference policy (requires at least 2 seeds)",
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
    parameters = {
        "n_steps": int(configuration["steps"]),
        "n_seeds": int(configuration["seeds"]),
        "log_every": int(configuration["log_every"]),
        "seed": args.seed,
        "policy": args.policy,
        "n_states": args.states,
        "gamma": args.gamma,
        "sigma_r": args.reward_noise,
    }
    if args.compare_to is None:
        rows = run_comparison(**parameters)
        renderers = {"table": _render_table, "json": _render_json, "csv": _render_csv}
        rendered = renderers[output_format](configuration, rows)
    else:
        try:
            report = run_paired_comparison(**parameters, reference_policy=args.compare_to)
        except ValueError as exc:
            parser.error(str(exc))
        paired_renderers = {
            "table": _render_paired_table,
            "json": _render_paired_json,
            "csv": _render_paired_csv,
        }
        rendered = paired_renderers[output_format](configuration, report)
    if args.output is None:
        print(rendered, end="")
        return 0

    try:
        args.output.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        parser.error(f"cannot write output file: {exc}")
    print(f"Wrote {output_format} results to {args.output}.")
    return 0
