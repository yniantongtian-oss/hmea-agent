"""Compare immediate, delayed, and leaky-state update modulation."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from hmea import (
        ChainMDP,
        HomeostaticModulation,
        LaggedModulation,
        NoModulation,
        OnlineModulation,
        QAgent,
    )
except ModuleNotFoundError:  # pragma: no cover - convenience for an uninstalled checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hmea import (
        ChainMDP,
        HomeostaticModulation,
        LaggedModulation,
        NoModulation,
        OnlineModulation,
        QAgent,
    )

DEFAULT_STEPS = 80_000
DEFAULT_SEEDS = 128
DEFAULT_LOG_EVERY = 400
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "bias_comparison.png"

COLORS = {
    "Baseline": "#315B7D",
    "Immediate": "#C85C47",
    "Fixed lag": "#3B7F6B",
    "Leaky state": "#8667A7",
}


def _factories() -> dict[str, Callable[[], object]]:
    return {
        "Baseline": NoModulation,
        "Immediate": lambda: OnlineModulation(beta=1.5, tau=0.5, bounds=(0.5, 2.0)),
        "Fixed lag": lambda: LaggedModulation(lag=8, beta=1.5, tau=0.5, bounds=(0.5, 2.0)),
        "Leaky state": lambda: HomeostaticModulation(
            beta=0.8, tau=0.5, bounds=(0.5, 1.5), eta=0.05
        ),
    }


def _mean_ci95(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=1)
    if values.shape[1] < 2:
        return mean, np.zeros_like(mean)
    half_width = 1.96 * values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])
    return mean, half_width


def run_experiment(*, n_steps: int, n_seeds: int, log_every: int) -> dict[str, dict[str, object]]:
    """Train each policy with common seed streams and return plot-ready results."""
    results: dict[str, dict[str, object]] = {}
    for name, factory in _factories().items():
        started = time.perf_counter()
        env = ChainMDP(n_states=8, gamma=0.99, sigma_r=0.1)
        modulator = factory()
        agent = QAgent(env, modulator, seed=0)
        log = agent.train(n_steps=n_steps, n_seeds=n_seeds, log_every=log_every)
        mean, ci = _mean_ci95(log["bias_by_seed"])
        final_by_seed = log["bias_by_seed"][-1]
        final_mean = float(final_by_seed.mean())
        final_ci = float(
            0.0 if n_seeds < 2 else 1.96 * final_by_seed.std(ddof=1) / np.sqrt(n_seeds)
        )
        results[name] = {
            "steps": log["steps"],
            "mean": mean,
            "ci": ci,
            "final_mean": final_mean,
            "final_ci": final_ci,
            "rmse": log["rmse"],
            "audit": log["audit"],
        }
        elapsed = time.perf_counter() - started
        print(
            f"{name:<12} final bias={final_mean:+.4f} "
            f"final RMSE={log['rmse'][-1]:.4f} time={elapsed:.1f}s"
        )
    return results


def check_results(results: dict[str, dict[str, object]], *, reference_configuration: bool) -> None:
    """Run finite-output checks and fixed-reference regression thresholds."""
    for name, result in results.items():
        arrays = (result["steps"], result["mean"], result["ci"], result["rmse"])
        if not all(np.isfinite(np.asarray(values)).all() for values in arrays):
            raise RuntimeError(f"{name} produced non-finite output")

    if not reference_configuration:
        print("Smoke checks passed for the custom configuration.")
        return

    baseline = float(results["Baseline"]["final_mean"])
    immediate = float(results["Immediate"]["final_mean"])
    lagged = float(results["Fixed lag"]["final_mean"])
    if abs(baseline) >= 0.10:
        raise RuntimeError(f"baseline final bias moved outside the reference band: {baseline:+.4f}")
    if immediate <= 0.50:
        raise RuntimeError(f"immediate final bias fell below the reference band: {immediate:+.4f}")
    if abs(lagged) >= 0.10:
        raise RuntimeError(f"fixed-lag final bias moved outside the reference band: {lagged:+.4f}")
    print("Reference regression checks passed.")


def plot_results(
    results: dict[str, dict[str, object]], *, output: Path, n_steps: int, n_seeds: int
) -> None:
    """Write the two-panel comparison figure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, (curve_axis, bar_axis) = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.0),
        facecolor="white",
        gridspec_kw={"width_ratios": [2.25, 1.0]},
    )

    names = list(results)
    for name in names:
        result = results[name]
        steps = np.asarray(result["steps"])
        mean = np.asarray(result["mean"])
        ci = np.asarray(result["ci"])
        color = COLORS[name]
        curve_axis.plot(steps, mean, color=color, linewidth=2.0, label=name)
        curve_axis.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.14)

    curve_axis.axhline(0.0, color="#59636E", linewidth=1.0, linestyle="--")
    curve_axis.set_xlabel("Training steps")
    curve_axis.set_ylabel("Signed mean value error")
    curve_axis.set_title("Bias over training")
    curve_axis.legend(frameon=False, ncol=2)
    curve_axis.spines[["top", "right"]].set_visible(False)

    positions = np.arange(len(names))
    final_means = [float(results[name]["final_mean"]) for name in names]
    final_cis = [float(results[name]["final_ci"]) for name in names]
    bar_axis.bar(
        positions,
        final_means,
        yerr=final_cis,
        capsize=4,
        color=[COLORS[name] for name in names],
        alpha=0.9,
        error_kw={"ecolor": "#3D4650", "linewidth": 1.2},
    )
    bar_axis.axhline(0.0, color="#59636E", linewidth=1.0, linestyle="--")
    bar_axis.set_xticks(positions)
    bar_axis.set_xticklabels(["Base", "Now", "Lag", "State"])
    bar_axis.set_ylabel("Final signed mean error")
    bar_axis.set_title("Final logged point")
    bar_axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle("Immediate and delayed update modulation on ChainMDP", fontsize=14)
    figure.text(
        0.5,
        0.01,
        f"{n_seeds} seeded replicas per policy · {n_steps:,} steps · mean ± 95% CI",
        ha="center",
        color="#59636E",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(output, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run a short smoke configuration")
    parser.add_argument("--steps", type=int, help="training steps per policy")
    parser.add_argument("--seeds", type=int, help="seeded replicas per policy")
    parser.add_argument("--log-every", type=int, help="metric logging interval")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output PNG path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.quick:
        n_steps = args.steps or 5_000
        n_seeds = args.seeds or 16
        log_every = args.log_every or 100
    else:
        n_steps = args.steps or DEFAULT_STEPS
        n_seeds = args.seeds or DEFAULT_SEEDS
        log_every = args.log_every or DEFAULT_LOG_EVERY

    print(f"ChainMDP comparison: steps={n_steps:,}, seeds={n_seeds}, log_every={log_every}")
    started = time.perf_counter()
    results = run_experiment(n_steps=n_steps, n_seeds=n_seeds, log_every=log_every)
    reference = (
        n_steps == DEFAULT_STEPS and n_seeds == DEFAULT_SEEDS and log_every == DEFAULT_LOG_EVERY
    )
    check_results(results, reference_configuration=reference)
    plot_results(results, output=args.output, n_steps=n_steps, n_seeds=n_seeds)
    print(f"Figure written to {args.output.resolve()}")
    print(f"Total time: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
