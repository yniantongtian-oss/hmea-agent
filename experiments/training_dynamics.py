"""Plot signed bias and RMSE for baseline, clipped, and immediate modulation."""

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
        ClippedModulation,
        NoModulation,
        OnlineModulation,
        QAgent,
    )
except ModuleNotFoundError:  # pragma: no cover - convenience for an uninstalled checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hmea import (
        ChainMDP,
        ClippedModulation,
        NoModulation,
        OnlineModulation,
        QAgent,
    )

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "training_dynamics.png"
COLORS = {"Baseline": "#315B7D", "Clipped": "#8667A7", "Immediate": "#C85C47"}


def _factories() -> dict[str, Callable[[], object]]:
    return {
        "Baseline": NoModulation,
        "Clipped": lambda: ClippedModulation(beta=0.4, tau=0.5, bounds=(0.8, 1.2)),
        "Immediate": lambda: OnlineModulation(beta=1.5, tau=0.5, bounds=(0.5, 2.0)),
    }


def run(*, n_steps: int, n_seeds: int, log_every: int) -> dict[str, dict[str, np.ndarray]]:
    results: dict[str, dict[str, np.ndarray]] = {}
    for name, factory in _factories().items():
        log = QAgent(
            ChainMDP(),
            factory(),
            alpha=0.05,
            eps_min=0.05,
            seed=0,
        ).train(n_steps=n_steps, n_seeds=n_seeds, log_every=log_every)
        results[name] = {"steps": log["steps"], "bias": log["bias"], "rmse": log["rmse"]}
        print(f"{name:<10} final bias={log['bias'][-1]:+.4f} RMSE={log['rmse'][-1]:.4f}")
    return results


def plot(results: dict[str, dict[str, np.ndarray]], *, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, (bias_axis, rmse_axis) = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor="white")
    for name, result in results.items():
        color = COLORS[name]
        bias_axis.plot(result["steps"], result["bias"], label=name, color=color, linewidth=2)
        rmse_axis.plot(result["steps"], result["rmse"], label=name, color=color, linewidth=2)
    bias_axis.axhline(0.0, color="#59636E", linewidth=1, linestyle="--")
    bias_axis.set_title("Signed mean error")
    rmse_axis.set_title("Root mean squared error")
    for axis in (bias_axis, rmse_axis):
        axis.set_xlabel("Training steps")
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Training dynamics under three update multipliers", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=180, metadata={"Software": "matplotlib"})
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    n_steps, n_seeds, log_every = (5_000, 16, 100) if args.quick else (60_000, 64, 500)
    started = time.perf_counter()
    results = run(n_steps=n_steps, n_seeds=n_seeds, log_every=log_every)
    if not all(np.isfinite(values).all() for item in results.values() for values in item.values()):
        raise RuntimeError("experiment produced non-finite output")
    plot(results, output=args.output)
    print(f"Figure written to {args.output.resolve()}")
    print(f"Total time: {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
