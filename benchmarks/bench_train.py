"""Measure QAgent training throughput under each built-in modulation policy."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

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


def _factories() -> dict[str, Callable[[], object]]:
    return {
        "Baseline": NoModulation,
        "Immediate": OnlineModulation,
        "Fixed lag": LaggedModulation,
        "Leaky state": HomeostaticModulation,
    }


def measure(
    factory: Callable[[], object], *, n_steps: int, n_seeds: int, repeats: int
) -> list[float]:
    """Return throughput samples in seed-steps per second."""
    QAgent(ChainMDP(), factory(), seed=0).train(
        n_steps=min(2_000, n_steps), n_seeds=n_seeds, log_every=min(400, n_steps)
    )
    samples = []
    for _ in range(repeats):
        agent = QAgent(ChainMDP(), factory(), seed=0)
        started = time.perf_counter()
        agent.train(n_steps=n_steps, n_seeds=n_seeds, log_every=400)
        samples.append(n_steps * n_seeds / (time.perf_counter() - started))
    return samples


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--repeats", type=int)
    args = parser.parse_args(argv)
    n_steps = args.steps or (10_000 if args.quick else 80_000)
    repeats = args.repeats or (2 if args.quick else 5)
    print(f"steps={n_steps:,} seeds={args.seeds} repeats={repeats}")
    print(f"{'Policy':<14} {'median':>14} {'minimum':>14} {'maximum':>14}")
    for name, factory in _factories().items():
        samples = measure(factory, n_steps=n_steps, n_seeds=args.seeds, repeats=repeats)
        print(
            f"{name:<14} {statistics.median(samples):>14,.0f} "
            f"{min(samples):>14,.0f} {max(samples):>14,.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
