"""Command-line access to local HMEA diagnostic tools."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__
from .diagnostics import (
    audit_policy,
    list_policies,
    parse_experiment_spec,
    run_experiment,
    summarize_paired_difference,
)
from .mcp import serve
from .spec import SpecError


def _numbers(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected a comma-separated list of numbers")
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated list of numbers") from exc


def build_parser() -> argparse.ArgumentParser:
    """Create the diagnostics command parser."""
    parser = argparse.ArgumentParser(
        prog="hmea-tools",
        description="Run local HMEA diagnostic tools. No language model or network call is used.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list-policies", help="list built-in policies")

    audit = commands.add_parser("audit", help="show one policy's declared structure")
    audit.add_argument("policy", help="policy id")

    parse = commands.add_parser("parse", help="compile an experiment description")
    parse.add_argument("text", help="experiment description")

    run = commands.add_parser("run", help="compile and train an experiment description")
    run.add_argument("text", help="experiment description")

    paired = commands.add_parser(
        "paired-difference",
        help="paired mean difference of two comma-separated samples",
    )
    paired.add_argument("--candidate", required=True, type=_numbers)
    paired.add_argument("--reference", required=True, type=_numbers)
    paired.add_argument("--resamples", type=int, default=10_000)
    paired.add_argument("--seed", type=int, default=0)

    commands.add_parser("serve", help="serve the tools over stdio MCP")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one diagnostic tool or the MCP server."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        return serve()
    try:
        payload = _invoke(args)
    except (SpecError, TypeError, ValueError) as exc:
        print(f"hmea-tools: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0


def _invoke(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "list-policies":
        return list_policies()
    if args.command == "audit":
        return audit_policy(args.policy)
    if args.command == "parse":
        return parse_experiment_spec(args.text)
    if args.command == "run":
        return run_experiment(text=args.text)
    if args.command == "paired-difference":
        return summarize_paired_difference(
            args.candidate,
            args.reference,
            n_resamples=args.resamples,
            seed=args.seed,
        )
    raise SpecError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
