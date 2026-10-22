# Changelog

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- A deterministic experiment-spec compiler for prose, key-value text, and JSON, plus an optional
  caller-supplied language model that falls back to that parser. Compiled specs expose a stable
  `spec_id` and can be trained with `run_spec` or `hmea-spec --run`.
- `hmea-tools` and a newline-delimited MCP server (`python -m hmea.mcp`) for listing policies,
  auditing a built-in rule, compiling a spec, running it, and summarizing a paired difference.
  These entry points do not call a language model or open a network connection.
- Explicit `--policy clipped` selection. The default `--policy all` comparison remains the four
  reference rules.
- Opt-in `--compare-to` reference comparisons for final RMSE and signed bias using matched seeds.
- NumPy-only paired percentile bootstrap intervals, with reproducible sampling and an explicit
  indicator for zero observed variation.
- Paired JSON exports with per-seed differences and long-form CSV summaries with experiment
  and comparison metadata; output without `--compare-to` is unchanged.
- A practical comparison guide covering interpretation, repeatable runs, and uncertainty limits.
- Guides for [experiment specs](docs/experiment-specs.md) and [diagnostic tools](docs/diagnostics-tools.md).

### Fixed

- Reward-noise draws larger than the internal block are sampled at the requested length instead
  of being truncated.
- Whole-number floats and booleans are rejected where an integer count, seed, or modulation lag
  is required.
- A multiplier that returns the TD-error array itself is copied before the in-place update, so
  the scale is applied once. Multipliers with the wrong length or a non-finite value are rejected
  instead of being broadcast into the Q update.
- `run_comparison` raises `ValueError` for an unknown policy name, matching paired comparison.

### Changed

- Raised install floors for the supported Python 3.10–3.14 range: NumPy 2.2.6, Matplotlib 3.10,
  pytest 9, pytest-cov 7, Ruff 0.17, build 1.6, Twine 7, pre-commit 4.6, and setuptools 80.
  Newer interpreters still resolve newer releases when those releases support them.
- Pinned CI to `actions/setup-python` v7.1.0 and the Ruff pre-commit hook to v0.17.1.

## [0.2.0] - 2026-08-16

### Added

- CLI controls for chain length, discount factor, and reward-noise level.
- CSV output, file output, and an explicit `--format` option while retaining `--json`.
- Schema and software-version metadata in machine-readable experiment results.
- A weekly repository health workflow and a documented 7/14/28-day maintenance cycle.
- A dated market-direction review with evidence, forecasts, and decision gates.

### Changed

- Expanded the command documentation around user outcomes and reproducible exports.
- Moved dependency and GitHub Actions update checks from monthly to weekly.

## [0.1.0] - 2026-08-15

### Added

- Vectorized tabular Q-learning with per-seed bias, RMSE, and start-state metrics.
- Baseline, online, clipped, fixed-lag, and leaky-state modulation policies.
- A reproducible four-policy comparison and a separate training-dynamics experiment.
- An installed `hmea-compare` command and `python -m hmea` entry point for no-code comparisons.
- Structural validation for custom modulators and explicit parameter checks.
- Current Python package metadata, typed-package marker, coverage settings, and build smoke
  checks.
- GitHub Issue Forms, pull request template, Dependabot configuration, security policy, and
  least-privilege CI.

### Changed

- Reframed the repository as an empirical research prototype.
- Replaced unsupported publication and proof claims with testable implementation notes and
  established Q-learning references.
- Replaced the watermarked bitmap header with a source-controlled SVG banner.
- Made the package, source, and citation versions consistent.

### Removed

- Broken publication, package-index, CI, and Discussions links.
- Build products, interpreter caches, test caches, and generated package metadata.
