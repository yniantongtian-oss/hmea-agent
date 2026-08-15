# Changelog

This file follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/).

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
