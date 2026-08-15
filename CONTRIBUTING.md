# Contributing

Focused fixes, tests, documentation corrections, and well-scoped experiments are welcome.

## Set up a development environment

```bash
python -m venv .venv
```

Activate the environment with the command for your shell, then install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

## Before opening a pull request

Run the same checks used in CI:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov=hmea --cov-report=term-missing
python -m build
python -m twine check dist/*
python experiments/reproduce_bias_comparison.py --quick
```

Create a focused branch with `git switch -c <short-name>`. Keep one logical change per pull
request and explain the reason for the change, not only the files touched.

## Behavioral changes

- Add or update tests for every public behavior change.
- Keep random seeds explicit and report the number of independent runs.
- Label fixed-configuration observations as empirical results.
- Do not turn a finite run or interface check into a general convergence claim.
- Update `README.md`, `docs/method.md`, and `CHANGELOG.md` when public behavior changes.

## Performance changes

The training loop is vectorized across seeds. Performance pull requests should include the
benchmark command, Python and NumPy versions, CPU model, seed count, and before/after median
throughput. A faster best-case run by itself is not enough evidence.

## Reporting problems

Use the repository's issue forms for bugs, feature proposals, and usage questions. Report
security-sensitive problems through the private process in [SECURITY.md](SECURITY.md).
