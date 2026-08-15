# Release guide

## Prepare

1. Update the version in `pyproject.toml`, `src/hmea/__init__.py`, and `CITATION.cff`.
2. Move user-visible changes into a dated section in `CHANGELOG.md`.
3. Regenerate both tracked figures with the release code.
4. Run the full local gate:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov=hmea --cov-report=term-missing
python experiments/reproduce_bias_comparison.py
python experiments/training_dynamics.py
python -m build
python -m twine check dist/*
```

5. Install the wheel into a clean environment and import `hmea` from outside the repository.
6. Review the rendered README, both figures, and all relative links on GitHub.

## Repository settings

Before the first public release:

- protect the default branch and require the CI workflow;
- enable Dependabot alerts, secret scanning, and push protection where available;
- enable private vulnerability reporting;
- add concise repository topics and a factual description;
- add the real repository URL and named authors to package and citation metadata;
- add package-index and CI badges only after their targets exist.

## Tag and publish

Create a signed or annotated tag only after the release commit passes CI. Build distributions from
that tag, verify their hashes, and publish through a trusted publisher or another short-lived
credential flow. Create GitHub release notes from the matching changelog section.

Do not publish from an uncommitted working tree or reuse a version number for different artifacts.
