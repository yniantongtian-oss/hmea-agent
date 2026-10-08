# Market direction

Snapshot date: October 5, 2026 (UTC). Star counts and last repository pushes were checked on
the linked GitHub repositories. Stars measure attention, not adoption or product quality.

## Current signals

| Project | Current focus | Stars | Change since August 16 | Last repository push checked |
|---|---|---:|---:|---|
| [Ray](https://github.com/ray-project/ray) | Distributed AI compute | 43,970 | +450 | October 5, 2026 |
| [verl](https://github.com/verl-project/verl) | Foundation-model RL post-training | 23,747 | +777 | October 3, 2026 |
| [TRL](https://github.com/huggingface/trl) | Foundation-model post-training | 19,450 | +373 | October 4, 2026 |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | Reliable deep-RL implementations | 13,864 | +170 | September 9, 2026 |
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | Standard environment interface | 12,618 | +293 | October 5, 2026 |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | Inspectable reference implementations | 10,490 | +216 | April 20, 2026 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Distributed agentic RL | 10,067 | +153 | September 17, 2026 |
| [TorchRL](https://github.com/pytorch/rl) | Modular PyTorch RL primitives | 3,590 | +71 | October 4, 2026 |

Repository push dates include branch and maintenance activity; they do not by themselves prove a
new release or user adoption. The faster star growth around foundation-model post-training does
not establish demand for HMEA to enter that category. Continued activity around Gymnasium and
inspectable reference implementations supports keeping reproducibility and clear environment
interfaces in view.

HMEA's direct public signals remain limited: [one star, no forks](https://github.com/yniantongtian-oss/hmea-agent),
no standalone issues or Discussions, and no public request for another environment or framework
integration at this review. The [paired-comparison pull request](https://github.com/yniantongtian-oss/hmea-agent/pull/3)
was merged on September 30; the latest tagged release is still
[v0.2.0](https://github.com/yniantongtian-oss/hmea-agent/releases/tag/v0.2.0).

## Where HMEA should compete

HMEA remains a small, CPU-native diagnostic for update feedback and estimation bias. Its useful
distinction is an experiment whose seeds, calculations, and results can be inspected end to end.
Distributed post-training would add substantial validation and maintenance cost without a verified
HMEA user workflow.

| Opportunity | Verified value and fit | Validation cost | Maintenance cost | Decision |
|---|---|---|---|---|
| Paired comparisons and reproducible exports | Implemented, directly useful within the existing diagnostic | Low; merged code and CI passed | Low | Prepare a release after review and release gates |
| Additional state-action diagnostics | Plausible way to expose errors hidden by aggregate bias; no direct request yet | Medium | Low | Seek a concrete use case before implementation |
| Optional Gymnasium toy-text adapter | Standard interface with strong adjacent activity; no HMEA demand yet | Medium to high, especially seed and termination semantics | Medium | Keep conditional on a reproducible user case |
| Distributed model post-training | High adjacent attention, poor fit with this diagnostic | Very high | Very high | Defer |

The [maintenance cycle](maintenance.md) sets demand gates for a new dependency or problem domain.
Adjacent stars do not satisfy those gates on their own.

## Release recommendation

The merged paired comparison is a meaningful release candidate. Do not publish it as v0.3.0
until the version, citation, installation instructions, changelog date, and package metadata agree
with the actual release; the complete build, installed-wheel, experiment, and CI gates must pass
from the release commit. There is no v0.3.0 tag or release as of this review.

## Forward view

These are forecasts, not observed demand.

- Over the next six months, foundation-model RL projects will probably continue to attract the
  most attention. This need not change HMEA's small diagnostic scope. Confidence: medium.
- Reproducible metrics and explicit seed behavior are likely to remain useful adoption filters
  for evaluation tools. Confidence: medium.
- A Gymnasium example could make HMEA easier to discover, but current public evidence does not
  show that it would bring users. Confidence: low.

## Next review priorities

1. Review the merged paired-comparison behavior and complete a truthful release candidate.
2. Collect a concrete request and evaluation plan before adding another environment or metric.
3. Recheck direct use, issues, Discussions, and adjacent projects at the next 28-day review.
