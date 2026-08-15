# Market direction

Snapshot date: August 16, 2026. Counts are GitHub stars at the time of review. Stars are a rough
attention proxy, not proof of quality, active use, or willingness to adopt HMEA.

## Current signals

| Project | Current focus | Stars | Created | Last code push checked |
|---|---|---:|---|---|
| [Ray](https://github.com/ray-project/ray) | Distributed AI compute | 43,520 | 2016 | August 15, 2026 |
| [verl](https://github.com/verl-project/verl) | Foundation-model RL post-training | 22,970 | 2024 | August 14, 2026 |
| [TRL](https://github.com/huggingface/trl) | Foundation-model post-training | 19,077 | 2020 | August 15, 2026 |
| [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) | Reliable deep-RL implementations | 13,694 | 2020 | July 25, 2026 |
| [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) | Standard environment interface | 12,325 | 2022 | August 5, 2026 |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | Inspectable reference implementations | 10,274 | 2019 | April 20, 2026 |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Distributed agentic RL | 9,914 | 2023 | August 13, 2026 |
| [TorchRL](https://github.com/pytorch/rl) | Modular PyTorch RL primitives | 3,519 | 2022 | August 15, 2026 |

GitHub's 2025 Octoverse review reports that Python remained the leading language in newly created
AI projects and that production-oriented packaging, typing, and reproducibility became more
important as experimentation accelerated. The selected repositories show two simultaneous
signals: rapid attention around foundation-model post-training and durable demand for standard,
inspectable reinforcement-learning tools.

## Where HMEA should compete

HMEA should remain a small, CPU-native diagnostic for update feedback and estimation bias. It
should not imitate distributed post-training frameworks whose value depends on model fleets,
accelerators, rollout engines, and a large integration surface.

The strongest adjacent opportunity is interoperability with standard discrete environments while
preserving exact seeds, transparent calculations, and exportable results. That direction serves
students and researchers who need to understand a failure mode before moving to a larger stack.

| Opportunity | Attention signal | Fit with HMEA | Decision |
|---|---|---|---|
| Reproducible result export | Broad Python tooling trend | High | Shipped in 0.2.0 |
| Gymnasium toy-text adapter | Durable standard interface | High | Next implementation target |
| Paired statistical comparisons | Repeated research need | High | Near-term target |
| Distributed model post-training | Fast-growing repositories | Low | Defer |
| Generic autonomous-agent framework | Crowded and dependency-heavy | Low | Decline without demand evidence |

## Forward view

These are conditional forecasts, not measured future demand.

- **Next 6 months:** foundation-model and agentic RL will probably continue to attract repository
  attention. Small projects will benefit more from compatible evaluation and diagnostics than
  from copying the full training stack. Confidence: medium.
- **6 to 12 months:** reproducible run metadata, comparable benchmarks, and standard environment
  interfaces are likely to become stronger adoption filters as generated prototypes compete for
  trust. Confidence: medium.
- **Beyond 12 months:** interoperability is likely to matter more than adding another monolithic
  framework. HMEA should add adapters only when they preserve its inspectable core. Confidence:
  low to medium.

## Ninety-day priorities

1. Ship parameterized CLI experiments and CSV/JSON file exports.
2. Add an optional Gymnasium adapter for a small discrete environment and compare it with the
   built-in chain under common reporting.
3. Add paired comparisons and effect-size reporting across shared seed streams.
4. Review requests at the 14-day checkpoint and revise the roadmap at the 28-day checkpoint.

The [maintenance cycle](maintenance.md) defines the recurring review and the evidence required to
change direction.
