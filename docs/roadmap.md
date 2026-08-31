# Roadmap

These are open directions, not release promises.

## Implemented, unreleased

- Paired comparisons against a chosen reference policy, with final RMSE and signed-bias
  differences, reproducible bootstrap intervals, and per-seed JSON exports. See the
  [comparison guide](comparing-policies.md).

## Near term

- Add property-based tests for custom modulator shape, bounds, and reset behavior.
- Report additional diagnostics such as policy accuracy and state-action error heatmaps.
- Add a small deterministic environment with stochastic transitions.
- Add an optional Gymnasium adapter for discrete toy-text environments without making it a
  core dependency.

## Later

- Separate the environment protocol from `ChainMDP` without adding a large framework dependency.
- Evaluate adaptive multipliers on several tabular tasks and hyperparameter grids.
- Add linear function approximation only after the tabular API and evaluation protocol stabilize.
- Consider a post-training or agent-evaluation adapter only after independent user requests show
  a reproducible use case that this small diagnostic can serve better than existing frameworks.

Requests should include a concrete use case, an evaluation plan, and the maintenance cost of any
new dependency. Priorities are reviewed using the evidence gates in
[market-direction.md](market-direction.md).
