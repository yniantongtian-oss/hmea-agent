# Roadmap

These are open directions, not release promises.

## Near term

- Add property-based tests for custom modulator shape, bounds, and reset behavior.
- Report additional diagnostics such as policy accuracy and state-action error heatmaps.
- Add a small deterministic environment with stochastic transitions.
- Document paired statistical comparisons across common seed streams.

## Later

- Separate the environment protocol from `ChainMDP` without adding a large framework dependency.
- Evaluate adaptive multipliers on several tabular tasks and hyperparameter grids.
- Add linear function approximation only after the tabular API and evaluation protocol stabilize.
- Publish package-index metadata and badges only after a real release exists.

Requests should include a concrete use case, an evaluation plan, and the maintenance cost of any
new dependency.
