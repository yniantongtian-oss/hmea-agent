# Method and interpretation

## Update rule

The package applies a scalar multiplier to the standard tabular Q-learning update:

```text
delta_t = r_t + gamma * max_a Q(s_{t+1}, a) - Q(s_t, a_t)
Q(s_t, a_t) <- Q(s_t, a_t) + alpha * m_t * delta_t
```

`m_t = 1` is ordinary Q-learning. The included policies change only how `m_t` is chosen.

| Policy | Signal used for `m_t` |
|---|---|
| `NoModulation` | Constant one |
| `OnlineModulation` | Current TD error |
| `ClippedModulation` | Current TD error, with narrower output bounds |
| `LaggedModulation` | TD error from a fixed number of earlier steps |
| `HomeostaticModulation` | Previous value of a leaky TD-error state |

The tanh-based policies use:

```text
m_t = clip(1 + beta * tanh(x_t / tau), lower, upper)
```

The homeostatic state tracks deviation from a setpoint:

```text
x_{t+1} = x_t + eta * ((delta_t - setpoint) - x_t)
```

Its multiplier is computed from `x_t` before the current TD error is incorporated.

## Why immediate modulation can shift values

When the multiplier is an increasing function of the same TD error it scales, positive and
negative errors receive different effective step sizes. The update contains
`m(delta_t) * delta_t`, so its average need not match a constant-step update even when the
unscaled errors average near zero.

The included chain experiment demonstrates a large upward shift for one fixed online policy.
The delayed policies stay near the baseline in that experiment. Temporal delay alone does not
establish independence in every environment, and the result should not be generalized without
new experiments or analysis.

## Environment and baseline

`ChainMDP` has `n_states` positions and two actions: left and right. Entering the rightmost
state yields reward one and ends the episode. Non-terminal rewards may include zero-mean
Gaussian noise. `optimal_q()` solves the expected tabular model by value iteration.

Potential-based shaping is accepted as a transition callback. For episodic policy-invariance
arguments, use a terminal potential of zero. The implementation does not infer whether an
arbitrary callback has that form.

## Logged metrics

At each log point the agent records values for every seed:

- **signed bias**: mean of `Q - Q*` over state-action pairs;
- **RMSE**: square root of the mean squared state-action error;
- **start value**: maximum action value at state zero.

Signed bias is useful for detecting a directional shift, but opposite errors can cancel. Read it
together with RMSE and, for detailed work, the returned final `q_values` array.

## Reproducibility

Each replica has its own action-selection generator, and reward noise uses a separate generator;
all are spawned from a shared `SeedSequence`. Runs with the same configuration and seed are
deterministic on the tested NumPy execution path. The comparison uses common streams across
policies to reduce irrelevant variation.

The plotted confidence intervals use seeded replicas as the sampling units. They summarize
run-to-run variation in this simulation; they are not uncertainty intervals over all possible
environments or hyperparameters.

## Paired reference comparisons

This section describes the default branch; the paired report is not included in v0.2.0.

The opt-in `--compare-to` report compares each candidate with a named reference policy using
the final RMSE and signed bias from matching seeded replicas. For each metric, it computes:

```text
d_i = candidate_metric_i - reference_metric_i
mean_difference = mean(d_i)
```

Pairing happens before summarization. Subtracting separately reported confidence limits, or
judging whether two marginal intervals overlap, is not the method used here. Negative RMSE
differences indicate lower estimation error in the evaluated configuration. A negative signed
bias difference indicates a downward shift in estimates; it does not by itself indicate lower
absolute bias or lower RMSE.

The interval uses a paired percentile bootstrap: sample `n_pairs` complete differences with
replacement, calculate their mean, and repeat 10,000 times. The 2.5th and 97.5th percentiles of
those bootstrap means form the reported approximate 95% interval. The command uses bootstrap
seed `0`, separate from the training root seed. The Python helper
`hmea.statistics.paired_mean_difference` exposes `n_resamples` and `seed` for bootstrap control.
The runner `hmea.cli.run_paired_comparison` names these arguments `bootstrap_resamples` and
`bootstrap_seed`, keeping them separate from its training `seed` parameter.

At least two pairs are required. Small samples can give unstable intervals, and a larger
bootstrap resample count does not replace collecting more seeded replicas. When all observed
differences are identical, the interval collapses; `observed_zero_variance` records this case.
It describes the observed sample, not certainty about unseen runs.

These intervals are exploratory and are not simultaneous confidence intervals across policies
or metrics. There is no multiple-comparison correction, p-value, or automatic winner selection.
An interval containing zero is inconclusive, not an equivalence test. Inspect absolute RMSE and
bias alongside differences, and evaluate additional configurations before generalizing.

### What shared seed streams control

Candidate and reference runs use the same root seed, number of seeded replicas, and experiment
configuration. This shares the underlying random draws, but different update rules can still
produce different action choices and state trajectories. Common streams do not guarantee lower
variance for every comparison.

Keep the seed count fixed when reproducing a paired run. Changing it changes the mapping of
random streams; a larger run is not a simple extension of the earlier seed sample. The comparison
report records the training configuration and bootstrap metadata so that the complete experiment
can be repeated.

See [Comparing policies](comparing-policies.md) for commands and output-field interpretation.

## References

- Watkins, C. J. C. H. and Dayan, P. (1992),
  [Q-learning](https://doi.org/10.1007/BF00992698).
- Ng, A. Y., Harada, D., and Russell, S. (1999),
  [Policy Invariance Under Reward Transformations: Theory and Application to Reward
  Shaping](https://people.eecs.berkeley.edu/~russell/publications.html).
- Sutton, R. S. and Barto, A. G. (2018),
  [Reinforcement Learning: An Introduction, second edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/).
