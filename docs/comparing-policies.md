# Comparing policies

Use a paired comparison to ask whether an update rule changes estimation error relative to a
reference under one controlled experiment. HMEA compares final metrics from matching seeded
replicas instead of treating the two policies as unrelated samples.

The paired CLI and Python APIs described here are implemented in the development version and
are not part of the v0.2.0 release. From a source checkout containing these changes, install
with `python -m pip install -e .`. A v0.2.0 installation, or the default branch before these
changes are merged, does not support `--compare-to`.

## Start with the baseline

After installing the development checkout, run:

```bash
python -m hmea --compare-to baseline
```

The default is 5,000 training steps and 16 seeded replicas per policy. The table first reports
absolute results for the policies, then paired differences for each candidate against ordinary
Q-learning. The reference is trained once and reused for every candidate comparison.

To focus on one candidate:

```bash
python -m hmea --policy fixed-lag --compare-to baseline
```

The reference runs automatically even when it is not selected by `--policy`. Available reference
names are `baseline`, `immediate`, `fixed-lag`, and `leaky-state`. Selecting only the reference
itself, such as `--policy baseline --compare-to baseline`, is rejected because it would not
compare two different policies. Paired mode also requires at least two seeds.

The installed `hmea-compare` command accepts the same arguments as `python -m hmea`.

## Read differences alongside absolute results

Every difference is **candidate minus reference**. Do not use the sign alone to choose a policy.

| Metric or field | Interpretation |
|---|---|
| `rmse` | A negative mean difference means lower state-action estimation error for the candidate. |
| `bias` | A negative mean difference means a downward shift in signed bias, not necessarily a better estimate. |
| `ci95_low`, `ci95_high` | Approximate 95% paired percentile bootstrap interval for the mean difference. |
| `n_pairs` | Number of matching seeded replicas used in the comparison. |
| `observed_zero_variance` | All observed paired differences are identical; the interval collapses, but unseen-run uncertainty is not eliminated. |
| `per_seed_differences` | Candidate-minus-reference values for each pair, retained in JSON for further inspection. |

If an RMSE interval lies below zero, it supports lower error for the candidate under this
configuration and interval method. If it spans zero, the result is inconclusive; it does not
establish that the methods are equivalent. A downward signed-bias shift can help an overestimating
policy or worsen an underestimating one, so also inspect absolute bias and RMSE.

Intervals describe seeded-run variation in the included chain environment. They are exploratory,
not a simultaneous guarantee across candidates and metrics. The report does not perform an
equivalence test, correct for multiple comparisons, or select a winner.

## Run a larger comparison and keep the evidence

```bash
python -m hmea --policy fixed-lag --compare-to baseline --full --format json --output comparison.json
```

The `--full` preset uses 80,000 steps, 128 seeds, and a logging interval of 400 steps. Explicit
`--steps`, `--seeds`, or `--log-every` arguments override their respective preset values. For a
different training budget:

```bash
python -m hmea --policy fixed-lag --compare-to baseline --steps 20000 --seeds 64 --format json --output comparison.json
```

Choose the environment configuration before interpreting the results. Parameters such as
`--states`, `--gamma`, and `--reward-noise` can change the conclusion. A result from one setting
does not establish performance across a parameter sweep or other environments.

These commands do not need Matplotlib or rewrite the repository's reference figure. They write
only the requested output file; use a new filename to retain earlier results.

## Choose an export format

For a spreadsheet or a table-processing script:

```bash
python -m hmea --compare-to baseline --format csv --output comparison.csv
```

- **Table:** absolute results followed by paired differences and their intervals.
- **JSON:** absolute results, paired summaries, and the per-seed difference arrays.
- **CSV:** one paired summary row per candidate and metric, with comparison, experiment, and
  software metadata repeated on each row. Per-seed arrays are omitted.

Paired JSON uses `schema_version: 2` and these top-level fields:

| Field | Contents |
|---|---|
| `software` | HMEA and NumPy versions. |
| `configuration` | Training budget, seed count, root seed, and environment settings. |
| `results` | Absolute per-policy summaries, including the reference. |
| `comparison` | Reference display name, difference direction, and interval settings. |
| `paired_comparisons` | Candidate-by-metric summaries and their per-seed differences. |

The comparison metadata specifies `difference_direction: candidate_minus_reference`,
`interval_method: paired_percentile_bootstrap`, `confidence_level: 0.95`,
`bootstrap_resamples: 10000`, and `bootstrap_seed: 0`. The reference is recorded in
`reference_policy` using its display label.

Without `--compare-to`, JSON remains schema version 1 and CSV retains its existing one-row-per-policy
format. Scripts should check the JSON schema version before interpreting a report. Do not feed
paired CSV into a parser that expects the ordinary CSV columns.

## Repeat a result correctly

Keep the complete command, output file, HMEA version, and NumPy version. To reproduce a run,
use the same root `--seed`, seed count, training budget, environment settings, and policy choices.
For development builds, also retain the source commit from `git rev-parse HEAD` and any local
changes; the package version alone does not identify an unreleased checkout.

The pairing shares random draws across policies, not necessarily actions or trajectories: an
update rule can change subsequent choices. Changing the seed count also changes how random
streams are assigned, so a larger seed count does not just append samples to a smaller run.

The default interval uses 10,000 bootstrap resamples with a separate fixed seed of `0`.
More resamples refine the bootstrap calculation; they do not replace additional training seeds.
For programmatic control, `hmea.cli.run_paired_comparison` exposes `bootstrap_resamples` and
`bootstrap_seed`. The lower-level `hmea.statistics.paired_mean_difference` helper accepts
`n_resamples` and `seed` for the same bootstrap controls. The runner's separate `seed` parameter
controls training, not bootstrap sampling.

The [method notes](method.md#paired-reference-comparisons) describe the estimator, sampling unit,
and limitations.
