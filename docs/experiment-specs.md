# Experiment specs

An experiment spec turns a short description into one reproducible training request.
The resolved parameters, not the original wording, are what the run executes.

This is available on the default branch and is not part of the v0.2.0 release.
Install a current checkout with `python -m pip install -e .`.

## Compile a description

```bash
python -m hmea.spec "compare immediate to baseline with 5000 steps and 16 seeds"
```

The command prints JSON. It uses the deterministic parser and does not call a language model.
`--run` trains the spec. `-o` writes the JSON file. `-f` reads the description from a file.

```bash
python -m hmea.spec "baseline, 2000 steps, 8 seeds, gamma 0.95, reward noise 0" --run
```

The installed `hmea-spec` command accepts the same arguments.

Equivalent descriptions share a `spec_id`. These two compile to the same parameters and the
same id:

```text
compare immediate to baseline with 40 steps and 2 seeds
```

```json
{"policies": ["immediate"], "compare_to": "baseline", "steps": 40, "seeds": 2}
```

Omitted fields use the short CLI defaults: 5,000 steps, 16 seeds, logging every 100 steps,
seed 0, 8 states, discount 0.99, and reward noise 0.1. `full reference` uses 80,000 steps,
128 seeds, and logging every 400 steps. An explicit count replaces only that field, so
`full reference, 1000 steps` keeps the reference seed count and logging interval.

## What the parser accepts

Prose, `key: value` lines, and a JSON object are all deterministic. `#` comments are ignored
outside JSON.

| Intent | Examples |
|---|---|
| One policy | `clipped, 15 steps, 1 seed` |
| Several policies | `immediate and fixed-lag for 12 steps and 2 seeds` |
| Default comparison | `all policies` or `full reference` |
| Paired comparison | `compare immediate to baseline` or `fixed-lag vs leaky-state` |
| Training seed | `seed 4` or `root seed 4` |
| Bootstrap controls | `bootstrap seed 3` and `100 bootstrap resamples` |

Policy names accept a few aliases: `online` is `immediate`, `delayed` and `lagged` are
`fixed-lag`, and `homeostatic` or `leaky state` is `leaky-state`. `all` means the four-rule
reference comparison. `clipped` is available only when named; it is not added to `all`.

Assignments override prose when both set the same field. Lists use brackets:
`policies: [immediate, clipped]`. A comparison needs one reference. The reference cannot be
the only selected policy, and a paired spec needs at least two seeds. Unknown names,
contradictory counts, and empty text are errors.

`bootstrap seed` sets the bootstrap stream. `seed` sets the training stream. They stay
independent.

## Language model fallback

The Python API accepts an optional model. This package does not include a client and does
not open a network connection while compiling or training.

```python
from hmea.spec import compile_spec, run_spec


class LocalModel:
    def complete(self, prompt: str) -> str:
        return '{"policies": ["baseline"], "steps": 20, "seeds": 2}'


spec = compile_spec("run the baseline", model=LocalModel())
report = run_spec(spec)
```

`complete` should return one JSON object using the keys `policies`, `steps`, `seeds`,
`log_every`, `seed`, `states`, `gamma`, `reward_noise`, `compare_to`, `bootstrap_resamples`,
`bootstrap_seed`, and `full`. If the model raises, returns non-JSON, or returns a spec that
does not validate, compilation uses the deterministic parser and sets `parse_source` to
`deterministic_fallback`. A valid model result uses `llm`. The `spec_id` hashes only the
resolved parameters, so the source does not change the id of an equivalent spec.

`run_spec` never calls the model. It calls the same comparison routines as the CLI, including
paired bootstrap intervals when `compare_to` is set.

## Limits

A spec reproduces one controlled chain experiment. It does not search hyperparameters, choose
a winner, or transfer the result to another environment. Read paired intervals with the
limits in [Comparing policies](comparing-policies.md).
