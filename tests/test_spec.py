import json
import socket

import numpy as np
import pytest

from hmea.cli import run_comparison, run_paired_comparison
from hmea.spec import ExperimentSpec, SpecError, compile_spec, main, run_spec


class FakeModel:
    def __init__(self, output):
        self.output = output
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def test_natural_language_comparison_uses_cli_defaults():
    spec = compile_spec("compare immediate to baseline with 40 steps and 2 seeds")

    assert spec.parse_source == "deterministic"
    assert spec.policies == ("immediate",)
    assert spec.compare_to == "baseline"
    assert spec.steps == 40
    assert spec.seeds == 2
    assert spec.log_every == 100
    assert spec.seed == 0
    assert spec.states == 8
    assert spec.gamma == 0.99
    assert spec.reward_noise == 0.1
    assert spec.bootstrap_resamples == 10_000
    assert spec.bootstrap_seed == 0


def test_equivalent_text_and_json_share_a_spec_id():
    text = compile_spec("compare immediate to baseline with 40 steps and 2 seeds")
    document = compile_spec(
        json.dumps(
            {
                "policies": ["immediate"],
                "compare_to": "baseline",
                "steps": 40,
                "seeds": 2,
            }
        )
    )

    assert text.spec_id == document.spec_id
    assert text.canonical() == document.canonical()


def test_key_value_and_bracket_lists():
    spec = compile_spec(
        "\n".join(
            [
                "policies: [immediate, clipped]",
                "steps: 10",
                "seeds: 2",
                "compare_to: baseline",
                "gamma: 0.9",
                "reward_noise: 0",
            ]
        )
    )

    assert spec.policies == ("immediate", "clipped")
    assert spec.compare_to == "baseline"
    assert spec.gamma == 0.9
    assert spec.reward_noise == 0.0


def test_full_reference_preset_keeps_explicit_overrides():
    preset = compile_spec("full reference")
    mixed = compile_spec("full reference, 1000 steps, 4 seeds")

    assert (preset.steps, preset.seeds, preset.log_every) == (80_000, 128, 400)
    assert mixed.steps == 1000
    assert mixed.seeds == 4
    assert mixed.log_every == 400


def test_aliases_comments_and_separate_seeds():
    spec = compile_spec("# ignore this\nleaky state vs delayed, 30 steps, 4 seeds, seed 4")

    assert spec.policies == ("leaky-state",)
    assert spec.compare_to == "fixed-lag"
    assert spec.seed == 4
    assert spec.seeds == 4


def test_bootstrap_seed_does_not_replace_the_training_seed():
    spec = compile_spec("baseline, bootstrap seed 3, seed 4, 20 steps")

    assert spec.bootstrap_seed == 3
    assert spec.seed == 4
    assert spec.steps == 20


def test_all_expands_to_the_default_comparison():
    spec = compile_spec("compare all policies to baseline with 20 steps and 2 seeds")

    assert spec.policies == ("baseline", "immediate", "fixed-lag", "leaky-state")
    assert spec.compare_to == "baseline"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("   ", "empty"),
        ("hello", "could not recognize"),
        ("compare baseline to baseline, 10 steps, 2 seeds", "different from the reference"),
        ("compare immediate to baseline with 10 steps and 1 seed", "at least 2 seeds"),
        ("40 steps and 50 steps, baseline", "repeated spec field: steps"),
        ("all and immediate, 10 steps", "cannot be combined"),
        ("immediate and immediate, 10 steps", "duplicates"),
        ("compare banana to baseline, 10 steps, 2 seeds", "candidate policy"),
        ("gamma 1, baseline, 10 steps", "gamma"),
        ("reward noise -1, baseline, 10 steps", "reward_noise"),
        ("{", "invalid JSON"),
        ("[1, 2]", "JSON spec must be an object"),
        ("{}", "no fields"),
        ("clipped. compare immediate to baseline with 10 steps and 2 seeds", "ambiguous"),
    ],
)
def test_invalid_specs_are_rejected(text, message):
    with pytest.raises(SpecError, match=message):
        compile_spec(text)


def test_model_json_is_used_when_it_validates():
    model = FakeModel('Here is the spec: {"policies":["clipped"],"steps":10,"seeds":1}')
    spec = compile_spec("please run something", model=model)

    assert spec.parse_source == "llm"
    assert spec.policies == ("clipped",)
    assert spec.steps == 10
    assert "please run something" in model.prompts[0]
    assert "JSON" in model.prompts[0]


def test_model_failure_falls_back_to_the_deterministic_parser():
    direct = compile_spec("baseline, 10 steps, 1 seed")
    for output in ('{"unknown_field": 1}', "not json", RuntimeError("offline")):
        spec = compile_spec("baseline, 10 steps, 1 seed", model=FakeModel(output))
        assert spec.parse_source == "deterministic_fallback"
        assert spec.spec_id == direct.spec_id


def test_model_and_parser_failures_are_reported_together():
    with pytest.raises(SpecError, match="language model output was not used"):
        compile_spec("hello", model=FakeModel("nope"))


def test_compile_spec_rejects_non_text_and_does_not_use_sockets(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("network is disabled")

    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)

    with pytest.raises(SpecError, match="must be text"):
        compile_spec(None)  # type: ignore[arg-type]
    spec = compile_spec("baseline, 12 steps, 1 seed, reward noise 0")
    report = run_spec(spec)
    assert report["schema_version"] == 3
    assert report["results"][0]["policy"] == "Baseline"
    assert report["paired_comparisons"] is None


def test_run_spec_matches_the_comparison_helpers_and_is_reproducible():
    absolute = compile_spec("baseline, 24 steps, 2 seeds, reward noise 0, seed 3, log every 12")
    first = run_spec(absolute)
    second = run_spec(absolute)
    direct = run_comparison(
        n_steps=24,
        n_seeds=2,
        log_every=12,
        seed=3,
        policies=["baseline"],
        n_states=8,
        gamma=0.99,
        sigma_r=0.0,
    )
    assert first["results"] == second["results"] == direct

    paired = compile_spec(
        "compare immediate to baseline, 20 steps, 2 seeds, reward noise 0, seed 1, "
        "100 bootstrap resamples, bootstrap seed 5"
    )
    report = run_spec(paired)
    expected = run_paired_comparison(
        n_steps=20,
        n_seeds=2,
        log_every=100,
        seed=1,
        policies=["immediate"],
        n_states=8,
        gamma=0.99,
        sigma_r=0.0,
        reference_policy="baseline",
        bootstrap_resamples=100,
        bootstrap_seed=5,
    )
    assert report["results"] == expected["results"]
    assert report["paired_comparisons"] == expected["paired_comparisons"]
    assert report["comparison"] == expected["comparison"]
    assert report["software"]["hmea"] == "0.2.0"


def test_structured_mapping_records_its_source():
    spec = ExperimentSpec.from_mapping(
        {"policy": "clipped", "steps": 10, "seeds": 1},
        parse_source="structured",
    )

    assert spec.parse_source == "structured"
    assert spec.policies == ("clipped",)
    with pytest.raises(TypeError):
        run_spec({"policies": ["clipped"]})  # type: ignore[arg-type]


def test_spec_cli_prints_compiles_and_writes(tmp_path, capsys):
    assert main(["baseline, 10 steps, 1 seed"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["spec"]["policies"] == ["baseline"]

    output = tmp_path / "spec.json"
    source = tmp_path / "spec.txt"
    source.write_text("clipped, 8 steps, 1 seed\n", encoding="utf-8")
    assert main(["--file", str(source), "--run", "--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["schema_version"] == 3
    assert written["results"][0]["policy"] == "Clipped"
    assert "Wrote spec JSON" in capsys.readouterr().err


def test_mapping_and_prose_validation_cover_rejected_fields():
    valid = ExperimentSpec.from_mapping(
        {
            "steps": np.int64(10),
            "seeds": "1",
            "log_every": "10.0",
            "gamma": "0.5",
            "full": False,
            "compare_to": None,
            "policy": "baseline",
        },
        parse_source="structured",
    )
    assert valid.steps == 10
    assert valid.log_every == 10
    assert valid.compare_to is None

    rejected = [
        ({}, "no fields"),
        ("not-a-mapping", "must be an object"),
        ({"steps": True}, "steps"),
        ({"steps": "nope"}, "steps"),
        ({"steps": "1.5"}, "steps"),
        ({"steps": 10.5}, "steps"),
        ({"steps": 0}, "steps"),
        ({"steps": float("nan")}, "steps"),
        ({"gamma": True, "steps": 10}, "gamma"),
        ({"gamma": "nope", "steps": 10}, "gamma"),
        ({"gamma": float("inf"), "steps": 10}, "gamma"),
        ({"full": "maybe", "steps": 10}, "full"),
        ({"policies": []}, "must not be empty"),
        ({"policies": 3}, "policy name"),
        ({"policies": ["immediate", "immediate"], "steps": 10}, "duplicates"),
        ({"policies": ["all", "immediate"]}, "cannot be combined"),
        ({"policy": "baseline", "policies": ["baseline"], "steps": 10}, "repeated"),
        ({"compare_to": "all", "policies": ["baseline"], "steps": 10, "seeds": 2}, "one policy"),
        ({"compare_to": "", "policy": "baseline", "steps": 10, "seeds": 1}, None),
        ({"not_a_field": 1}, "unknown spec field"),
        ({1: "baseline"}, "must be strings"),
        ({"policies": [1], "steps": 10}, "unknown policy"),
    ]
    for mapping, message in rejected:
        if message is None:
            ExperimentSpec.from_mapping(mapping, parse_source="structured")
            continue
        with pytest.raises(SpecError, match=message):
            ExperimentSpec.from_mapping(mapping, parse_source="structured")
    with pytest.raises(ValueError, match="parse source"):
        ExperimentSpec.from_mapping({"steps": 10}, parse_source="other")

    prose = [
        ("# nothing to parse", "empty"),
        ("steps: 1\nsteps: 2", "repeated spec field"),
        ('policy: "baseline"\nsteps: 10\nseeds: 1\ncompare_to: none', None),
        ("policies: []", "must not be empty"),
        ("full: false\nbaseline, 10 steps, 1 seed", None),
        (
            "compare immediate to baseline. compare clipped to baseline, 10 steps, 2 seeds",
            "more than one comparison",
        ),
        (
            "compare immediate to baseline and fixed-lag, 10 steps, 2 seeds",
            "one reference policy",
        ),
        ("compare all and immediate to baseline, 10 steps, 2 seeds", "cannot be combined"),
        ("compare immediate and immediate to baseline, 10 steps, 2 seeds", "duplicates"),
    ]
    for text, message in prose:
        if message is None:
            compile_spec(text)
            continue
        with pytest.raises(SpecError, match=message):
            compile_spec(text)

    numeric = FakeModel(123)
    fallback = compile_spec("baseline, 10 steps, 1 seed", model=numeric)
    assert fallback.parse_source == "deterministic_fallback"
    with pytest.raises(SpecError, match="RuntimeError"):
        compile_spec("hello", model=FakeModel(RuntimeError()))
    with pytest.raises(SpecError, match="not valid JSON"):
        compile_spec("hello", model=FakeModel("prefix {"))


def test_spec_cli_rejects_bad_input(tmp_path):
    with pytest.raises(SystemExit):
        main([])
    with pytest.raises(SystemExit):
        main(["baseline", "--file", str(tmp_path / "missing.txt")])
    with pytest.raises(SystemExit):
        main(["hello there"])
    with pytest.raises(SystemExit):
        main(["--version"])
    directory = tmp_path / "dir"
    directory.mkdir()
    with pytest.raises(SystemExit):
        main(["--file", str(directory)])
    with pytest.raises(SystemExit):
        main(["baseline, 10 steps, 1 seed", "--output", str(directory)])
