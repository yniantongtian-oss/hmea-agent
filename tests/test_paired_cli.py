import csv
import json
from collections import Counter
from io import StringIO

import numpy as np
import pytest

from hmea import QAgent
from hmea.cli import main, run_comparison, run_paired_comparison


@pytest.fixture
def recorded_training(monkeypatch):
    calls = []
    offsets = {"none": 0.0, "online": 0.3, "lagged": 0.02, "homeostatic": -0.01}

    def train(agent, n_steps, n_seeds=64, log_every=500):
        calls.append(
            {
                "name": agent.modulator.name,
                "seed": agent.seed,
                "n_steps": n_steps,
                "n_seeds": n_seeds,
                "log_every": log_every,
                "n_states": agent.env.n_states,
                "gamma": agent.env.gamma,
                "sigma_r": agent.env.sigma_r,
            }
        )
        replicas = np.arange(1, n_seeds + 1, dtype=float)
        offset = offsets[agent.modulator.name]
        bias = -0.5 + 0.1 * replicas + offset * replicas
        rmse = replicas + offset * replicas
        start = 0.75 + offset * replicas
        return {
            "steps": np.array([log_every, n_steps]),
            "bias": np.array([100.0, bias.mean()]),
            "bias_by_seed": np.stack([np.full(n_seeds, 100.0), bias]),
            "rmse": np.array([100.0, rmse.mean()]),
            "rmse_by_seed": np.stack([np.full(n_seeds, 100.0), rmse]),
            "q_start": np.array([100.0, start.mean()]),
            "q_start_by_seed": np.stack([np.full(n_seeds, 100.0), start]),
            "q_values": np.zeros((n_seeds, agent.env.n_states, 2)),
            "audit": {"uses_current_td_error": agent.modulator.uses_current_td_error},
        }

    monkeypatch.setattr(QAgent, "train", train)
    return calls


def test_all_policies_train_once_with_matching_configuration(recorded_training):
    result = run_paired_comparison(
        n_steps=11,
        n_seeds=4,
        log_every=5,
        seed=17,
        n_states=5,
        gamma=0.9,
        sigma_r=0.2,
        bootstrap_resamples=101,
        bootstrap_seed=9,
    )

    assert Counter(call["name"] for call in recorded_training) == {
        "none": 1,
        "online": 1,
        "lagged": 1,
        "homeostatic": 1,
    }
    for call in recorded_training:
        assert {key: value for key, value in call.items() if key != "name"} == {
            "seed": 17,
            "n_steps": 11,
            "n_seeds": 4,
            "log_every": 5,
            "n_states": 5,
            "gamma": 0.9,
            "sigma_r": 0.2,
        }
    assert result["comparison"] == {
        "reference_policy": "Baseline",
        "difference_direction": "candidate_minus_reference",
        "interval_method": "paired_percentile_bootstrap",
        "confidence_level": 0.95,
        "bootstrap_resamples": 101,
        "bootstrap_seed": 9,
    }
    assert len(result["results"]) == 4
    assert len(result["paired_comparisons"]) == 6
    assert {row["metric"] for row in result["paired_comparisons"]} == {"rmse", "bias"}
    assert all(row["policy"] != "Baseline" for row in result["paired_comparisons"])


def test_single_policy_adds_reference_and_pairs_final_step(recorded_training):
    result = run_paired_comparison(
        n_steps=11,
        n_seeds=4,
        log_every=5,
        policy="immediate",
        bootstrap_resamples=101,
    )

    assert Counter(call["name"] for call in recorded_training) == {"none": 1, "online": 1}
    assert {row["policy"] for row in result["results"]} == {"Immediate", "Baseline"}
    assert len(result["paired_comparisons"]) == 2
    for row in result["paired_comparisons"]:
        assert row["policy"] == "Immediate"
        assert row["reference_policy"] == "Baseline"
        assert row["n_pairs"] == 4
        assert row["mean_difference"] == pytest.approx(0.75)
        np.testing.assert_allclose(row["per_seed_differences"], [0.3, 0.6, 0.9, 1.2])


def test_single_and_all_policy_requests_have_identical_paired_rows(recorded_training):
    options = {
        "n_steps": 11,
        "n_seeds": 4,
        "log_every": 5,
        "bootstrap_resamples": 101,
        "bootstrap_seed": 42,
    }
    all_policies = run_paired_comparison(**options)
    single_policy = run_paired_comparison(policy="leaky-state", **options)

    expected = [row for row in all_policies["paired_comparisons"] if row["policy"] == "Leaky state"]
    assert single_policy["paired_comparisons"] == expected


def test_nonbaseline_reference_is_trained_once_and_has_no_self_comparison(recorded_training):
    result = run_paired_comparison(
        n_steps=11,
        n_seeds=4,
        log_every=5,
        reference_policy="fixed-lag",
        bootstrap_resamples=101,
    )

    assert Counter(call["name"] for call in recorded_training)["lagged"] == 1
    assert len(result["results"]) == 4
    assert result["comparison"]["reference_policy"] == "Fixed lag"
    assert len(result["paired_comparisons"]) == 6
    assert all(row["policy"] != "Fixed lag" for row in result["paired_comparisons"])
    assert all(row["reference_policy"] == "Fixed lag" for row in result["paired_comparisons"])


def test_bootstrap_settings_leave_training_summaries_unchanged(recorded_training):
    options = {"n_steps": 11, "n_seeds": 4, "log_every": 5, "policy": "all"}
    ordinary = run_comparison(**options)
    first = run_paired_comparison(**options, bootstrap_resamples=31, bootstrap_seed=9)
    second = run_paired_comparison(**options, bootstrap_resamples=53, bootstrap_seed=24)

    assert first["results"] == second["results"] == ordinary
    for row1, row2 in zip(first["paired_comparisons"], second["paired_comparisons"], strict=True):
        assert row1["mean_difference"] == row2["mean_difference"]
        assert row1["per_seed_differences"] == row2["per_seed_differences"]


@pytest.mark.parametrize(
    "options",
    [
        {"n_seeds": 1},
        {"n_seeds": 1.5},
        {"n_seeds": True},
        {"n_seeds": np.bool_(True)},
        {"policy": "baseline", "reference_policy": "baseline"},
        {"policy": "immediate", "reference_policy": "immediate"},
        {"reference_policy": "unknown"},
        {"policy": "unknown"},
        {"bootstrap_resamples": 0},
        {"bootstrap_resamples": True},
        {"bootstrap_seed": -1},
        {"bootstrap_seed": True},
    ],
)
def test_invalid_paired_requests_are_rejected_before_training(options, recorded_training):
    arguments = {"n_steps": 11, "n_seeds": 4, "log_every": 5, **options}

    with pytest.raises((TypeError, ValueError)):
        run_paired_comparison(**arguments)

    assert recorded_training == []


def test_paired_cli_json_has_versioned_metadata_and_raw_vectors(capsys, recorded_training):
    assert (
        main(
            [
                "--steps",
                "11",
                "--seeds",
                "4",
                "--policy",
                "immediate",
                "--compare-to",
                "baseline",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 2
    assert {
        "software",
        "configuration",
        "results",
        "comparison",
        "paired_comparisons",
    } <= payload.keys()
    assert payload["comparison"]["reference_policy"] == "Baseline"
    assert payload["comparison"]["bootstrap_resamples"] == 10_000
    assert payload["comparison"]["bootstrap_seed"] == 0
    assert payload["comparison"]["interval_method"] == "paired_percentile_bootstrap"
    assert payload["comparison"]["confidence_level"] == 0.95
    assert len(payload["paired_comparisons"]) == 2
    for row in payload["paired_comparisons"]:
        assert len(row["per_seed_differences"]) == 4
        assert row["ci95_low"] <= row["ci95_high"]
    json.dumps(payload, allow_nan=False)


def test_paired_cli_csv_is_long_form_with_method_and_configuration(capsys, recorded_training):
    assert (
        main(
            [
                "--steps",
                "11",
                "--seeds",
                "4",
                "--states",
                "5",
                "--gamma",
                "0.9",
                "--reward-noise",
                "0.2",
                "--policy",
                "immediate",
                "--compare-to",
                "baseline",
                "--format",
                "csv",
            ]
        )
        == 0
    )

    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert len(rows) == 2
    assert {row["metric"] for row in rows} == {"rmse", "bias"}
    for row in rows:
        assert row["policy"] == "Immediate"
        assert row["reference_policy"] == "Baseline"
        assert row["interval_method"] == "paired_percentile_bootstrap"
        assert float(row["confidence_level"]) == 0.95
        assert int(row["bootstrap_resamples"]) == 10_000
        assert int(row["bootstrap_seed"]) == 0
        assert int(row["steps"]) == 11
        assert int(row["seeds"]) == 4
        assert int(row["states"]) == 5
        assert float(row["gamma"]) == 0.9
        assert float(row["reward_noise"]) == 0.2
        assert float(row["mean_difference"]) == pytest.approx(0.75)
        assert float(row["ci95_low"]) <= float(row["ci95_high"])
        assert "per_seed_differences" not in row


def test_paired_cli_table_explains_the_comparison(capsys, recorded_training):
    assert (
        main(["--steps", "11", "--seeds", "4", "--policy", "immediate", "--compare-to", "baseline"])
        == 0
    )

    output = capsys.readouterr().out
    assert "Immediate" in output
    assert "Baseline" in output
    assert "paired" in output.lower()
    assert "candidate" in output.lower()
    assert "reference" in output.lower()


def test_paired_cli_table_explains_collapsed_intervals(capsys, recorded_training, monkeypatch):
    original_train = QAgent.train

    def train_without_difference(agent, n_steps, n_seeds=64, log_every=500):
        log = original_train(agent, n_steps, n_seeds, log_every)
        log["bias_by_seed"] = np.zeros((2, n_seeds))
        log["rmse_by_seed"] = np.ones((2, n_seeds))
        return log

    monkeypatch.setattr(QAgent, "train", train_without_difference)
    assert (
        main(
            [
                "--steps",
                "11",
                "--seeds",
                "4",
                "--policy",
                "immediate",
                "--compare-to",
                "baseline",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "zero observed variance" in output.lower()
    assert "do not imply certainty" in output.lower()


def test_paired_cli_writes_complete_json_export(tmp_path, capsys, recorded_training):
    output = tmp_path / "paired-results.json"
    assert (
        main(
            [
                "--steps",
                "11",
                "--seeds",
                "4",
                "--policy",
                "immediate",
                "--compare-to",
                "baseline",
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert len(payload["results"]) == 2
    assert len(payload["paired_comparisons"]) == 2
    assert payload["comparison"]["reference_policy"] == "Baseline"
    assert "Wrote json results" in capsys.readouterr().out


@pytest.mark.parametrize(
    "arguments",
    [
        ["--seeds", "1", "--compare-to", "baseline"],
        ["--policy", "baseline", "--compare-to", "baseline"],
        ["--compare-to", "unknown"],
    ],
)
def test_paired_cli_reports_argument_errors(arguments, recorded_training):
    with pytest.raises(SystemExit) as exc_info:
        main(["--steps", "11", *arguments])

    assert exc_info.value.code == 2
    assert recorded_training == []


def test_without_compare_to_json_retains_schema_one(capsys, recorded_training):
    assert main(["--steps", "11", "--seeds", "1", "--policy", "baseline", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert set(payload) == {"schema_version", "software", "configuration", "results"}
    assert len(payload["results"]) == 1
    assert payload["results"][0]["policy"] == "Baseline"


def test_without_compare_to_csv_retains_existing_columns(capsys, recorded_training):
    assert main(["--steps", "11", "--seeds", "1", "--policy", "baseline", "--format", "csv"]) == 0

    reader = csv.DictReader(StringIO(capsys.readouterr().out))
    assert reader.fieldnames == [
        "policy",
        "mean_bias",
        "bias_ci95",
        "mean_rmse",
        "rmse_ci95",
        "mean_start_value",
        "uses_current_td_error",
        "steps",
        "seeds",
        "log_every",
        "seed",
        "states",
        "gamma",
        "reward_noise",
        "hmea_version",
        "numpy_version",
    ]
    assert len(list(reader)) == 1


def test_real_shared_stream_comparison_detects_reference_error_shift():
    result = run_paired_comparison(
        n_steps=20_000,
        n_seeds=24,
        log_every=999,
        policy="immediate",
        bootstrap_resamples=501,
    )

    rmse = next(row for row in result["paired_comparisons"] if row["metric"] == "rmse")
    bias = next(row for row in result["paired_comparisons"] if row["metric"] == "bias")
    assert rmse["mean_difference"] > 0.1
    assert bias["mean_difference"] > 0.1
    assert rmse["n_pairs"] == 24
    assert len(rmse["per_seed_differences"]) == 24
