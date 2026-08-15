import csv
import json
from io import StringIO

import pytest

from hmea.cli import main, run_comparison


def test_run_comparison_returns_requested_policy():
    rows = run_comparison(
        n_steps=40,
        n_seeds=2,
        log_every=20,
        seed=3,
        policy="baseline",
    )

    assert len(rows) == 1
    assert rows[0]["policy"] == "Baseline"
    assert rows[0]["uses_current_td_error"] is False
    assert rows[0]["bias_ci95"] >= 0.0
    assert rows[0]["rmse_ci95"] >= 0.0


def test_main_prints_readable_table(capsys):
    assert (
        main(
            [
                "--steps",
                "40",
                "--seeds",
                "2",
                "--log-every",
                "20",
                "--policy",
                "fixed-lag",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "HMEA comparison" in output
    assert "Fixed lag" in output
    assert "Current TD error" in output


def test_main_prints_machine_readable_json(capsys):
    assert (
        main(
            [
                "--steps",
                "30",
                "--seeds",
                "1",
                "--log-every",
                "10",
                "--policy",
                "immediate",
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["configuration"] == {
        "steps": 30,
        "seeds": 1,
        "log_every": 10,
        "seed": 0,
        "states": 8,
        "gamma": 0.99,
        "reward_noise": 0.1,
    }
    assert payload["schema_version"] == 1
    assert payload["software"]["hmea"] == "0.2.0"
    assert payload["results"][0]["policy"] == "Immediate"
    assert payload["results"][0]["bias_ci95"] == 0.0


def test_main_prints_csv_with_configuration(capsys):
    assert (
        main(
            [
                "--steps",
                "20",
                "--seeds",
                "1",
                "--log-every",
                "10",
                "--policy",
                "baseline",
                "--states",
                "5",
                "--gamma",
                "0.9",
                "--reward-noise",
                "0",
                "--format",
                "csv",
            ]
        )
        == 0
    )

    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert len(rows) == 1
    assert rows[0]["policy"] == "Baseline"
    assert rows[0]["states"] == "5"
    assert rows[0]["gamma"] == "0.9"
    assert rows[0]["reward_noise"] == "0.0"
    assert rows[0]["hmea_version"] == "0.2.0"


def test_main_writes_json_file(tmp_path, capsys):
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "--steps",
                "20",
                "--seeds",
                "1",
                "--log-every",
                "10",
                "--policy",
                "baseline",
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    assert json.loads(output.read_text(encoding="utf-8"))["results"][0]["policy"] == "Baseline"
    assert "Wrote json results" in capsys.readouterr().out


def test_main_rejects_conflicting_output_flags():
    with pytest.raises(SystemExit):
        main(["--json", "--format", "csv"])


def test_main_prints_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == "hmea-compare 0.2.0"


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "not-a-number"])
def test_main_rejects_invalid_sizes(value):
    with pytest.raises(SystemExit):
        main(["--steps", value])


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--states", "1"),
        ("--gamma", "1"),
        ("--gamma", "nan"),
        ("--reward-noise", "-0.1"),
        ("--reward-noise", "inf"),
    ],
)
def test_main_rejects_invalid_environment_values(option, value):
    with pytest.raises(SystemExit):
        main([option, value])
