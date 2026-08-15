import json

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
    }
    assert payload["results"][0]["policy"] == "Immediate"
    assert payload["results"][0]["bias_ci95"] == 0.0


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "not-a-number"])
def test_main_rejects_invalid_sizes(value):
    with pytest.raises(SystemExit):
        main(["--steps", value])
