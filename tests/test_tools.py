import json

import pytest

from hmea.tools import main


def test_list_and_audit_commands(capsys):
    assert main(["list-policies"]) == 0
    payload = json.loads(capsys.readouterr().out)
    by_id = {item["id"]: item for item in payload["policies"]}
    assert by_id["clipped"]["in_default_comparison"] is False
    assert by_id["baseline"]["in_default_comparison"] is True
    assert by_id["immediate"]["uses_current_td_error"] is True

    assert main(["audit", "leaky-state"]) == 0
    audit = json.loads(capsys.readouterr().out)
    assert audit["audit"]["uses_current_td_error"] is False
    assert audit["audit"]["lag"] == 1
    assert audit["parameters"]["eta"] == 0.05
    assert audit["label"] == "Leaky state"


def test_parse_and_run_commands(capsys):
    assert main(["parse", "compare clipped to baseline, 10 steps, 2 seeds"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["parse_source"] == "deterministic"
    assert parsed["spec"]["policies"] == ["clipped"]
    assert parsed["spec"]["compare_to"] == "baseline"

    assert main(["run", "baseline, 12 steps, 1 seed, reward noise 0, log every 6"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 3
    assert report["results"][0]["policy"] == "Baseline"
    assert report["paired_comparisons"] is None


def test_paired_difference_command(capsys):
    assert (
        main(
            [
                "paired-difference",
                "--candidate",
                "1,2,3",
                "--reference",
                "1,1,1",
                "--resamples",
                "50",
                "--seed",
                "2",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_pairs"] == 3
    assert payload["mean_difference"] == pytest.approx(1.0)
    assert payload["per_seed_differences"] == [0.0, 1.0, 2.0]


def test_serve_command_uses_the_mcp_server(monkeypatch):
    monkeypatch.setattr("hmea.tools.serve", lambda: 0)
    assert main(["serve"]) == 0


def test_tool_errors_and_version(capsys):
    assert main(["audit", "missing"]) == 2
    assert "unknown policy" in capsys.readouterr().err
    assert main(["parse", "hello"]) == 2
    with pytest.raises(SystemExit):
        main(["paired-difference", "--candidate", "nope", "--reference", "1,2"])
    with pytest.raises(SystemExit):
        main(["paired-difference", "--candidate", ",", "--reference", "1"])
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "hmea-tools" in capsys.readouterr().out
