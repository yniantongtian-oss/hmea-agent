import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

from hmea.diagnostics import run_experiment
from hmea.mcp import SERVER_NAME, dispatch_line, dispatch_message, main, serve
from hmea.spec import SpecError


def _request(method, request_id=1, params=None):
    message = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        message["params"] = params
    return message


def test_initialize_lists_tools_and_calls_the_parser():
    initialized = dispatch_message(
        _request(
            "initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        )
    )
    assert initialized["result"]["protocolVersion"] == "2024-11-05"
    assert initialized["result"]["serverInfo"]["name"] == SERVER_NAME
    assert initialized["result"]["serverInfo"]["version"] == "0.2.0"

    fallback = dispatch_message(_request("initialize", params={"protocolVersion": "1999-01-01"}))
    assert fallback["result"]["protocolVersion"] == "2025-06-18"
    assert dispatch_message({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert dispatch_message(_request("ping", request_id=2))["result"] == {}

    listed = dispatch_message(_request("tools/list", request_id=3))
    names = [tool["name"] for tool in listed["result"]["tools"]]
    assert names == [
        "list_policies",
        "audit_policy",
        "parse_experiment_spec",
        "run_experiment",
        "paired_mean_difference",
    ]

    parsed = dispatch_message(
        _request(
            "tools/call",
            request_id=4,
            params={
                "name": "parse_experiment_spec",
                "arguments": {"text": "baseline, 10 steps, 1 seed"},
            },
        )
    )
    body = json.loads(parsed["result"]["content"][0]["text"])
    assert parsed["result"]["isError"] is False
    assert body["spec"]["policies"] == ["baseline"]


def test_tool_and_protocol_errors():
    missing = dispatch_message(
        _request("tools/call", params={"name": "audit_policy", "arguments": {}})
    )
    assert missing["result"]["isError"] is True
    assert "missing argument" in missing["result"]["content"][0]["text"]

    unknown = dispatch_message(_request("tools/call", params={"name": "missing", "arguments": {}}))
    assert unknown["result"]["isError"] is True

    failed = dispatch_message(
        _request(
            "tools/call",
            params={"name": "run_experiment", "arguments": {"text": "hello"}},
        )
    )
    assert failed["result"]["isError"] is True

    both = dispatch_message(
        _request(
            "tools/call",
            params={
                "name": "run_experiment",
                "arguments": {"text": "baseline, 10 steps, 1 seed", "spec": {"steps": 10}},
            },
        )
    )
    assert both["result"]["isError"] is True

    assert dispatch_line("not-json")["error"]["code"] == -32700
    assert dispatch_line("[]")["error"]["code"] == -32600
    assert dispatch_message(_request("unknown/method"))["error"]["code"] == -32601
    assert dispatch_message({"jsonrpc": "2.0", "id": 1})["error"]["code"] == -32600
    assert dispatch_message({"jsonrpc": "2.0", "method": "ping"}) is None
    assert dispatch_message(_request("ping", params=[]))["error"]["code"] == -32602


def test_serve_writes_one_response_per_request():
    stdin = StringIO(
        "\n".join(
            [
                "",
                json.dumps(_request("ping", request_id=7)),
                json.dumps(
                    _request(
                        "tools/call",
                        request_id=8,
                        params={
                            "name": "paired_mean_difference",
                            "arguments": {
                                "candidate": [1, 2],
                                "reference": [0, 0],
                                "n_resamples": 20,
                                "seed": 1,
                            },
                        },
                    )
                ),
            ]
        )
        + "\n"
    )
    stdout = StringIO()

    assert serve(stdin, stdout) == 0
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert [item["id"] for item in responses] == [7, 8]
    summary = json.loads(responses[1]["result"]["content"][0]["text"])
    assert summary["n_pairs"] == 2
    assert summary["mean_difference"] == 1.5


def test_protocol_edges_and_structured_runs(monkeypatch):
    assert dispatch_message("nope")["error"]["code"] == -32600  # type: ignore[arg-type]
    assert dispatch_message({"jsonrpc": "2.0"}) is None
    assert dispatch_message(_request("ping", params=None))["result"] == {}
    listed = dispatch_message(
        _request("tools/call", params={"name": "list_policies", "arguments": None})
    )
    assert listed["result"]["isError"] is False
    bad_arguments = dispatch_message(
        _request("tools/call", params={"name": "audit_policy", "arguments": ["baseline"]})
    )
    assert bad_arguments["result"]["isError"] is True

    stdin = StringIO(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    stdout = StringIO()
    assert serve(stdin, stdout) == 0
    assert stdout.getvalue() == ""

    monkeypatch.setattr("sys.stdin", StringIO(""))
    assert main() == 0

    report = run_experiment(spec={"policy": "baseline", "steps": 8, "seeds": 1, "reward_noise": 0})
    assert report["parse_source"] == "structured"
    assert report["results"][0]["policy"] == "Baseline"
    with pytest.raises(SpecError, match="exactly one"):
        run_experiment()
    with pytest.raises(SpecError, match="must be text"):
        run_experiment(text=1)  # type: ignore[arg-type]


def test_module_entry_point_speaks_json_rpc():
    payload = (
        "\n".join(
            [
                json.dumps(
                    _request(
                        "initialize",
                        params={"protocolVersion": "2025-06-18", "capabilities": {}},
                    )
                ),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                json.dumps(_request("tools/list", request_id=2)),
            ]
        )
        + "\n"
    )
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "hmea.mcp"],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
        cwd=root,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    responses = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    assert len(responses) == 2
    assert responses[0]["result"]["serverInfo"]["name"] == SERVER_NAME
    assert {tool["name"] for tool in responses[1]["result"]["tools"]} >= {"run_experiment"}
    assert completed.stderr == ""
