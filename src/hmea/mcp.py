"""Stdio MCP server for local HMEA diagnostics.

Messages are newline-delimited JSON-RPC 2.0. The server writes only protocol
messages to stdout. It does not open a network connection. Tool runs use the
deterministic experiment parser, not a language model.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any, TextIO

from . import __version__
from .diagnostics import (
    audit_policy,
    list_policies,
    parse_experiment_spec,
    run_experiment,
    summarize_paired_difference,
)
from .spec import SpecError

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
PREFERRED_PROTOCOL = PROTOCOL_VERSIONS[0]
SERVER_NAME = "hmea-diagnostics"

_TOOLS: dict[str, dict[str, Any]] = {
    "list_policies": {
        "description": (
            "List built-in update policies, their bounds, and whether each policy "
            "is part of the default four-rule comparison. Does not train."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "handler": lambda _arguments: list_policies(),
    },
    "audit_policy": {
        "description": (
            "Report the declared structure of one built-in policy. "
            "This is an interface check, not a claim about learning performance."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"policy": {"type": "string"}},
            "required": ["policy"],
            "additionalProperties": False,
        },
        "handler": lambda arguments: audit_policy(arguments["policy"]),
    },
    "parse_experiment_spec": {
        "description": (
            "Compile an experiment description with the deterministic parser. "
            "No language model or network call is used. The result includes a "
            "spec id that identifies the resolved parameters."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        "handler": lambda arguments: parse_experiment_spec(arguments["text"]),
    },
    "run_experiment": {
        "description": (
            "Train the policies in a deterministic experiment description or a "
            "structured spec and return bias, RMSE, and any paired differences. "
            "Provide exactly one of text or spec. The run uses the seeds in the spec."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "spec": {"type": "object"},
            },
            "additionalProperties": False,
        },
        "handler": lambda arguments: run_experiment(
            text=arguments.get("text"),
            spec=arguments.get("spec"),
        ),
    },
    "paired_mean_difference": {
        "description": (
            "Estimate a candidate-minus-reference mean with a paired percentile "
            "bootstrap. Inputs are aligned per-seed samples. This does not train."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "candidate": {"type": "array", "items": {"type": "number"}},
                "reference": {"type": "array", "items": {"type": "number"}},
                "n_resamples": {"type": "integer", "minimum": 1},
                "seed": {"type": "integer", "minimum": 0},
            },
            "required": ["candidate", "reference"],
            "additionalProperties": False,
        },
        "handler": lambda arguments: summarize_paired_difference(
            arguments["candidate"],
            arguments["reference"],
            n_resamples=arguments.get("n_resamples", 10_000),
            seed=arguments.get("seed", 0),
        ),
    },
}


def tool_definitions() -> list[dict[str, Any]]:
    """Return MCP tool metadata without handler callables."""
    return [
        {
            "name": name,
            "description": spec["description"],
            "inputSchema": spec["inputSchema"],
        }
        for name, spec in _TOOLS.items()
    ]


def dispatch_message(message: Mapping[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC message and return a response, or None for notifications."""
    if not isinstance(message, Mapping):
        return _error(None, -32600, "invalid request")
    method = message.get("method")
    has_id = "id" in message
    request_id = message.get("id")
    if not isinstance(method, str):
        if has_id:
            return _error(request_id, -32600, "invalid request")
        return None
    if method.startswith("notifications/"):
        return None
    if not has_id:
        return None
    params = message.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return _error(request_id, -32602, "invalid params")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": request_id, "result": _initialize(params)}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": tool_definitions()}}
    if method == "tools/call":
        return {"jsonrpc": "2.0", "id": request_id, "result": _call_tool(params)}
    return _error(request_id, -32601, "method not found")


def dispatch_line(line: str) -> dict[str, Any] | None:
    """Parse one line and dispatch it."""
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        return _error(None, -32700, "parse error")
    if isinstance(message, list):
        return _error(None, -32600, "invalid request")
    return dispatch_message(message)


def serve(stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    """Read newline-delimited JSON-RPC from stdin until EOF."""
    incoming = sys.stdin if stdin is None else stdin
    outgoing = sys.stdout if stdout is None else stdout
    for line in incoming:
        if not line.strip():
            continue
        response = dispatch_line(line)
        if response is not None:
            outgoing.write(json.dumps(response, allow_nan=False) + "\n")
            outgoing.flush()
    return 0


def main() -> int:
    """Run the stdio server."""
    return serve()


def _initialize(params: Mapping[str, Any]) -> dict[str, Any]:
    requested = params.get("protocolVersion")
    protocol = requested if requested in PROTOCOL_VERSIONS else PREFERRED_PROTOCOL
    return {
        "protocolVersion": protocol,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": SERVER_NAME, "version": __version__},
        "instructions": (
            "HMEA diagnostics run locally. parse_experiment_spec and run_experiment "
            "use the deterministic parser and do not call a language model."
        ),
    }


def _call_tool(params: Mapping[str, Any]) -> dict[str, Any]:
    name = params.get("name")
    arguments = params.get("arguments", {})
    if not isinstance(name, str) or name not in _TOOLS:
        return _tool_error(f"unknown tool: {name}")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return _tool_error("tool arguments must be an object")
    required = _TOOLS[name]["inputSchema"].get("required", [])
    missing = [item for item in required if item not in arguments]
    if missing:
        return _tool_error(f"missing argument: {missing[0]}")
    try:
        payload = _TOOLS[name]["handler"](arguments)
    except (SpecError, TypeError, ValueError, KeyError) as exc:
        return _tool_error(str(exc))
    return _tool_result(payload)


def _tool_result(payload: dict[str, object]) -> dict[str, Any]:
    return {
        "content": [
            {"type": "text", "text": json.dumps(payload, indent=2, allow_nan=False) + "\n"}
        ],
        "isError": False,
    }


def _tool_error(message: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": message}],
        "isError": True,
    }


def _error(request_id: object, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


if __name__ == "__main__":
    raise SystemExit(main())
