# Diagnostic tools

`hmea-tools` and the stdio MCP server expose the same local operations: list policies, audit
one policy, compile a spec, run a spec, and summarize a paired difference. They do not call a
language model or open a network connection.

This is available on the default branch and is not part of the v0.2.0 release.

## Command line

```bash
python -m hmea.tools list-policies
python -m hmea.tools audit leaky-state
python -m hmea.tools parse "compare clipped to baseline, 20 steps, 2 seeds"
python -m hmea.tools run "baseline, 20 steps, 2 seeds, reward noise 0"
python -m hmea.tools paired-difference --candidate 1,2,3 --reference 1,1,1 --resamples 200
python -m hmea.tools serve
```

`hmea-tools` is the installed form of the same commands. `list-policies`, `audit`, `parse`,
`run`, and `paired-difference` print JSON. Failures are reported on stderr and exit with
status 2. `serve` speaks MCP on stdin and stdout.

`audit` reports declared bounds, lag, and whether the policy says it reads the current TD
error. It does not certify that a custom rule behaves that way, and it does not train.

`run` compiles with the deterministic parser and then trains. The result uses schema version 3
and includes the `spec_id`, absolute rows, and paired rows when the spec names a reference.

## MCP server

`python -m hmea.mcp` reads newline-delimited JSON-RPC 2.0 and writes only protocol messages
to stdout. Supported protocol versions are `2025-06-18`, `2025-03-26`, and `2024-11-05`.
An unrecognized client version receives `2025-06-18`.

Implemented methods are `initialize`, `ping`, `tools/list`, and `tools/call`. Notifications,
including `notifications/initialized`, receive no response.

| Tool | Arguments | Result |
|---|---|---|
| `list_policies` | none | ids, labels, bounds, lag, and default-comparison membership |
| `audit_policy` | `policy` | structural audit and built-in parameters |
| `parse_experiment_spec` | `text` | compiled spec and `spec_id` |
| `run_experiment` | exactly one of `text` or `spec` | schema-3 training report |
| `paired_mean_difference` | `candidate`, `reference`, optional `n_resamples` and `seed` | paired mean and percentile interval |

Tool failures set `isError` and return the message as text. They are not JSON-RPC transport
errors. A malformed line, an unknown method, or invalid params use JSON-RPC error codes.

Example:

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"parse_experiment_spec","arguments":{"text":"baseline, 10 steps, 1 seed"}}}
```

The Python functions behind these tools live in `hmea.diagnostics`. `hmea.spec.compile_spec`
remains the direct API when a caller wants to supply a language model; the tools always use
the deterministic parser so a tool call cannot contact a model by itself.
