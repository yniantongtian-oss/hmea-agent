"""Diagnostic operations shared by the tools CLI and the MCP server.

These functions run locally. They do not open a network connection and do not
call a language model. Experiment text is compiled with the deterministic parser.
"""

from __future__ import annotations

from collections.abc import Sequence

from .cli import DEFAULT_COMPARISON_POLICIES, _policy_factories
from .spec import ExperimentSpec, SpecError, compile_spec, run_spec
from .statistics import paired_mean_difference
from .validation import audit


def list_policies() -> dict[str, object]:
    """Describe built-in policies without training them."""
    policies = []
    for policy_id, (label, factory) in _policy_factories().items():
        modulator = factory()
        policies.append(
            {
                "id": policy_id,
                "label": label,
                "in_default_comparison": policy_id in DEFAULT_COMPARISON_POLICIES,
                "uses_current_td_error": modulator.uses_current_td_error,
                "lag": modulator.lag,
                "bounds": [float(bound) for bound in modulator.bounds],
            }
        )
    return {"policies": policies}


def audit_policy(policy: str) -> dict[str, object]:
    """Return the structural audit for one built-in policy."""
    factories = _policy_factories()
    if not isinstance(policy, str) or policy not in factories:
        raise SpecError(f"unknown policy: {policy}")
    label, factory = factories[policy]
    modulator = factory()
    parameters: dict[str, object] = {
        "lag": modulator.lag,
        "bounds": [float(bound) for bound in modulator.bounds],
        "uses_current_td_error": modulator.uses_current_td_error,
    }
    for name in ("beta", "tau"):
        if hasattr(modulator, name):
            parameters[name] = float(getattr(modulator, name))
    state = getattr(modulator, "state", None)
    if state is not None:
        parameters["eta"] = float(state.eta)
        parameters["setpoint"] = float(state.setpoint)
    return {
        "id": policy,
        "label": label,
        "in_default_comparison": policy in DEFAULT_COMPARISON_POLICIES,
        "audit": audit(modulator),
        "parameters": parameters,
    }


def parse_experiment_spec(text: str) -> dict[str, object]:
    """Compile experiment text with the deterministic parser."""
    return compile_spec(text).to_dict()


def run_experiment(
    *,
    text: str | None = None,
    spec: dict[str, object] | None = None,
) -> dict[str, object]:
    """Train a deterministic spec or an explicit mapping.

    Provide exactly one of ``text`` or ``spec``. ``spec`` is validated as a
    structured mapping and is not sent to a language model.
    """
    if (text is None) == (spec is None):
        raise SpecError("provide exactly one of text or spec")
    if spec is not None:
        compiled = ExperimentSpec.from_mapping(spec, parse_source="structured")
    else:
        if not isinstance(text, str):
            raise SpecError("experiment spec must be text")
        compiled = compile_spec(text)
    return run_spec(compiled)


def summarize_paired_difference(
    candidate: Sequence[float],
    reference: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, object]:
    """Return a paired mean difference for two aligned per-seed samples."""
    return paired_mean_difference(candidate, reference, n_resamples=n_resamples, seed=seed)
