"""Natural-language experiment specs with a deterministic parser.

The parser in this module does not open a network connection. A caller may
supply a :class:`LanguageModel` to translate free text. If that model is
absent, raises, or returns a spec that does not validate, compilation uses
the deterministic parser instead. The compiled spec is what the run executes,
so the same canonical spec and seed produce the same training call.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from . import __version__
from .cli import (
    DEFAULT_COMPARISON_POLICIES,
    DEFAULT_GAMMA,
    DEFAULT_REWARD_NOISE,
    DEFAULT_STATES,
    FULL_LOG_EVERY,
    FULL_SEEDS,
    FULL_STEPS,
    QUICK_LOG_EVERY,
    QUICK_SEEDS,
    QUICK_STEPS,
    available_policies,
    run_comparison,
    run_paired_comparison,
)

_POLICY_ALIASES = {
    "baseline": "baseline",
    "no modulation": "baseline",
    "no-modulation": "baseline",
    "q-learning": "baseline",
    "q learning": "baseline",
    "immediate": "immediate",
    "online": "immediate",
    "fixed-lag": "fixed-lag",
    "fixed lag": "fixed-lag",
    "lagged": "fixed-lag",
    "delayed": "fixed-lag",
    "leaky-state": "leaky-state",
    "leaky state": "leaky-state",
    "homeostatic": "leaky-state",
    "homeostasis": "leaky-state",
    "leaky": "leaky-state",
    "clipped": "clipped",
    "all policies": "all",
    "all": "all",
}
_POLICY_PATTERN = "|".join(
    re.escape(alias) for alias in sorted(_POLICY_ALIASES, key=len, reverse=True)
)
_POLICY_RE = re.compile(rf"\b(?:{_POLICY_PATTERN})\b", re.IGNORECASE)
_COMPARE_RE = re.compile(
    r"\bcompare\s+(?P<left>.+?)\s+(?:to|with|against|versus|vs\.?)\s+(?P<right>.+?)"
    r"(?=[.;\n]|$)",
    re.IGNORECASE,
)
_VERSUS_RE = re.compile(
    r"\b(?P<left>[^.\n;]{1,80}?)\s+(?:versus|vs\.?)\s+(?P<right>[^.\n;]+)",
    re.IGNORECASE,
)
_FIELDS = {
    "policy": "policies",
    "policies": "policies",
    "rule": "policies",
    "steps": "steps",
    "n_steps": "steps",
    "training_steps": "steps",
    "seeds": "seeds",
    "n_seeds": "seeds",
    "replicas": "seeds",
    "log_every": "log_every",
    "seed": "seed",
    "root_seed": "seed",
    "states": "states",
    "n_states": "states",
    "chain_length": "states",
    "gamma": "gamma",
    "discount": "gamma",
    "reward_noise": "reward_noise",
    "sigma": "reward_noise",
    "sigma_r": "reward_noise",
    "compare_to": "compare_to",
    "reference": "compare_to",
    "bootstrap_resamples": "bootstrap_resamples",
    "resamples": "bootstrap_resamples",
    "bootstrap_seed": "bootstrap_seed",
    "full": "full",
}


def _assign_key(key: str) -> str:
    return r"[\s_]+".join(re.escape(part) for part in key.split("_"))


_ASSIGN_KEYS = sorted(_FIELDS, key=len, reverse=True)
_ASSIGN_RE = re.compile(
    r"\b(?P<key>" + "|".join(_assign_key(key) for key in _ASSIGN_KEYS) + r")\s*[:=]\s*"
    r"(?P<value>\[[^\]]*\]|\"[^\"]+\"|'[^']+'|"
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|"
    r"[A-Za-z][A-Za-z0-9_-]*)",
    re.IGNORECASE,
)
_PARSE_SOURCES = frozenset({"deterministic", "deterministic_fallback", "llm", "structured"})


class SpecError(ValueError):
    """The experiment description could not be compiled into a spec."""


class LanguageModel(Protocol):
    """Optional translator from free text to a JSON object.

    Implementations may call a hosted model. This package does not ship one
    and does not call one on the user's behalf.
    """

    def complete(self, prompt: str) -> str:
        """Return text containing one JSON object."""
        ...


@dataclass(frozen=True)
class ExperimentSpec:
    """Resolved, reproducible training request.

    ``spec_id`` hashes the canonical parameters only. ``parse_source`` records
    whether a model, the deterministic parser, or a direct mapping produced it.
    """

    policies: tuple[str, ...]
    steps: int
    seeds: int
    log_every: int
    seed: int
    states: int
    gamma: float
    reward_noise: float
    compare_to: str | None
    bootstrap_resamples: int
    bootstrap_seed: int
    parse_source: str
    spec_id: str

    def canonical(self) -> dict[str, object]:
        """Return the parameters that define the run."""
        return {
            "policies": list(self.policies),
            "steps": self.steps,
            "seeds": self.seeds,
            "log_every": self.log_every,
            "seed": self.seed,
            "states": self.states,
            "gamma": self.gamma,
            "reward_noise": self.reward_noise,
            "compare_to": self.compare_to,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the spec, its id, and how it was parsed."""
        return {
            "schema_version": 3,
            "spec_id": self.spec_id,
            "parse_source": self.parse_source,
            "spec": self.canonical(),
        }

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, object],
        *,
        parse_source: str,
    ) -> ExperimentSpec:
        """Validate an explicit mapping and fill omitted fields from defaults."""
        if parse_source not in _PARSE_SOURCES:
            raise ValueError(f"unknown parse source: {parse_source}")
        if not isinstance(mapping, Mapping):
            raise SpecError("experiment spec must be an object")
        fields = _canonicalize_fields(mapping)
        if not fields:
            raise SpecError("experiment spec has no fields")

        full = _as_bool(fields.pop("full"), "full") if "full" in fields else False
        policies = _resolve_policy_field(fields)
        steps = _as_int(fields.pop("steps"), "steps", 1) if "steps" in fields else None
        seeds = _as_int(fields.pop("seeds"), "seeds", 1) if "seeds" in fields else None
        log_every = (
            _as_int(fields.pop("log_every"), "log_every", 1) if "log_every" in fields else None
        )
        if full:
            steps = FULL_STEPS if steps is None else steps
            seeds = FULL_SEEDS if seeds is None else seeds
            log_every = FULL_LOG_EVERY if log_every is None else log_every
        steps = QUICK_STEPS if steps is None else steps
        seeds = QUICK_SEEDS if seeds is None else seeds
        log_every = QUICK_LOG_EVERY if log_every is None else log_every
        seed = _as_int(fields.pop("seed"), "seed", 0) if "seed" in fields else 0
        states = (
            _as_int(fields.pop("states"), "states", 2) if "states" in fields else DEFAULT_STATES
        )
        gamma = _as_float(fields.pop("gamma"), "gamma") if "gamma" in fields else DEFAULT_GAMMA
        if not 0.0 <= gamma < 1.0:
            raise SpecError("gamma must satisfy 0 <= gamma < 1")
        reward_noise = (
            _as_float(fields.pop("reward_noise"), "reward_noise")
            if "reward_noise" in fields
            else DEFAULT_REWARD_NOISE
        )
        if reward_noise < 0.0:
            raise SpecError("reward_noise must be finite and at least 0")
        compare_to = _optional_policy(fields.pop("compare_to")) if "compare_to" in fields else None
        bootstrap_resamples = (
            _as_int(fields.pop("bootstrap_resamples"), "bootstrap_resamples", 1)
            if "bootstrap_resamples" in fields
            else 10_000
        )
        bootstrap_seed = (
            _as_int(fields.pop("bootstrap_seed"), "bootstrap_seed", 0)
            if "bootstrap_seed" in fields
            else 0
        )
        if fields:
            unknown = sorted(fields)[0]
            raise SpecError(f"unknown spec field: {unknown}")
        if compare_to is not None and set(policies) == {compare_to}:
            raise SpecError("select a candidate policy different from the reference policy")
        if compare_to is not None and seeds < 2:
            raise SpecError("paired comparison requires at least 2 seeds")

        canonical = {
            "policies": list(policies),
            "steps": steps,
            "seeds": seeds,
            "log_every": log_every,
            "seed": seed,
            "states": states,
            "gamma": gamma,
            "reward_noise": reward_noise,
            "compare_to": compare_to,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
        }
        spec_id = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                "utf-8"
            )
        ).hexdigest()
        return cls(
            policies=policies,
            steps=steps,
            seeds=seeds,
            log_every=log_every,
            seed=seed,
            states=states,
            gamma=gamma,
            reward_noise=reward_noise,
            compare_to=compare_to,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            parse_source=parse_source,
            spec_id=spec_id,
        )


def compile_spec(text: str, *, model: LanguageModel | None = None) -> ExperimentSpec:
    """Compile ``text`` into a validated experiment spec.

    With no model, only the deterministic parser runs. A model is tried first.
    Any failure, including invalid JSON or unknown fields, falls back to the
    deterministic parser and records ``deterministic_fallback``.
    """
    if not isinstance(text, str):
        raise SpecError("experiment spec must be text")
    if not text.strip():
        raise SpecError("experiment spec is empty")
    if model is None:
        return _compile_deterministic(text, "deterministic")

    model_error: Exception | None = None
    try:
        raw = model.complete(_model_prompt(text))
        if not isinstance(raw, str):
            raise SpecError("language model output must be text")
        return ExperimentSpec.from_mapping(_extract_json_object(raw), parse_source="llm")
    except Exception as exc:
        model_error = exc
    try:
        return _compile_deterministic(text, "deterministic_fallback")
    except SpecError as exc:
        detail = str(model_error).strip() or type(model_error).__name__
        raise SpecError(f"{exc} (language model output was not used: {detail[:200]})") from exc


def run_spec(spec: ExperimentSpec) -> dict[str, object]:
    """Train ``spec`` and return a schema-3 result document.

    Paired runs use candidate-minus-reference differences. Absolute rows are
    always included. No language model is called here.
    """
    if not isinstance(spec, ExperimentSpec):
        raise TypeError("spec must be an ExperimentSpec")
    common = {
        "n_steps": spec.steps,
        "n_seeds": spec.seeds,
        "log_every": spec.log_every,
        "seed": spec.seed,
        "policies": list(spec.policies),
        "n_states": spec.states,
        "gamma": spec.gamma,
        "sigma_r": spec.reward_noise,
    }
    paired: object = None
    comparison: object = None
    if spec.compare_to is None:
        results = run_comparison(**common)
    else:
        report = run_paired_comparison(
            **common,
            reference_policy=spec.compare_to,
            bootstrap_resamples=spec.bootstrap_resamples,
            bootstrap_seed=spec.bootstrap_seed,
        )
        results = report["results"]
        paired = report["paired_comparisons"]
        comparison = report["comparison"]
    return {
        "schema_version": 3,
        "spec_id": spec.spec_id,
        "parse_source": spec.parse_source,
        "spec": spec.canonical(),
        "software": {"hmea": __version__, "numpy": np.__version__},
        "results": results,
        "paired_comparisons": paired,
        "comparison": comparison,
    }


def _model_prompt(text: str) -> str:
    return (
        "Convert the experiment description into one JSON object. "
        "Allowed keys: policies, steps, seeds, log_every, seed, states, gamma, "
        "reward_noise, compare_to, bootstrap_resamples, bootstrap_seed, full. "
        "policies is an array of baseline, immediate, fixed-lag, leaky-state, "
        "clipped, or all. compare_to is one policy id or null. full is boolean. "
        "Omit keys the description does not specify. Return JSON only.\n\n"
        f"{text}"
    )


def _compile_deterministic(text: str, parse_source: str) -> ExperimentSpec:
    stripped = _strip_comments(text)
    if not stripped.strip():
        raise SpecError("experiment spec is empty")
    payload = _json_object(stripped)
    if payload is None:
        payload = _parse_text(stripped)
        if not payload:
            raise SpecError("could not recognize an experiment spec")
    return ExperimentSpec.from_mapping(payload, parse_source=parse_source)


def _json_object(text: str) -> dict[str, object] | None:
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON spec: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise SpecError("JSON spec must be an object")
    return value


def _extract_json_object(text: str) -> dict[str, object]:
    start = text.find("{")
    if start < 0:
        raise SpecError("language model output did not contain a JSON object")
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise SpecError("language model output was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise SpecError("language model output must be a JSON object")
    return payload


def _strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        marker = line.find("#")
        lines.append(line if marker < 0 else line[:marker])
    return "\n".join(lines)


def _parse_text(text: str) -> dict[str, object]:
    normalized = re.sub(r"\s+", " ", text).strip()
    prose = _parse_prose(normalized)
    assignments = _parse_assignments(normalized)
    return {**prose, **assignments}


def _parse_assignments(text: str) -> dict[str, object]:
    found: dict[str, object] = {}
    for match in _ASSIGN_RE.finditer(text):
        key = re.sub(r"\s+", "_", match.group("key").lower())
        field = _FIELDS[key]
        value: object = _decode_assignment(field, match.group("value"))
        if field in found:
            raise SpecError(f"repeated spec field: {field}")
        found[field] = value
    return found


def _decode_assignment(field: str, raw: str) -> object:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    if field == "policies":
        if value.startswith("["):
            inner = value[1:-1].strip()
            parts = [part.strip().strip("'\"") for part in inner.split(",") if part.strip()]
            if not parts:
                raise SpecError("policies must not be empty")
            return [_policy_token(part) for part in parts]
        return [_policy_token(value)]
    if field == "compare_to":
        if value.lower() in {"none", "null"}:
            return None
        return _policy_token(value)
    if field == "full":
        return _as_bool(value, "full")
    if field in {"gamma", "reward_noise"}:
        return value
    return value


def _parse_prose(text: str) -> dict[str, object]:
    found: dict[str, object] = {}
    comparisons = list(_COMPARE_RE.finditer(text))
    if len(comparisons) > 1:
        raise SpecError("experiment spec contains more than one comparison")
    comparison = comparisons[0] if comparisons else _VERSUS_RE.search(text)
    mentioned = _policy_ids(text)
    if comparison is not None:
        left = _policy_ids(comparison.group("left"))
        right = _policy_ids(comparison.group("right"))
        if not left:
            raise SpecError("comparison requires a candidate policy")
        if len(right) != 1:
            raise SpecError("comparison requires one reference policy")
        if "all" in left and len(left) != 1:
            raise SpecError("all cannot be combined with other policies")
        if len(left) != len(set(left)):
            raise SpecError("policies must not contain duplicates")
        outside = [name for name in mentioned if name not in set(left) | set(right)]
        if outside:
            raise SpecError("policy names outside the comparison are ambiguous")
        found["policies"] = left
        found["compare_to"] = right[0]
    elif mentioned:
        if "all" in mentioned and len(mentioned) != 1:
            raise SpecError("all cannot be combined with other policies")
        if len(mentioned) != len(set(mentioned)):
            raise SpecError("policies must not contain duplicates")
        found["policies"] = mentioned

    numbered = (
        ("steps", r"\b(\d+)\s+steps\b", True),
        ("seeds", r"\b(\d+)\s+seeds?\b", True),
        ("log_every", r"\blog(?:ging)? every\s+(\d+)\b", True),
        ("states", r"\b(\d+)\s+states\b", True),
        ("gamma", r"\b(?:gamma|discount(?: factor)?)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))\b", False),
        (
            "reward_noise",
            r"\b(?:reward noise|sigma(?:_r)?)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+))\b",
            False,
        ),
        ("bootstrap_resamples", r"\b(\d+)\s+bootstrap resamples\b", True),
        ("bootstrap_seed", r"\bbootstrap seed\s+(\d+)\b", True),
        ("seed", r"\b(?<!bootstrap )(?:root )?seed\s+(\d+)\b", True),
    )
    for key, pattern, integer in numbered:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if not matches:
            continue
        values = {item.group(1) for item in matches}
        if len(values) > 1:
            raise SpecError(f"repeated spec field: {key}")
        token = matches[0].group(1)
        found[key] = int(token) if integer else float(token)
    if re.search(
        r"\bfull reference\b|\breference configuration\b|\bfull configuration\b",
        text,
        re.IGNORECASE,
    ):
        found["full"] = True
    return found


def _policy_ids(text: str) -> list[str]:
    return [_POLICY_ALIASES[match.group(0).lower()] for match in _POLICY_RE.finditer(text)]


def _policy_token(value: object) -> str:
    if not isinstance(value, str):
        raise SpecError(f"unknown policy: {value}")
    token = re.sub(r"\s+", " ", value.strip().lower())
    try:
        return _POLICY_ALIASES[token]
    except KeyError as exc:
        raise SpecError(f"unknown policy: {value.strip()}") from exc


def _canonicalize_fields(mapping: Mapping[str, object]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise SpecError("spec fields must be strings")
        normalized = re.sub(r"[\s-]+", "_", key.strip().lower())
        try:
            field = _FIELDS[normalized]
        except KeyError as exc:
            raise SpecError(f"unknown spec field: {key}") from exc
        if field in fields:
            raise SpecError(f"repeated spec field: {field}")
        fields[field] = value
    return fields


def _resolve_policy_field(fields: dict[str, object]) -> tuple[str, ...]:
    if "policies" not in fields:
        return DEFAULT_COMPARISON_POLICIES
    raw = fields.pop("policies")
    if isinstance(raw, str):
        items: Sequence[object] = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        items = raw
    else:
        raise SpecError("policies must be a policy name or a list of policy names")
    if not items:
        raise SpecError("policies must not be empty")
    names = tuple(_policy_token(item) for item in items)
    if "all" in names:
        if names != ("all",):
            raise SpecError("all cannot be combined with other policies")
        return DEFAULT_COMPARISON_POLICIES
    known = set(available_policies())
    for name in names:
        if name not in known:
            raise SpecError(f"unknown policy: {name}")
    if len(names) != len(set(names)):
        raise SpecError("policies must not contain duplicates")
    return names


def _optional_policy(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    name = _policy_token(value)
    if name == "all":
        raise SpecError("compare_to must name one policy")
    if name not in available_policies():
        raise SpecError(f"unknown policy: {name}")
    return name


def _as_bool(value: object, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "yes", "1"}:
            return True
        if token in {"false", "no", "0"}:
            return False
    raise SpecError(f"{name} must be true or false")


def _as_int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise SpecError(f"{name} must be an integer >= {minimum}")
    number: int | None = None
    if isinstance(value, str):
        token = value.strip()
        if re.fullmatch(r"[+-]?\d+", token):
            number = int(token)
        else:
            try:
                numeric = float(token)
            except ValueError as exc:
                raise SpecError(f"{name} must be an integer >= {minimum}") from exc
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise SpecError(f"{name} must be an integer >= {minimum}")
            number = int(numeric)
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise SpecError(f"{name} must be an integer >= {minimum}")
        number = int(value)
    else:
        try:
            number = operator.index(value)
        except TypeError as exc:
            raise SpecError(f"{name} must be an integer >= {minimum}") from exc
    if number < minimum:
        raise SpecError(f"{name} must be an integer >= {minimum}")
    return number


def _as_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise SpecError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise SpecError(f"{name} must be a finite number")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Compile a spec from the command line and optionally run it."""
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="hmea-spec",
        description=(
            "Compile a natural-language HMEA experiment spec. "
            "The command uses the deterministic parser and does not call a language model."
        ),
    )
    parser.add_argument("text", nargs="?", help="experiment description")
    parser.add_argument("-f", "--file", type=Path, help="read the description from a file")
    parser.add_argument("--run", action="store_true", help="train the compiled spec")
    parser.add_argument("-o", "--output", type=Path, help="write JSON to this file")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    if (args.text is None) == (args.file is None):
        parser.error("provide exactly one of a spec text or --file")
    try:
        text = args.file.read_text(encoding="utf-8") if args.file is not None else args.text
        spec = compile_spec(text)
        payload = run_spec(spec) if args.run else spec.to_dict()
        rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    except SpecError as exc:
        parser.error(str(exc))
    except OSError as exc:
        parser.error(f"cannot read spec file: {exc}")
    if args.output is None:
        print(rendered, end="")
        return 0
    try:
        args.output.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        parser.error(f"cannot write output file: {exc}")
    print(f"Wrote spec JSON to {args.output}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
