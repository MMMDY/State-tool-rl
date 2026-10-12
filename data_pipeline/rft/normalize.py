"""Trajectory normalization and stable fingerprints."""

from __future__ import annotations

import json
import re
from typing import Any

from data_pipeline.rft.io import sha256_json

_WHITESPACE = re.compile(r"\s+")
_TRACE_FIELDS = {"id", "tool_call_id", "trace_id", "request_id", "timestamp"}


def normalize_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalize_value(item)
            for key, item in sorted(value.items())
            if key not in _TRACE_FIELDS
        }
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, str):
        return normalize_text(value)
    return value


def normalized_actions(record: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for step in record.get("steps") or []:
        call = step.get("tool_call")
        if not isinstance(call, dict):
            continue
        actions.append(
            {
                "name": call.get("name"),
                "arguments": normalize_value(call.get("arguments") or {}),
            }
        )
    return actions


def action_fingerprint(record: dict[str, Any]) -> str:
    return sha256_json(normalized_actions(record))


def state_fingerprint(record: dict[str, Any]) -> str:
    sequence = [record.get("env", {}).get("initial_state_hash")]
    sequence.extend(step.get("state_hash_after") for step in record.get("steps") or [])
    return sha256_json(sequence)


def semantic_fingerprint(record: dict[str, Any]) -> str:
    intent: list[Any] = []
    for step in record.get("steps") or []:
        call = step.get("tool_call")
        if isinstance(call, dict):
            intent.append(
                {
                    "tool": call.get("name"),
                    "argument_keys": sorted((call.get("arguments") or {}).keys()),
                }
            )
        else:
            raw = normalize_text(str(step.get("assistant_raw_text") or "")).lower()
            intent.append(
                {"response_terms": sorted(set(re.findall(r"[a-z0-9_]+", raw)))[:64]}
            )
    return sha256_json(intent)


def fingerprints(record: dict[str, Any]) -> dict[str, str]:
    return {
        "action_fingerprint": action_fingerprint(record),
        "state_fingerprint": state_fingerprint(record),
        "semantic_fingerprint": semantic_fingerprint(record),
    }


def normalized_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    call_id_map: dict[str, str] = {}
    last_call_id: str | None = None
    call_index = 0

    for message in messages:
        normalized = normalize_value(message)
        if normalized.get("content") is None:
            normalized["content"] = ""
        raw_calls = message.get("tool_calls") or []
        normalized_calls = normalized.get("tool_calls") or []
        for raw_call, call in zip(raw_calls, normalized_calls):
            deterministic_id = f"call_{call_index}"
            call_index += 1
            original_id = str(raw_call.get("id") or "")
            if original_id:
                call_id_map[original_id] = deterministic_id
            call["id"] = deterministic_id
            function = call.get("function")
            if isinstance(function, dict):
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments) if arguments.strip() else {}
                    except json.JSONDecodeError:
                        pass
                if isinstance(arguments, dict):
                    function["arguments"] = json.dumps(
                        normalize_value(arguments),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
            last_call_id = deterministic_id
        if normalized.get("role") == "tool":
            original_id = str(message.get("tool_call_id") or "")
            normalized["tool_call_id"] = call_id_map.get(
                original_id, last_call_id or "call_unknown"
            )
        result.append(normalized)
    return result


def trajectory_length(record: dict[str, Any]) -> int:
    return sum(
        len(json.dumps(message, ensure_ascii=False))
        for message in record.get("messages") or []
    )
