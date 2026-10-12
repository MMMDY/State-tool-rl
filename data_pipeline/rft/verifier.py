"""Rule-based schema, process, replay, and terminal verification."""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from data_pipeline.rft.io import sha256_json, utc_now
from data_pipeline.rft.normalize import fingerprints, trajectory_length

HARD_FLAGS = {
    "invalid_schema",
    "wrong_tool",
    "bad_args",
    "invalid_precondition",
    "ignored_observation",
    "repeated_invalid_action",
    "no_progress",
    "false_complete",
    "terminal_fail",
    "replay_mismatch",
    "observation_mismatch",
    "overlong",
    "cross_split",
    "missing_trajectory",
}


def _schema_error(
    value: Any, schema: dict[str, Any], *, root: dict[str, Any], path: str
) -> str | None:
    if "$ref" in schema:
        reference = schema["$ref"]
        if not isinstance(reference, str) or not reference.startswith("#/"):
            return f"{path}: unsupported schema reference {reference!r}"
        target: Any = root
        for part in reference[2:].split("/"):
            target = target.get(part) if isinstance(target, dict) else None
        if not isinstance(target, dict):
            return f"{path}: unresolved schema reference {reference!r}"
        return _schema_error(value, target, root=root, path=path)
    alternatives = schema.get("anyOf") or schema.get("oneOf")
    if isinstance(alternatives, list):
        errors = [
            _schema_error(value, item, root=root, path=path)
            for item in alternatives
            if isinstance(item, dict)
        ]
        if any(error is None for error in errors):
            return None
        return f"{path}: value matches no allowed schema ({'; '.join(error for error in errors if error)})"
    if "enum" in schema and value not in schema["enum"]:
        return f"{path}: {value!r} is not in enum {schema['enum']!r}"
    if "const" in schema and value != schema["const"]:
        return f"{path}: {value!r} does not equal const {schema['const']!r}"

    expected = schema.get("type")
    valid_type = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, True)
    if not valid_type:
        return f"{path}: expected {expected}, got {type(value).__name__}"
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        missing = [name for name in required if name not in value]
        if missing:
            return f"{path}: missing required fields {missing}"
        extra = sorted(set(value).difference(properties))
        # Tau2's Pydantic schemas do not consistently emit additionalProperties=false;
        # RFT's hard gate intentionally forbids undeclared arguments regardless.
        if extra:
            return f"{path}: undeclared fields {extra}"
        for name, item in value.items():
            child = properties.get(name)
            if isinstance(child, dict):
                error = _schema_error(item, child, root=root, path=f"{path}.{name}")
                if error:
                    return error
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            error = _schema_error(
                item, schema["items"], root=root, path=f"{path}[{index}]"
            )
            if error:
                return error
    return None


def schema_verifier(record: dict[str, Any]) -> dict[str, Any]:
    flags: list[str] = []
    errors: list[str] = []
    tool_schemas: dict[str, dict[str, Any]] = {}
    raw_schemas = record.get("tools", {}).get("schema") or []
    if record.get("tools", {}).get("schema_hash") != sha256_json(raw_schemas):
        flags.append("invalid_schema")
        errors.append("stored tool schema hash does not match the saved schema")
    if record.get("system_prompt_hash") != sha256_json(record.get("system_prompt", "")):
        flags.append("invalid_schema")
        errors.append("stored system prompt hash does not match the saved prompt")

    for item in raw_schemas:
        function = item.get("function") if isinstance(item, dict) else None
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            tool_schemas[function["name"]] = function.get("parameters") or {
                "type": "object"
            }
    if not tool_schemas:
        flags.append("invalid_schema")
        errors.append("trajectory contains no tool schema")
    for step in record.get("steps") or []:
        call = step.get("tool_call")
        if call is None:
            if "invalid_schema" in (step.get("flags") or []):
                flags.append("invalid_schema")
                errors.append(
                    f"step {step.get('step_idx')}: model action was not parseable"
                )
            continue
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            flags.append("invalid_schema")
            errors.append(f"step {step.get('step_idx')}: malformed tool_call record")
            continue
        name = call["name"]
        if name not in tool_schemas:
            flags.append("wrong_tool")
            errors.append(f"step {step.get('step_idx')}: undeclared tool {name!r}")
            continue
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            flags.append("bad_args")
            errors.append(f"step {step.get('step_idx')}: arguments must be an object")
            continue
        error = _schema_error(
            arguments, tool_schemas[name], root=tool_schemas[name], path=name
        )
        if error:
            flags.append("bad_args")
            errors.append(f"step {step.get('step_idx')}: {error}")
    if any(
        "overlong" in (step.get("flags") or []) for step in record.get("steps") or []
    ):
        flags.append("overlong")
    return {"passed": not flags, "flags": sorted(set(flags)), "errors": errors}


def _task_initial_parts(task: Any) -> tuple[Any, Any, list[Any]]:
    initial = task.initial_state
    if initial is None:
        return None, None, []
    return (
        initial.initialization_data,
        initial.initialization_actions,
        list(initial.message_history or []),
    )


def _environment_hash(environment: Any) -> str:
    return sha256_json([environment.get_db_hash(), environment.get_user_db_hash()])


def _content_value(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def replay_messages(
    *,
    domain: str,
    task: Any,
    simulation: Any,
    expected_step_count: int | None = None,
) -> dict[str, Any]:
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        ToolMessage,
        UserMessage,
    )
    from tau2.registry import registry

    initialization_data, initialization_actions, initial_history = _task_initial_parts(
        task
    )
    environment = registry.get_env_constructor(domain)(solo_mode=False)
    environment.set_state(initialization_data, initialization_actions, [], strict=True)
    messages = list(simulation.messages or [])
    initial_assistant_count = sum(
        isinstance(message, AssistantMessage) for message in initial_history
    )
    if expected_step_count is not None:
        total_assistant_count = sum(
            isinstance(message, AssistantMessage) for message in messages
        )
        initial_assistant_count = max(0, total_assistant_count - expected_step_count)
    assistant_seen = 0
    active_new_step = False
    state_before_first: str | None = None
    state_after_steps: list[str] = []
    observation_errors: list[str] = []
    assistant_tool_errors: list[int] = []

    index = 0
    while index < len(messages):
        message = messages[index]
        if isinstance(message, AssistantMessage):
            assistant_seen += 1
            is_new = assistant_seen > initial_assistant_count
            if is_new:
                if active_new_step:
                    state_after_steps.append(_environment_hash(environment))
                else:
                    state_before_first = _environment_hash(environment)
                active_new_step = True
        calls = (
            getattr(message, "tool_calls", None)
            if isinstance(message, (AssistantMessage, UserMessage))
            else None
        )
        if calls:
            following = messages[index + 1] if index + 1 < len(messages) else None
            multi_results = (
                following.tool_messages
                if isinstance(following, MultiToolMessage)
                else None
            )
            for call_offset, call in enumerate(calls, 1):
                if multi_results is not None:
                    expected = (
                        multi_results[call_offset - 1]
                        if call_offset <= len(multi_results)
                        else None
                    )
                else:
                    result_index = index + call_offset
                    expected = (
                        messages[result_index] if result_index < len(messages) else None
                    )
                if not isinstance(expected, ToolMessage):
                    observation_errors.append(
                        f"tool call {call.name!r} has no matching ToolMessage"
                    )
                    continue
                actual = environment.get_response(call)
                if call.id != expected.id or _content_value(
                    actual.content
                ) != _content_value(expected.content):
                    observation_errors.append(
                        f"tool call {call.name!r} replayed observation differs from recorded observation"
                    )
                if isinstance(message, AssistantMessage) and expected.error:
                    assistant_tool_errors.append(
                        max(0, assistant_seen - initial_assistant_count - 1)
                    )
        index += 1
    if active_new_step:
        state_after_steps.append(_environment_hash(environment))
    return {
        "initial_state_hash": state_before_first or _environment_hash(environment),
        "state_hash_after_steps": state_after_steps,
        "final_state_hash": _environment_hash(environment),
        "observation_errors": observation_errors,
        "assistant_tool_error_steps": assistant_tool_errors,
    }


def _load_task(record: dict[str, Any]) -> Any:
    from tau2.registry import registry

    if record.get("split") != "train":
        raise ValueError("cross_split")
    domain = str(record["domain"])
    task_id = str(record["task_id"])
    tasks = registry.get_tasks_loader(domain)("train")
    for task in tasks:
        if str(task.id) == task_id:
            return task
    raise ValueError(f"task {domain}/{task_id} does not exist in train split")


def _load_simulation(record: dict[str, Any]) -> Any:
    from tau2.data_model.simulation import SimulationRun

    payload = record.get("simulation")
    if not isinstance(payload, dict):
        raise TypeError("missing_trajectory")
    return SimulationRun.model_validate(payload)


def terminal_and_replay_verifier(
    record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.metrics.agent_metrics import is_successful
    from tau2.registry import registry

    terminal_flags: list[str] = []
    replay_flags: list[str] = []
    errors: list[str] = []
    replay: dict[str, Any] = {}
    score = 0.0
    try:
        task = _load_task(record)
    except ValueError as exc:
        flag = "cross_split" if str(exc) == "cross_split" else "terminal_fail"
        return (
            {"passed": False, "score": 0.0, "flags": [flag], "errors": [str(exc)]},
            {"passed": False, "flags": [flag], "errors": [str(exc)]},
        )
    try:
        simulation = _load_simulation(record)
    except (TypeError, ValueError) as exc:
        return (
            {
                "passed": False,
                "score": 0.0,
                "flags": ["terminal_fail"],
                "errors": [str(exc)],
            },
            {"passed": False, "flags": ["missing_trajectory"], "errors": [str(exc)]},
        )

    if str(simulation.task_id) != str(record["task_id"]):
        terminal_flags.append("terminal_fail")
        replay_flags.append("replay_mismatch")
        errors.append("simulation task_id differs from trajectory task_id")

    initial_payload = (
        task.initial_state.model_dump(mode="json")
        if task.initial_state is not None
        else None
    )
    if record.get("env", {}).get("initial_state_snapshot_hash") != sha256_json(
        initial_payload
    ):
        replay_flags.append("replay_mismatch")
        errors.append(
            "task initial-state snapshot hash differs from current train task"
        )

    from tau2.environment.tool import as_tool
    from tau2.gym.gym_agent import done

    current_environment = registry.get_env_constructor(str(record["domain"]))(
        solo_mode=False
    )
    current_tools = [tool.openai_schema for tool in current_environment.get_tools()]
    current_tools.append(as_tool(done).openai_schema)
    if record.get("tools", {}).get("schema_hash") != sha256_json(current_tools):
        replay_flags.append("replay_mismatch")
        errors.append("saved tool schema differs from the current domain tool schema")

    try:
        reward_info = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=str(record["domain"]),
            strict_replay=True,
        )
        score = float(reward_info.reward)
        if not is_successful(score):
            terminal_flags.append("terminal_fail")
        replay = replay_messages(
            domain=str(record["domain"]),
            task=task,
            simulation=simulation,
            expected_step_count=len(record.get("steps") or []),
        )
        if replay["observation_errors"]:
            replay_flags.append("observation_mismatch")
            errors.extend(replay["observation_errors"])
        recorded_initial = record.get("env", {}).get("initial_state_hash")
        if recorded_initial and replay["initial_state_hash"] != recorded_initial:
            replay_flags.append("replay_mismatch")
            errors.append("initial state hash differs during replay")
        recorded_states = [
            step.get("state_hash_after") for step in record.get("steps") or []
        ]
        replayed_states = replay["state_hash_after_steps"]
        if len(recorded_states) != len(replayed_states) or any(
            expected != actual
            for expected, actual in zip(recorded_states, replayed_states)
        ):
            replay_flags.append("replay_mismatch")
            errors.append("per-step state hashes differ during replay")
        recorded_final = record.get("terminal", {}).get("state_hash")
        if recorded_final and replay["final_state_hash"] != recorded_final:
            replay_flags.append("replay_mismatch")
            errors.append("terminal state hash differs during replay")
    except Exception as exc:  # noqa: BLE001 - replay errors are verifier output
        replay_flags.append("replay_mismatch")
        errors.append(f"strict replay failed: {type(exc).__name__}: {exc}")
    if not record.get("terminal", {}).get("done"):
        terminal_flags.append("terminal_fail")
    if not record.get("terminal", {}).get("success"):
        terminal_flags.append("terminal_fail")
    return (
        {
            "passed": not terminal_flags and not replay_flags,
            "score": score,
            "flags": sorted(set(terminal_flags + replay_flags)),
            "errors": errors,
        },
        {
            "passed": not replay_flags,
            "flags": sorted(set(replay_flags)),
            "errors": errors,
            "details": replay,
        },
    )


def process_verifier(
    record: dict[str, Any],
    replay_result: dict[str, Any],
    *,
    max_identical_no_progress_calls: int,
) -> dict[str, Any]:
    flags: list[str] = []
    errors: list[str] = []
    steps = record.get("steps") or []
    previous_after = record.get("env", {}).get("initial_state_hash")
    repeated: Counter[tuple[str, str]] = Counter()
    error_steps = set(
        replay_result.get("details", {}).get("assistant_tool_error_steps") or []
    )
    for index, step in enumerate(steps):
        before = step.get("state_hash_before")
        after = step.get("state_hash_after")
        if previous_after and before != previous_after:
            flags.append("replay_mismatch")
            errors.append(f"step {index}: state hash chain is broken")
        previous_after = after
        call = step.get("tool_call")
        if not isinstance(call, dict):
            continue
        signature = sha256_json([call.get("name"), call.get("arguments")])
        key = (str(before), signature)
        if index in error_steps:
            flags.append("invalid_precondition")
            errors.append(f"step {index}: environment rejected the tool call")
        if index in error_steps or before == after:
            repeated[key] += 1
            if repeated[key] > max_identical_no_progress_calls:
                if index in error_steps:
                    flags.append("repeated_invalid_action")
                    errors.append(f"step {index}: repeated an invalid tool call")
                else:
                    flags.extend(["ignored_observation", "no_progress"])
                    errors.append(
                        f"step {index}: repeated the same no-progress call in the same state"
                    )
    if record.get("terminal", {}).get("success") and steps:
        success_indices = [
            i
            for i, step in enumerate(steps)
            if float(step.get("step_reward") or 0) >= 1 - 1e-6
        ]
        if success_indices and success_indices[0] != len(steps) - 1:
            flags.append("no_progress")
            errors.append(
                "trajectory continued after an official successful terminal step"
            )
    if not record.get("terminal", {}).get("success"):
        final_text = str(steps[-1].get("assistant_raw_text") if steps else "").lower()
        if any(
            phrase in final_text
            for phrase in ("completed", "all set", "successfully", "done")
        ):
            flags.append("false_complete")
    flags.extend(replay_result.get("flags") or [])
    return {"passed": not flags, "flags": sorted(set(flags)), "errors": errors}


def verify_record(record: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    schema = schema_verifier(record)
    terminal, replay = terminal_and_replay_verifier(record)
    process = process_verifier(
        record,
        replay,
        max_identical_no_progress_calls=int(
            config["verification"].get("max_identical_no_progress_calls", 1)
        ),
    )
    all_flags = sorted(set(schema["flags"] + process["flags"] + terminal["flags"]))
    verified = dict(record)
    verified["verifier"] = {
        "version": config["verification"]["version"],
        "verified_at_utc": utc_now(),
        "schema": schema,
        "process": process,
        "terminal": terminal,
        "replay": replay,
        "flags": all_flags,
        "passed": not all_flags,
    }
    verified["fingerprints"] = fingerprints(verified)
    verified["quality"] = {
        "official_score": terminal["score"],
        "trajectory_chars": trajectory_length(verified),
        "tool_calls": sum(
            1 for step in verified.get("steps") or [] if step.get("tool_call")
        ),
        "verifier_passes": sum(
            (schema["passed"], process["passed"], terminal["passed"])
        ),
    }
    verified["status"] = "verified" if not all_flags else "rejected"
    return verified
