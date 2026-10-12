from __future__ import annotations

from copy import deepcopy

import pytest

from data_pipeline.rft.exporter import (
    deduplicate_and_cap,
    difficulty_from_rate,
    export_row,
    recovery_row,
)
from data_pipeline.rft.io import sha256_json, stable_int, validate_config
from data_pipeline.rft.normalize import fingerprints
from data_pipeline.rft.sampler import _model_action
from data_pipeline.rft.verifier import process_verifier, schema_verifier


def base_config() -> dict:
    return {
        "run": {},
        "model": {},
        "tasks": {"split": "train"},
        "sampling": {
            "initial_candidates": 16,
            "additional_candidates": 8,
            "max_candidates": 32,
            "success_quota": 4,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_response_tokens": 4096,
            "max_steps": 100,
            "base_seed": 7,
            "thinking": False,
        },
        "simulator": {},
        "prompt": {},
        "verification": {},
        "export": {
            "max_per_task": {"hard": 4, "medium": 4, "easy": 2},
        },
    }


def record(trajectory_id: str = "t1") -> dict:
    candidate = {
        "run_id": "run",
        "trajectory_id": trajectory_id,
        "task_id": "task",
        "domain": "retail",
        "split": "train",
        "tools": {
            "schema_hash": "",
            "schema": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "order_id": {"type": "string", "enum": ["o1"]}
                            },
                            "required": ["order_id"],
                        },
                    },
                }
            ],
        },
        "system_prompt": "policy",
        "system_prompt_hash": "",
        "env": {"initial_state_hash": "s0"},
        "terminal": {"success": True},
        "messages": [{"role": "assistant", "content": ""}],
        "steps": [
            {
                "step_idx": 0,
                "tool_call": {"name": "lookup", "arguments": {"order_id": "o1"}},
                "assistant_raw_text": "call",
                "state_hash_before": "s0",
                "state_hash_after": "s1",
                "flags": [],
                "step_reward": 1.0,
            }
        ],
        "quality": {
            "official_score": 1.0,
            "verifier_passes": 3,
            "trajectory_chars": 10,
        },
        "verifier": {"passed": True},
    }
    candidate["tools"]["schema_hash"] = sha256_json(candidate["tools"]["schema"])
    candidate["system_prompt_hash"] = sha256_json(candidate["system_prompt"])
    return candidate


def test_config_rejects_non_train_and_thinking() -> None:
    config = base_config()
    validate_config(config)
    config["tasks"]["split"] = "test"
    with pytest.raises(ValueError, match="train"):
        validate_config(config)
    config["tasks"]["split"] = "train"
    config["sampling"]["thinking"] = True
    with pytest.raises(ValueError, match="thinking=false"):
        validate_config(config)


def test_stable_seed_and_fingerprints_are_order_independent() -> None:
    first = record()
    second = deepcopy(first)
    second["steps"][0]["tool_call"]["arguments"] = {"order_id": "o1"}
    assert stable_int("retail/task") == stable_int("retail/task")
    assert fingerprints(first) == fingerprints(second)
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def test_schema_verifier_rejects_extra_missing_and_enum_arguments() -> None:
    assert schema_verifier(record())["passed"] is True
    extra = record()
    extra["steps"][0]["tool_call"]["arguments"]["fabricated"] = 1
    assert schema_verifier(extra)["flags"] == ["bad_args"]
    missing = record()
    missing["steps"][0]["tool_call"]["arguments"] = {}
    assert "bad_args" in schema_verifier(missing)["flags"]
    wrong_enum = record()
    wrong_enum["steps"][0]["tool_call"]["arguments"]["order_id"] = "o2"
    assert "bad_args" in schema_verifier(wrong_enum)["flags"]


def test_schema_verifier_rejects_tampered_saved_schema() -> None:
    candidate = record()
    candidate["tools"]["schema"][0]["function"]["name"] = "tampered"
    result = schema_verifier(candidate)
    assert "invalid_schema" in result["flags"]
    assert any("schema hash" in error for error in result["errors"])


def test_process_verifier_rejects_repeated_no_progress_call() -> None:
    candidate = record()
    repeated = deepcopy(candidate["steps"][0])
    candidate["steps"][0]["state_hash_after"] = "s0"
    repeated.update(
        {"step_idx": 1, "state_hash_before": "s0", "state_hash_after": "s0"}
    )
    candidate["steps"].append(repeated)
    result = process_verifier(
        candidate, {"flags": [], "details": {}}, max_identical_no_progress_calls=1
    )
    assert {"ignored_observation", "no_progress"}.issubset(result["flags"])


def test_model_action_accepts_one_declared_structured_call() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lookup",
                                "arguments": '{"order_id":"o1"}',
                            }
                        }
                    ]
                }
            }
        ]
    }
    action = _model_action(response, {"lookup"}, step_idx=2)
    assert action.name == "lookup"
    assert action.arguments == {"order_id": "o1"}
    with pytest.raises(ValueError, match="Undeclared"):
        _model_action(response, {"other"}, step_idx=2)


def test_difficulty_boundaries_and_easy_cap() -> None:
    assert difficulty_from_rate(0.19) == "hard"
    assert difficulty_from_rate(0.20) == "medium"
    assert difficulty_from_rate(0.80) == "medium"
    assert difficulty_from_rate(0.81) == "easy"
    rows = []
    for index in range(4):
        item = record(f"t{index}")
        item["steps"][0]["tool_call"]["arguments"]["order_id"] = f"o{index}"
        item["steps"][0]["state_hash_after"] = f"s{index + 1}"
        item["fingerprints"] = fingerprints(item)
        rows.append(item)
    stats = {("retail", "task"): {"difficulty": "easy"}}
    selected, counts = deduplicate_and_cap(rows, stats, base_config())
    assert len(selected) == 2
    assert counts["per_task_cap_removed"] == 2


def test_export_contains_existing_sft_prompt_and_provenance() -> None:
    item = record()
    item["difficulty"] = "medium"
    item["fingerprints"] = fingerprints(item)
    exported = export_row(item)
    assert exported["prompt"] == exported["messages"]
    assert exported["quality"]["loss_mask_roles"] == ["assistant"]
    assert exported["source_run_id"] == "run"


def test_tau2_replay_regenerates_recorded_tool_observations() -> None:
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.registry import registry

    domain = "retail"
    task = registry.get_tasks_loader(domain)("train")[0]
    initial = task.initial_state
    initial_data = initial.initialization_data if initial else None
    initial_actions = initial.initialization_actions if initial else None
    initial_messages = list(initial.message_history or []) if initial else []
    environment = registry.get_env_constructor(domain)(solo_mode=False)
    environment.set_state(initial_data, initial_actions, initial_messages, strict=True)
    messages = list(initial_messages)
    messages.append(AssistantMessage(role="assistant", content="bootstrap greeting"))
    expected_assistant_steps = 0
    for index, expected in enumerate(task.evaluation_criteria.actions or []):
        call = ToolCall(
            id=f"gold_{index}",
            name=expected.name,
            arguments=expected.arguments,
            requestor=expected.requestor,
        )
        message_type = (
            AssistantMessage if expected.requestor == "assistant" else UserMessage
        )
        if expected.requestor == "assistant":
            expected_assistant_steps += 1
        messages.append(
            message_type(role=expected.requestor, content=None, tool_calls=[call])
        )
        messages.append(environment.get_response(call))
    simulation = SimulationRun(
        id="offline-replay-test",
        task_id=str(task.id),
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        duration=1.0,
        termination_reason=TerminationReason.AGENT_STOP,
        messages=messages,
    )
    from data_pipeline.rft.verifier import replay_messages

    replay = replay_messages(
        domain=domain,
        task=task,
        simulation=simulation,
        expected_step_count=expected_assistant_steps,
    )
    assert replay["observation_errors"] == []
    assert len(replay["state_hash_after_steps"]) == expected_assistant_steps
    assert replay["final_state_hash"] == sha256_json(
        [environment.get_db_hash(), environment.get_user_db_hash()]
    )


def test_single_environment_rejection_is_a_hard_process_failure() -> None:
    candidate = record()
    replay = {"flags": [], "details": {"assistant_tool_error_steps": [0]}}
    result = process_verifier(candidate, replay, max_identical_no_progress_calls=1)
    assert "invalid_precondition" in result["flags"]
    assert result["passed"] is False


def test_recovery_context_stops_before_future_assistant_turns() -> None:
    candidate = record()
    candidate["terminal"] = {"success": False}
    candidate["steps"][0]["flags"] = ["invalid_schema"]
    candidate["messages"] = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "request"},
        {"role": "assistant", "content": "bad call"},
        {"role": "tool", "content": "Error: invalid"},
        {"role": "user", "content": "try again"},
        {"role": "assistant", "content": "future turn"},
    ]
    recovery = recovery_row(candidate, "hard")
    assert recovery["training_eligible"] is False
    assert [message["content"] for message in recovery["recovery_context"]] == [
        "policy",
        "request",
        "bad call",
        "Error: invalid",
        "try again",
    ]


def test_message_normalization_preserves_deterministic_tool_pairing() -> None:
    from data_pipeline.rft.normalize import normalized_messages

    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "random-trace-id",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"z":2,"a":1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "random-trace-id", "content": "ok"},
    ]
    normalized = normalized_messages(messages)
    assert normalized[0]["tool_calls"][0]["id"] == "call_0"
    assert normalized[1]["tool_call_id"] == "call_0"
    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == '{"a":1,"z":2}'


def test_provisional_qualification_requires_schema_and_process_progress() -> None:
    from data_pipeline.rft.sampler import _provisionally_qualified

    candidate = record()
    candidate["status"] = "raw"
    assert _provisionally_qualified(candidate) is True
    candidate["steps"][0]["state_hash_after"] = "s0"
    repeated = deepcopy(candidate["steps"][0])
    repeated["step_idx"] = 1
    candidate["steps"].append(repeated)
    assert _provisionally_qualified(candidate) is False
