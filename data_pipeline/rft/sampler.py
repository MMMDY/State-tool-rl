"""Adaptive Qwen3 rollout sampling for the RFT pipeline."""

from __future__ import annotations

import asyncio
import json
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from data_pipeline.rft.io import (
    append_jsonl,
    atomic_write_json,
    canonical_json,
    git_commit,
    iter_jsonl,
    redact_sensitive,
    sha256_json,
    stable_int,
    utc_now,
    write_yaml_once,
)
from data_pipeline.rft.normalize import action_fingerprint
from tau2_rl_pipeline.actions import (
    env_action_from_parsed_action,
    parse_action,
    split_observation,
)

_NATIVE_CALL_RE = re.compile(
    r"^\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*$", re.DOTALL
)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    domain: str
    split: str
    task_index: int
    task_id: str
    task: Any

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.task_id}"


@dataclass(frozen=True, slots=True)
class ModelAction:
    raw_text: str
    env_text: str
    name: str | None
    arguments: dict[str, Any] | None
    history_message: dict[str, Any]
    tool_call_id: str | None


class SGLangClient:
    def __init__(self, api_base: str, model: str) -> None:
        self.endpoint = f"{api_base.rstrip('/')}/v1/chat/completions"
        self.model = model
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        sampling: dict[str, Any],
    ) -> dict[str, Any]:
        response = await self._client.post(
            self.endpoint,
            json={
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                **sampling,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError("Chat Completions response must be an object")
        return payload


def _model_action(
    response: dict[str, Any], declared_tools: set[str], *, step_idx: int
) -> ModelAction:
    choices = response.get("choices") or []
    if len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Expected exactly one completion choice")
    message = choices[0].get("message") or {}
    if not isinstance(message, dict):
        raise TypeError("Completion message must be an object")
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        if len(tool_calls) != 1:
            raise ValueError("Expected exactly one tool call per turn")
        function = tool_calls[0].get("function") or {}
        name = function.get("name")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments.strip() else {}
        return _tool_model_action(name, arguments, declared_tools, step_idx=step_idx)

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Completion has neither one tool call nor non-empty content")
    native = _NATIVE_CALL_RE.fullmatch(content)
    if native:
        call = json.loads(native.group(1))
        if not isinstance(call, dict):
            raise ValueError("Native tool call must contain an object")
        return _tool_model_action(
            call.get("name"),
            call.get("arguments", {}),
            declared_tools,
            step_idx=step_idx,
        )
    if "<tool_call>" in content or "</tool_call>" in content:
        raise ValueError("Malformed or mixed native tool-call content")
    content = content.strip()
    canonical = json.dumps(
        {"name": "respond", "arguments": {"content": content}}, ensure_ascii=False
    )
    return ModelAction(
        raw_text=content,
        env_text=f"<tool_call>{canonical}</tool_call>",
        name=None,
        arguments=None,
        history_message={"role": "assistant", "content": content},
        tool_call_id=None,
    )


def _tool_model_action(
    name: Any, arguments: Any, declared_tools: set[str], *, step_idx: int
) -> ModelAction:
    if not isinstance(name, str) or not name:
        raise ValueError("Tool call is missing a function name")
    if name not in declared_tools:
        raise ValueError(f"Undeclared tool call: {name}")
    if not isinstance(arguments, dict):
        raise TypeError("Tool-call arguments must be an object")
    call_id = (
        f"call_{step_idx}_{stable_int(canonical_json([name, arguments]), bits=32):08x}"
    )
    canonical = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return ModelAction(
        raw_text=f"<tool_call>{canonical}</tool_call>",
        env_text=f"<tool_call>{canonical}</tool_call>",
        name=name,
        arguments=arguments,
        history_message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            ],
        },
        tool_call_id=call_id,
    )


def _state_details(env: Any) -> dict[str, Any]:
    orchestrator = getattr(env, "_orchestrator", None)
    environment = getattr(orchestrator, "environment", None)
    if environment is None:
        return {
            "agent_db_hash": None,
            "user_db_hash": None,
            "state_hash": sha256_json([None, None]),
        }
    agent_hash = environment.get_db_hash()
    user_hash = environment.get_user_db_hash()
    return {
        "agent_db_hash": agent_hash,
        "user_db_hash": user_hash,
        "state_hash": sha256_json([agent_hash, user_hash]),
    }


def _observation_messages(
    observation: str, action: ModelAction
) -> list[dict[str, Any]]:
    parsed = split_observation(observation)
    messages: list[dict[str, Any]] = []
    if action.name is not None:
        content = parsed.tool or parsed.other or "[no_observation]"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": action.tool_call_id,
                "name": action.name,
                "content": content,
            }
        )
        if parsed.user:
            messages.append({"role": "user", "content": parsed.user})
    else:
        content = parsed.user or parsed.other
        if content:
            messages.append({"role": "user", "content": content})
    return messages


def _official_success(reward: float) -> bool:
    from tau2.metrics.agent_metrics import is_successful

    return bool(is_successful(reward))


def _termination_reason(env: Any) -> str | None:
    simulation = getattr(env, "_simulation_run", None)
    reason = getattr(simulation, "termination_reason", None)
    return getattr(reason, "value", None) or (
        str(reason) if reason is not None else None
    )


def _simulation_payload(env: Any) -> dict[str, Any] | None:
    simulation = getattr(env, "_simulation_run", None)
    if simulation is None:
        return None
    return redact_sensitive(simulation.model_dump(mode="json"))


def _render_system_prompt(
    template: str, *, domain: str, policy: str, tools: list[dict[str, Any]]
) -> str:
    return (
        template.replace("{domain}", domain)
        .replace("{policy}", policy)
        .replace(
            "{tools_schema}",
            json.dumps(tools, ensure_ascii=False, indent=2, sort_keys=True),
        )
    )


async def sample_one(
    *,
    client: SGLangClient,
    spec: TaskSpec,
    rollout_index: int,
    run_id: str,
    config: dict[str, Any],
    prompt_template: str,
    environment_version: str,
) -> dict[str, Any]:
    from tau2.gym.gym_agent import AgentGymEnv

    sampling_cfg = config["sampling"]
    simulator_cfg = config["simulator"]
    task_seed = int(sampling_cfg["base_seed"]) + stable_int(spec.key)
    rollout_seed = task_seed + rollout_index
    user_args: dict[str, Any] = {
        "temperature": float(simulator_cfg["temperature"]),
        "seed": rollout_seed,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": bool(simulator_cfg["thinking"])}
        },
    }
    endpoint_env = simulator_cfg.get("api_base_env")
    user_endpoint = (
        os.environ.get(str(endpoint_env), "").strip() if endpoint_env else ""
    )
    if user_endpoint:
        user_api_key = (
            os.environ.get("TAU2_USER_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
            or "dummy-key-for-local-server"
        )
        user_args.update(
            {"api_base": user_endpoint, "api_key": user_api_key}
        )

    env = AgentGymEnv(
        domain=spec.domain,
        task_id=spec.task_id,
        max_steps=int(sampling_cfg["max_steps"]),
        solo_mode=False,
        user_llm=str(simulator_cfg["model"]),
        user_llm_args=user_args,
        all_messages_as_observation=False,
    )
    trajectory_id = (
        f"{spec.domain}_{stable_int(spec.task_id, bits=48):012x}_r{rollout_index:03d}"
    )
    steps: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    reward = 0.0
    info: dict[str, Any] = {}
    status = "raw"
    error: str | None = None
    terminated = False
    initial_observation = ""

    try:
        initial_observation, reset_info = await asyncio.to_thread(
            env.reset, seed=rollout_seed
        )
        tools = [
            tool if isinstance(tool, dict) else tool.openai_schema
            for tool in reset_info.get("tools", [])
        ]
        declared = {
            tool.get("function", {}).get("name")
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
        }
        declared = {name for name in declared if isinstance(name, str) and name}
        if not declared:
            raise ValueError("Environment exposed no declared tools")
        system_prompt = _render_system_prompt(
            prompt_template,
            domain=spec.domain,
            policy=str(reset_info.get("policy", "")),
            tools=tools,
        )
        messages = [{"role": "system", "content": system_prompt}]
        if initial_observation.strip():
            initial_parsed = split_observation(initial_observation)
            initial_content = (
                initial_parsed.user or initial_parsed.other or initial_observation
            )
            messages.append({"role": "user", "content": initial_content})
        initial_state = _state_details(env)

        request_sampling = {
            "temperature": float(sampling_cfg["temperature"]),
            "top_p": float(sampling_cfg["top_p"]),
            "max_tokens": int(sampling_cfg["max_response_tokens"]),
            "seed": rollout_seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        for step_idx in range(int(sampling_cfg["max_steps"])):
            state_before = _state_details(env)
            response = await client.complete(
                messages=messages, tools=tools, sampling=request_sampling
            )
            choice = (response.get("choices") or [{}])[0]
            finish_reason = str(choice.get("finish_reason") or "")
            step: dict[str, Any] = {
                "step_idx": step_idx,
                "assistant_response": redact_sensitive(response),
                "assistant_raw_text": "",
                "tool_call": None,
                "tool_observation": None,
                "state_hash_before": state_before["state_hash"],
                "state_hash_after": state_before["state_hash"],
                "state_changed": False,
                "step_reward": 0.0,
                "flags": [],
                "finish_reason": finish_reason or None,
            }
            if finish_reason in {"length", "max_tokens", "max_token", "token_limit"}:
                step["flags"].append("overlong")
                steps.append(step)
                status = "overlong"
                error = "model completion reached the response token limit"
                break
            try:
                action = _model_action(response, declared, step_idx=step_idx)
                parsed_action = parse_action(action.env_text)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                step["flags"].append("invalid_schema")
                steps.append(step)
                status = "parse_error"
                error = str(exc)
                break

            step["assistant_raw_text"] = action.raw_text
            if action.name is not None:
                step["tool_call"] = {"name": action.name, "arguments": action.arguments}
            messages.append(action.history_message)
            env_action = env_action_from_parsed_action(parsed_action)
            observation, reward, terminated, truncated, info = await asyncio.to_thread(
                env.step, env_action
            )
            state_after = _state_details(env)
            step.update(
                {
                    "tool_observation": observation,
                    "state_hash_after": state_after["state_hash"],
                    "state_changed": state_before["state_hash"]
                    != state_after["state_hash"],
                    "step_reward": float(reward),
                    "truncated": bool(truncated),
                }
            )
            steps.append(step)
            messages.extend(_observation_messages(observation, action))
            if terminated:
                break
        else:
            status = "truncated"
            error = "maximum environment steps reached"

        final_state = _state_details(env)
        terminal_reason = _termination_reason(env)
        official_success = terminated and _official_success(float(reward))
        terminal = {
            "done": bool(terminated),
            "success": bool(official_success),
            "official_score": float(reward),
            "reason": terminal_reason or status,
            "state_hash": final_state["state_hash"],
        }
        if terminated and status == "raw":
            status = "raw"
    except Exception as exc:  # noqa: BLE001 - infrastructure failures become raw records
        tools = locals().get("tools", [])
        system_prompt = locals().get("system_prompt", "")
        initial_state = locals().get("initial_state", _state_details(env))
        final_state = _state_details(env)
        terminal = {
            "done": False,
            "success": False,
            "official_score": float(reward),
            "reason": "infrastructure_error",
            "state_hash": final_state["state_hash"],
        }
        status = "infrastructure_error"
        error = f"{type(exc).__name__}: {exc}"

    schema_hash = sha256_json(tools)
    task_snapshot = spec.task.model_dump(mode="json")
    prompt_hash = sha256_json(system_prompt)
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "trajectory_id": trajectory_id,
        "task_id": spec.task_id,
        "task_index": spec.task_index,
        "domain": spec.domain,
        "split": spec.split,
        "model": config["model"]["name"],
        "checkpoint": config["model"]["checkpoint"],
        "sampling": {
            "temperature": float(sampling_cfg["temperature"]),
            "top_p": float(sampling_cfg["top_p"]),
            "max_response_tokens": int(sampling_cfg["max_response_tokens"]),
            "thinking": False,
            "seed": rollout_seed,
            "task_seed": task_seed,
            "rollout_index": rollout_index,
        },
        "env": {
            "version": environment_version,
            "seed": rollout_seed,
            "initial_state_hash": initial_state["state_hash"],
            "initial_agent_db_hash": initial_state["agent_db_hash"],
            "initial_user_db_hash": initial_state["user_db_hash"],
            "initial_state_snapshot_hash": sha256_json(
                task_snapshot.get("initial_state")
            ),
        },
        "user_persona": task_snapshot.get("user_scenario", {}).get("persona"),
        "system_prompt_hash": prompt_hash,
        "system_prompt": system_prompt,
        "tools": {"schema": tools, "schema_hash": schema_hash},
        "messages": messages,
        "steps": steps,
        "simulation": _simulation_payload(env),
        "terminal": terminal,
        "verifier": {"schema": {}, "process": {}, "terminal": {}},
        "status": status,
        "error": error,
        "created_at_utc": utc_now(),
    }
    record["fingerprints"] = {"action_fingerprint": action_fingerprint(record)}
    return redact_sensitive(record)


def load_task_specs(config: dict[str, Any]) -> list[TaskSpec]:
    from tau2.registry import registry

    task_cfg = config["tasks"]
    requested_ids = {str(task_id) for task_id in (task_cfg.get("task_ids") or [])}
    maximum = task_cfg.get("max_tasks_per_domain")
    specs: list[TaskSpec] = []
    leakage_ids: set[tuple[str, str]] = set()
    for domain in task_cfg["domains"]:
        loader = registry.get_tasks_loader(domain)
        for split in ("test", "dev"):
            try:
                held_out_tasks = loader(split)
            except (FileNotFoundError, KeyError, ValueError):
                held_out_tasks = []
            leakage_ids.update((domain, str(task.id)) for task in held_out_tasks)
        tasks = loader("train")
        selected = [
            (index, task)
            for index, task in enumerate(tasks)
            if not requested_ids or str(task.id) in requested_ids
        ]
        if maximum is not None:
            selected = selected[: int(maximum)]
        specs.extend(
            TaskSpec(
                domain=domain,
                split="train",
                task_index=index,
                task_id=str(task.id),
                task=task,
            )
            for index, task in selected
        )
    selected_ids = {(spec.domain, spec.task_id) for spec in specs}
    overlap = selected_ids.intersection(leakage_ids)
    if overlap:
        raise ValueError(
            f"Task ID leakage between train and held-out splits: {sorted(overlap)[:10]}"
        )
    selected_task_ids = {spec.task_id for spec in specs}
    if requested_ids and requested_ids.difference(selected_task_ids):
        missing = sorted(requested_ids.difference(selected_task_ids))
        raise ValueError(
            f"Requested task IDs were not found in selected train domains: {missing[:10]}"
        )
    if not specs:
        raise ValueError("No train tasks selected")
    return specs


def _environment_version(repo_root: Path) -> str:
    submodule = repo_root / "third_party" / "tau2-bench"
    return f"tau2@{git_commit(submodule)}"


def _checkpoint_identity(checkpoint: str) -> dict[str, Any]:
    path = Path(checkpoint)
    identity: dict[str, Any] = {"path_or_id": checkpoint}
    if path.exists():
        identity["resolved_path"] = str(path.resolve())
        config_path = path / "config.json" if path.is_dir() else path
        if config_path.is_file():
            from data_pipeline.rft.io import sha256_file

            identity["config_sha256"] = sha256_file(config_path)
    return identity


def initialize_run(
    *,
    repo_root: Path,
    run_id: str,
    resolved_config: dict[str, Any],
    specs: list[TaskSpec],
) -> Path:
    run_root = Path(resolved_config["run"]["root"]) / run_id
    for relative in ("rollouts", "exports", "reports"):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    write_yaml_once(run_root / "config.resolved.yaml", resolved_config)

    snapshot_path = run_root / "tasks.snapshot.jsonl"
    existing = {row["task_key"]: row for row in iter_jsonl(snapshot_path)}
    new_rows: list[dict[str, Any]] = []
    for spec in specs:
        payload = spec.task.model_dump(mode="json")
        row = {
            "task_key": spec.key,
            "task_id": spec.task_id,
            "task_index": spec.task_index,
            "domain": spec.domain,
            "split": "train",
            "task": payload,
            "task_sha256": sha256_json(payload),
        }
        if (
            spec.key in existing
            and existing[spec.key]["task_sha256"] != row["task_sha256"]
        ):
            raise ValueError(f"Task snapshot changed for resumed run: {spec.key}")
        if spec.key not in existing:
            new_rows.append(row)
    append_jsonl(snapshot_path, new_rows)

    manifest_path = run_root / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at_utc": utc_now(),
            "model": _checkpoint_identity(str(resolved_config["model"]["checkpoint"])),
            "tokenizer": resolved_config["model"].get(
                "tokenizer", resolved_config["model"]["checkpoint"]
            ),
            "sampling": resolved_config["sampling"],
            "task_snapshot_sha256": None,
            "environment_version": _environment_version(repo_root),
            "tool_schema_version": resolved_config["prompt"]["schema_version"],
            "system_prompt_template_sha256": resolved_config["prompt"][
                "template_sha256"
            ],
            "code_commit": resolved_config["_meta"]["code_commit"],
            "python": platform.python_version(),
        }
        atomic_write_json(manifest_path, manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    from data_pipeline.rft.io import sha256_file

    manifest["task_snapshot_sha256"] = sha256_file(snapshot_path)
    atomic_write_json(manifest_path, manifest)
    return run_root


def _provisionally_qualified(record: dict[str, Any]) -> bool:
    from data_pipeline.rft.verifier import schema_verifier

    if record.get("status") != "raw" or not record.get("terminal", {}).get("success"):
        return False
    if not schema_verifier(record)["passed"]:
        return False
    repeated_no_progress: set[tuple[str, str]] = set()
    for step in record.get("steps") or []:
        if step.get("flags"):
            return False
        observation = str(step.get("tool_observation") or "").lower()
        if "error:" in observation or "invalid action" in observation:
            return False
        if step.get("tool_call") and step.get("state_hash_before") == step.get(
            "state_hash_after"
        ):
            signature = sha256_json(step["tool_call"])
            key = (str(step.get("state_hash_before")), signature)
            if key in repeated_no_progress:
                return False
            repeated_no_progress.add(key)
    return True


async def run_sampling(
    *, repo_root: Path, run_id: str, config: dict[str, Any], specs: list[TaskSpec]
) -> dict[str, int]:
    run_root = Path(config["run"]["root"]) / run_id
    raw_path = run_root / "rollouts" / "raw.jsonl"
    existing_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(raw_path):
        existing_by_task.setdefault(f"{row['domain']}/{row['task_id']}", []).append(row)
    prompt_template = Path(config["prompt"]["template"]).read_text(encoding="utf-8")
    environment_version = _environment_version(repo_root)
    client = SGLangClient(
        str(config["model"]["api_base"]), str(config["model"]["name"])
    )
    semaphore = asyncio.Semaphore(int(config["sampling"].get("max_concurrency", 8)))
    counts = {
        "tasks": len(specs),
        "new_rollouts": 0,
        "reused_rollouts": sum(map(len, existing_by_task.values())),
    }
    append_lock = asyncio.Lock()

    async def sample_task(spec: TaskSpec) -> list[dict[str, Any]]:
        async with semaphore:
            prior = sorted(
                existing_by_task.get(spec.key, []),
                key=lambda row: row["sampling"]["rollout_index"],
            )
            records = list(prior)
            initial = int(config["sampling"]["initial_candidates"])
            increment = int(config["sampling"]["additional_candidates"])
            maximum = int(config["sampling"]["max_candidates"])
            quota = int(config["sampling"]["success_quota"])
            new_records: list[dict[str, Any]] = []
            while True:
                qualified_fingerprints = {
                    action_fingerprint(row)
                    for row in records
                    if _provisionally_qualified(row)
                }
                if len(qualified_fingerprints) >= quota or len(records) >= maximum:
                    break
                if len(records) < initial:
                    next_target = initial
                else:
                    completed_increments = (len(records) - initial) // increment
                    next_target = min(
                        maximum, initial + (completed_increments + 1) * increment
                    )
                while len(records) < next_target:
                    rollout_index = len(records)
                    record = await sample_one(
                        client=client,
                        spec=spec,
                        rollout_index=rollout_index,
                        run_id=run_id,
                        config=config,
                        prompt_template=prompt_template,
                        environment_version=environment_version,
                    )
                    records.append(record)
                    async with append_lock:
                        append_jsonl(raw_path, [record])
                        counts["new_rollouts"] += 1
                    new_records.append(record)
            return new_records

    try:
        await asyncio.gather(*(sample_task(spec) for spec in specs))
    finally:
        await client.close()
    return counts


def default_run_id(config: dict[str, Any], repo_root: Path) -> str:
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    checkpoint = re.sub(
        r"[^a-zA-Z0-9._-]+", "-", Path(str(config["model"]["checkpoint"])).name
    ).strip("-")
    return f"{timestamp}_qwen3-4b_{checkpoint}_{git_commit(repo_root)[:8]}"
