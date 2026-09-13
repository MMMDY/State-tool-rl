#!/usr/bin/env python3
"""Tau2 Pass^k evaluation for an SGLang-served policy."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import yaml

from tau2_rl_pipeline.actions import (
    env_action_from_parsed_action,
    followup_messages_for_observation,
    parse_action,
    split_observation,
)
from tau2_rl_pipeline.env import compute_partial_score_from_reward_info, parse_reward_info
from tau2_rl_pipeline.prompting import build_tau2_agent_system_prompt

logger = logging.getLogger(__name__)

DEFAULT_DOMAINS = ("airline", "retail", "telecom")
OFFICIAL_METRIC_NOTE = (
    "Tau2 Pass^k is computed by tau2.metrics.agent_metrics.compute_metrics "
    "using C(success_count, k) / C(num_trials, k). "
    "best_of_k_success is reported separately as a diagnostic, not as Pass^k."
)
_SENSITIVE_FIELD_MARKERS = ("api_key", "authorization", "password", "secret")
_SENSITIVE_FIELD_NAMES = {"token", "access_token", "refresh_token", "id_token"}
DEFAULT_POLICY_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "qwen3-4b.yaml"
DEFAULT_SIMULATOR_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "simulator.yaml"


def _official_success(reward: float) -> bool:
    """Use Tau2's own tolerance-aware definition of a successful rollout."""
    from tau2.metrics.agent_metrics import is_successful

    return is_successful(reward)


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _get_user_llm_args(
    *, temperature: float, enable_thinking: bool, reasoning_effort: str | None
) -> dict[str, Any]:
    """Build simulator arguments, using DeepSeek's OpenAI-compatible thinking API."""
    args: dict[str, Any] = {
        "temperature": temperature,
        "extra_body": {"thinking": {"type": "enabled" if enable_thinking else "disabled"}},
    }
    if enable_thinking:
        args["reasoning_effort"] = reasoning_effort or "high"

    api_base = os.environ.get("TAU2_USER_API_BASE", "").strip()
    if api_base:
        args["api_base"] = api_base
        args["api_key"] = "dummy-key-for-local-server"
        # Local SGLang user simulators use their chat-template switch instead
        # of DeepSeek's thinking payload.
        args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
    return args


def _json_safe(value: Any) -> Any:
    """Produce serializable trace data while preventing credentials from being written."""
    try:
        normalized = json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        normalized = str(value)
    return _redact_sensitive_fields(normalized)


def _redact_sensitive_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[redacted]" if _is_sensitive_field_name(key) else _redact_sensitive_fields(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_fields(item) for item in value]
    return value


def _is_sensitive_field_name(name: str) -> bool:
    normalized = name.lower()
    return normalized in _SENSITIVE_FIELD_NAMES or any(marker in normalized for marker in _SENSITIVE_FIELD_MARKERS)


def _configured_simulator_endpoint() -> str | None:
    for name in ("TAU2_USER_API_BASE", "OPENAI_API_BASE", "OPENAI_BASE_URL"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _configured_judge() -> dict[str, Any]:
    """Return non-secret settings for Tau2's NL-assertion judge."""
    return {
        "model": os.environ.get("TAU2_JUDGE_MODEL", "openai/deepseek-flash").strip(),
        "api_endpoint": os.environ.get("TAU2_JUDGE_API_BASE", "").strip() or None,
        "temperature": 0.1,
    }


@dataclass(frozen=True, slots=True)
class AttemptResult:
    success: bool
    reward: float
    partial_score: float
    partial_components: dict[str, float]
    steps: int
    status: str
    error: str | None = None
    reward_info: dict[str, Any] | None = None
    completion_diagnostics: list[dict[str, Any]] | None = None
    trajectory: list[dict[str, Any]] | None = None
    trajectory_context: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PassKResult:
    domain: str
    task_split: str
    task_index: int
    task_id: str
    num_samples: int
    best_success: bool
    best_reward: float
    best_partial_score: float
    best_sample_idx: int
    trajectory_context: dict[str, Any]
    attempts: list[dict[str, Any]]
    first_sample_success: float
    best_of_k_success: float

@dataclass(frozen=True, slots=True)
class TaskSpec:
    domain: str
    task_split: str
    task_index: int
    task_id: str

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.task_split}/{self.task_id}"



class SGLangClient:
    """Small OpenAI-compatible client for SGLang Chat Completions."""

    def __init__(self, url: str, *, model: str) -> None:
        self.url = f"{url.rstrip('/')}/v1/chat/completions"
        self.model = model
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

    async def close(self) -> None:
        await self._client.aclose()

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        sampling_params: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            **sampling_params,
        }
        resp = await self._client.post(self.url, json=payload)
        resp.raise_for_status()
        return resp.json()


def _canonical_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Adapt OpenAI/SGLang function calls to the existing tau2 action bridge."""
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=False)}</tool_call>"


def _declared_tool_names(tools_openai: list[dict[str, Any]]) -> set[str]:
    """Return the exact function names exposed to the policy for this task."""
    names: set[str] = set()
    for tool in tools_openai:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            names.add(name)
    if not names:
        raise ValueError("Task exposes no valid function names in its tools schema")
    return names


def _require_declared_tool_name(name: str, declared_tool_names: set[str]) -> None:
    if name not in declared_tool_names:
        raise ValueError(f"Tool call {name!r} is not declared in this task's tools schema")


def _native_thinking_content(content: str) -> str | None:
    """Return a sole native Qwen thinking block, rejecting mixed text."""
    stripped = content.strip()
    start_tag = "<think>"
    end_tag = "</think>"
    if not stripped.startswith(start_tag) or not stripped.endswith(end_tag):
        return None
    inner = stripped[len(start_tag) : -len(end_tag)]
    if start_tag in inner or end_tag in inner:
        return None
    return inner.strip()


def _single_native_tool_call(content: str) -> tuple[dict[str, Any], str] | None:
    """Strictly extract one Qwen native tool call and its non-call residue.

    This is deliberately a narrow compatibility boundary, not a replacement for
    SGLang's parser: exactly one complete JSON call is accepted, and callers
    must decide whether the remaining text is safe to ignore.
    """
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    start = content.find(start_tag)
    if start == -1:
        return None
    end = content.find(end_tag, start + len(start_tag))
    if end == -1 or content.find(start_tag, end + len(end_tag)) != -1:
        return None
    try:
        call = json.loads(content[start + len(start_tag) : end].strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(call, dict):
        return None
    residue = (content[:start] + content[end + len(end_tag) :]).strip()
    return call, residue


def _action_from_chat_completion(
    response: dict[str, Any], *, declared_tool_names: set[str]
) -> tuple[str, str | None]:
    """Return a canonical tau2 action and separately retain model reasoning."""
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("Chat Completions response contains no choices")

    message = choices[0].get("message") or {}
    reasoning = message.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        if len(tool_calls) != 1:
            raise ValueError(f"Expected exactly one tool call, received {len(tool_calls)}")
        function = tool_calls[0].get("function") or {}
        name = function.get("name")
        arguments = function.get("arguments", {})
        if not isinstance(name, str) or not name:
            raise ValueError("Structured tool call is missing a function name")
        _require_declared_tool_name(name, declared_tool_names)
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments.strip() else {}
        if not isinstance(arguments, dict):
            raise ValueError("Structured tool-call arguments must be an object")
        return _canonical_tool_call(name, arguments), reasoning

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        content = content.strip()
        # SGLang's structured parser is the primary path. When it leaves one
        # native Qwen call in content, adapt that one well-formed call instead
        # of accidentally forwarding it as a customer-facing response.
        inline_call = _single_native_tool_call(content)
        if inline_call is not None:
            call, residue = inline_call
            inline_reasoning = _native_thinking_content(residue) if residue else ""
            if not residue or inline_reasoning is not None:
                name = call.get("name")
                arguments = call.get("arguments", {})
                if not isinstance(name, str) or not name:
                    raise ValueError("Native tool call is missing a function name")
                _require_declared_tool_name(name, declared_tool_names)
                if not isinstance(arguments, dict):
                    raise ValueError("Native tool-call arguments must be an object")
                return _canonical_tool_call(name, arguments), reasoning or inline_reasoning
        if "<tool_call>" in content or "</tool_call>" in content:
            raise ValueError("Native tool call was not exactly one valid standalone JSON call")
        if _native_thinking_content(content) is not None:
            raise ValueError("Chat Completions response contains reasoning only and no action")
        return _canonical_tool_call("respond", {"content": content}), reasoning
    raise ValueError("Chat Completions response has neither tool_calls nor content")


def _assistant_history_message(
    response: dict[str, Any], *, assistant_action: str, parsed_name: str
) -> dict[str, str]:
    """Preserve a normal assistant reply instead of leaking Tau2's respond shim."""
    if parsed_name != "respond":
        return {"role": "assistant", "content": assistant_action}
    content = _message_from_chat_completion(response).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Internal respond action requires a non-empty normal assistant response")
    content = content.strip()
    if "<tool_call>" in content or "</tool_call>" in content:
        raise ValueError("A native tool-call block cannot be replayed as a normal assistant response")
    return {"role": "assistant", "content": content}


def _message_from_chat_completion(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    return (choices[0].get("message") or {}) if choices else {}


def _trajectory_tool_call(
    *, name: Any, arguments: Any, call_id: Any = "", call_type: Any = "function"
) -> dict[str, Any] | None:
    """Normalize one OpenAI-style function call for the saved trajectory."""
    if not isinstance(name, str) or not name:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return {
        "id": str(call_id or ""),
        "type": str(call_type or "function"),
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _extract_inline_trajectory_tool_calls(content: str) -> tuple[str, list[dict[str, Any]]]:
    """Move native ``<tool_call>`` blocks out of assistant content.

    SGLang normally returns ``message.tool_calls``. Some responses instead keep
    Qwen's native XML block in ``message.content``; retaining it there makes
    saved trajectories ambiguous and hard to replay.
    """
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    cursor = 0
    plain_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    while True:
        start = content.find(start_tag, cursor)
        if start == -1:
            plain_parts.append(content[cursor:])
            break
        end = content.find(end_tag, start + len(start_tag))
        if end == -1:
            return content, []
        plain_parts.append(content[cursor:start])
        raw_call = content[start + len(start_tag) : end].strip()
        try:
            call = json.loads(raw_call)
        except json.JSONDecodeError:
            return content, []
        if not isinstance(call, dict):
            return content, []
        normalized = _trajectory_tool_call(
            name=call.get("name"),
            arguments=call.get("arguments", {}),
            call_id=f"inline_{len(tool_calls)}_{call.get('name', '')}",
        )
        if normalized is None:
            return content, []
        tool_calls.append(normalized)
        cursor = end + len(end_tag)
    return "".join(plain_parts).strip(), tool_calls


def _completion_diagnostics(
    response: dict[str, Any], *, declared_tool_names: set[str]
) -> dict[str, Any]:
    """Record the completion shape needed to diagnose parser and token failures."""
    choices = response.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    content = content if isinstance(content, str) else ""
    reasoning = message.get("reasoning_content")
    reasoning = reasoning if isinstance(reasoning, str) else "" if reasoning is None else str(reasoning)
    structured_calls = message.get("tool_calls") or []
    call_names = [
        (call.get("function") or {}).get("name")
        for call in structured_calls
        if isinstance(call, dict) and isinstance(call.get("function"), dict)
    ]
    native_call = _single_native_tool_call(content)
    if native_call is not None:
        call_names.append(native_call[0].get("name"))
    undeclared_names = sorted(
        {
            name if isinstance(name, str) else "<missing function name>"
            for name in call_names
            if not isinstance(name, str) or name not in declared_tool_names
        }
    )
    finish_reason = choice.get("finish_reason")
    finish_reason = str(finish_reason) if finish_reason is not None else None
    length_truncated = finish_reason in {"length", "max_tokens", "max_token", "token_limit"}

    if undeclared_names:
        response_kind = "invalid_tool_call"
    elif structured_calls:
        response_kind = "structured_tool_call"
    elif native_call is not None:
        response_kind = "native_tool_call_unstructured"
    elif not content.strip() or _native_thinking_content(content) is not None:
        response_kind = "reasoning_only" if reasoning.strip() or content.strip() else "empty"
    else:
        response_kind = "content"

    if response_kind == "invalid_tool_call":
        diagnostic = "undeclared_tool_call"
    elif response_kind == "reasoning_only" and length_truncated:
        diagnostic = "reasoning_only_length_truncated"
    elif response_kind == "reasoning_only":
        diagnostic = "reasoning_only"
    elif length_truncated:
        diagnostic = "length_truncated"
    else:
        diagnostic = "complete"

    return {
        "finish_reason": finish_reason,
        "usage": _json_safe(response.get("usage")) if response.get("usage") is not None else None,
        "reasoning_content_chars": len(reasoning),
        "response_kind": response_kind,
        "undeclared_tool_names": undeclared_names,
        "length_truncated": length_truncated,
        "diagnostic": diagnostic,
    }


def _trajectory_assistant_turn(
    response: dict[str, Any], *, declared_tool_names: set[str]
) -> dict[str, Any]:
    """Format an SGLang response as an example_trajectory.md-compatible turn."""
    message = _message_from_chat_completion(response)
    reasoning = message.get("reasoning_content")
    raw_content = message.get("content")
    content = raw_content if isinstance(raw_content, str) else ""
    content, inline_tool_calls = _extract_inline_trajectory_tool_calls(content)

    structured_tool_calls: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        if not isinstance(function, dict):
            continue
        normalized = _trajectory_tool_call(
            name=function.get("name"),
            arguments=function.get("arguments", {}),
            call_id=call.get("id", ""),
            call_type=call.get("type", "function"),
        )
        if normalized is not None:
            structured_tool_calls.append(normalized)

    valid_tool_calls: list[dict[str, Any]] = []
    invalid_tool_calls: list[dict[str, Any]] = []
    for call in structured_tool_calls or inline_tool_calls:
        name = (call.get("function") or {}).get("name") if isinstance(call, dict) else None
        if isinstance(name, str) and name in declared_tool_names:
            valid_tool_calls.append(call)
        else:
            invalid_tool_calls.append(
                {
                    **call,
                    "error": "Function is not declared in this task's tools schema",
                }
            )

    content_parts: list[str] = []
    if isinstance(reasoning, str) and reasoning.strip():
        content_parts.append(f"<think>\n{reasoning.strip()}\n</think>")
    if content.strip():
        content_parts.append(content.strip())
    return {
        "role": "assistant",
        "content": "\n\n".join(content_parts),
        "tool_calls": valid_tool_calls,
        "invalid_tool_calls": invalid_tool_calls,
        "completion": _completion_diagnostics(response, declared_tool_names=declared_tool_names),
    }


def _trajectory_observation_turns(*, observation: str, last_action_was_tool: bool) -> list[dict[str, Any]]:
    """Mirror the evaluator's observation bridge without serializing duplicate prompts."""
    parsed = split_observation(observation)
    if last_action_was_tool:
        tool_content = parsed.tool or parsed.other
        turns = (
            [{"role": "tool", "content": tool_content, "tool_calls": []}]
            if tool_content
            else []
        )
        if parsed.user:
            turns.append({"role": "user", "content": parsed.user, "tool_calls": []})
        return turns
    user_content = parsed.user or parsed.other
    return [{"role": "user", "content": user_content, "tool_calls": []}] if user_content else []


def _terminal_simulator_turn(*, info: dict[str, Any]) -> dict[str, Any] | None:
    """Recover tau2's final user/tool turn when its terminal observation is empty.

    AgentGymEnv intentionally returns an empty observation after a terminal user
    message such as ``###STOP###``. Its final ``simulation_run`` still contains
    that message, so recover it here instead of inventing ``[no_observation]``.
    The policy's final assistant action is already recorded before ``env.step()``,
    so only user and tool turns are needed from this fallback.
    """
    raw_run = info.get("simulation_run")
    try:
        run = json.loads(raw_run) if isinstance(raw_run, str) else raw_run
    except json.JSONDecodeError:
        return None
    if not isinstance(run, dict):
        return None
    messages = run.get("messages")
    if not isinstance(messages, list):
        return None

    last_message = messages[-1] if messages else None
    if not isinstance(last_message, dict):
        return None
    role = last_message.get("role")
    content = last_message.get("content")
    if role in {"user", "tool"} and isinstance(content, str) and content.strip():
        return {"role": role, "content": content, "tool_calls": []}
    return None


def _load_tasks(domain: str, task_split: str) -> list[str]:
    from tau2.registry import registry

    return [t.id for t in registry.get_tasks_loader(domain)(task_split)]


async def _run_one_attempt(
    *,
    client: SGLangClient,
    domain: str,
    task_id: str,
    sampling_params: dict[str, Any],
    repair_sampling_params: dict[str, Any],
    max_steps: int,
    user_llm: str,
    user_llm_args: dict[str, Any],
) -> AttemptResult:
    from tau2.gym.gym_agent import AgentGymEnv

    env = AgentGymEnv(
        domain=domain,
        task_id=task_id,
        max_steps=max_steps,
        solo_mode=False,
        user_llm=user_llm,
        user_llm_args=user_llm_args,
        all_messages_as_observation=False,
    )

    # AgentGymEnv waits synchronously for the simulator. Moving it off the
    # event loop is what makes task-level asyncio concurrency real.
    observation, info = await asyncio.to_thread(env.reset)
    tools = info.get("tools", [])
    tools_openai = [t if isinstance(t, dict) else t.openai_schema for t in tools]
    declared_tool_names = _declared_tool_names(tools_openai)
    policy = info.get("policy", "")
    simulator_system_prompt = getattr(getattr(env, "_user", None), "system_prompt", "")

    system_prompt = build_tau2_agent_system_prompt(
        domain=domain,
        policy=policy,
        tools_openai=tools_openai,
        include_tool_schema=False,
        use_structured_tool_calls=True,
    )
    trajectory_context = _json_safe(
        {
            "tools": tools_openai,
        }
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    # tau2 starts agent-side simulations with an empty observation: the policy
    # should make the opening greeting, not receive a fabricated blank user turn.
    if observation.strip():
        messages.extend(
            followup_messages_for_observation(
                observation=observation,
                last_action_call="(reset)",
                last_action_was_tool=False,
            )
        )
    trajectory = [
        {
            "role": "system",
            "content": f"<policy_system_prompt>\n{system_prompt}\n</policy_system_prompt>",
            "tool_calls": [],
        },
        {
            "role": "system",
            "content": f"<simulator_system_prompt>\n{simulator_system_prompt}\n</simulator_system_prompt>",
            "tool_calls": [],
        },
        *_trajectory_observation_turns(observation=observation, last_action_was_tool=False),
    ]

    reward = 0.0
    reward_info: dict[str, Any] = {}
    completion_diagnostics: list[dict[str, Any]] = []

    for step in range(max_steps):
        out = await client.chat_completion(messages=messages, tools=tools_openai, sampling_params=sampling_params)
        choice = (out.get("choices") or [{}])[0]
        trajectory.append(_trajectory_assistant_turn(out, declared_tool_names=declared_tool_names))
        completion_diagnostics.append(_completion_diagnostics(out, declared_tool_names=declared_tool_names))
        if choice.get("finish_reason") == "abort":
            return AttemptResult(
                success=False,
                reward=0.0,
                partial_score=0.0,
                partial_components={},
                steps=step,
                status="aborted",
                error="sglang_abort",
                completion_diagnostics=completion_diagnostics,
                trajectory=trajectory,
                trajectory_context=trajectory_context,
            )

        try:
            assistant_text, reasoning_content = _action_from_chat_completion(out, declared_tool_names=declared_tool_names)
            if reasoning_content:
                logger.debug("step=%d received %d reasoning characters", step, len(reasoning_content))
            parsed = parse_action(assistant_text)
        except Exception as exc:
            messages.append(
                {
                    "role": "user",
                    "content": "FORMAT ERROR. Make exactly one valid function call from the provided tools schema "
                    "with JSON object arguments, or provide a normal customer-facing response. Do not call "
                    "respond, done, or any undeclared function.",
                }
            )
            # Keep the configured non-greedy profile for repairs too. Greedy
            # decoding makes Qwen3 tool-call recovery less reliable.
            repair_params = repair_sampling_params.copy()
            out = await client.chat_completion(messages=messages, tools=tools_openai, sampling_params=repair_params)
            trajectory.append(_trajectory_assistant_turn(out, declared_tool_names=declared_tool_names))
            completion_diagnostics.append(_completion_diagnostics(out, declared_tool_names=declared_tool_names))
            try:
                assistant_text, reasoning_content = _action_from_chat_completion(out, declared_tool_names=declared_tool_names)
                if reasoning_content:
                    logger.debug("step=%d repair received %d reasoning characters", step, len(reasoning_content))
                parsed = parse_action(assistant_text)
            except Exception as repair_exc:
                partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)
                return AttemptResult(
                    success=False,
                    reward=float(reward),
                    partial_score=partial_score,
                    partial_components=partial_components,
                    steps=step + 1,
                    status="parse_error",
                    error=f"{exc}; repair failed: {repair_exc}",
                    reward_info=reward_info,
                    completion_diagnostics=completion_diagnostics,
                    trajectory=trajectory,
                    trajectory_context=trajectory_context,
                )

        messages.append(
            _assistant_history_message(response=out, assistant_action=assistant_text, parsed_name=parsed.name)
        )

        env_action = env_action_from_parsed_action(parsed)
        observation, reward, terminated, truncated, info = await asyncio.to_thread(env.step, env_action)
        last_action_was_tool = parsed.name != "respond"
        # Capture the environment reply even when this is the terminal action.
        trajectory.extend(
            _trajectory_observation_turns(observation=observation, last_action_was_tool=last_action_was_tool)
        )

        if terminated:
            if not observation.strip():
                terminal_turn = _terminal_simulator_turn(info=info)
                if terminal_turn is not None:
                    trajectory.append(terminal_turn)
            reward_info = parse_reward_info(info)
            partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)
            return AttemptResult(
                success=_official_success(float(reward)),
                reward=float(reward),
                partial_score=partial_score,
                partial_components=partial_components,
                steps=step + 1,
                status="completed",
                reward_info=reward_info,
                completion_diagnostics=completion_diagnostics,
                trajectory=trajectory,
                trajectory_context=trajectory_context,
            )

        messages.extend(
            followup_messages_for_observation(
                observation=observation,
                last_action_call=parsed.raw_action_call,
                last_action_was_tool=last_action_was_tool,
            )
        )

    partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)
    return AttemptResult(
        success=False,
        reward=float(reward),
        partial_score=partial_score,
        partial_components=partial_components,
        steps=max_steps,
        status="truncated",
        reward_info=reward_info,
        completion_diagnostics=completion_diagnostics,
        trajectory=trajectory,
        trajectory_context=trajectory_context,
    )


def _official_termination_reason(status: Any):
    """Map local attempt outcomes to Tau2's canonical termination taxonomy."""
    from tau2.data_model.simulation import TerminationReason

    if status == "infrastructure_error":
        return TerminationReason.INFRASTRUCTURE_ERROR
    if status == "truncated":
        return TerminationReason.MAX_STEPS
    if status in {"parse_error", "aborted"}:
        return TerminationReason.AGENT_ERROR
    return TerminationReason.AGENT_STOP


def _official_results(results: list[PassKResult], *, k: int):
    """Adapt persisted attempts to official Tau2 result models for metrics.

    Rollouts are still driven by AgentGymEnv so that the SGLang policy can be
    used directly. This adapter delegates Pass^k and infrastructure-error
    semantics to Tau2's official metric implementation.
    """
    from tau2.data_model.simulation import AgentInfo, Info, Results, RewardInfo, SimulationRun, UserInfo
    from tau2.environment.environment import EnvironmentInfo
    from tau2.registry import registry

    task_cache: dict[str, dict[str, Any]] = {}
    official_tasks = []
    simulations = []
    now = datetime.now(timezone.utc).isoformat()

    for result in results:
        if result.domain not in task_cache:
            task_cache[result.domain] = {
                task.id: task for task in registry.get_tasks_loader(result.domain)(result.task_split)
            }
        try:
            source_task = task_cache[result.domain][result.task_id]
        except KeyError as exc:
            raise ValueError(
                f"Unable to load official Tau2 task {result.domain}/{result.task_split}/{result.task_id}"
            ) from exc

        # Namespace IDs so a multi-domain local report cannot accidentally
        # group similarly named tasks from separate Tau2 domains.
        official_task_id = f"{result.domain}/{result.task_split}/{result.task_id}"
        official_tasks.append(source_task.model_copy(update={"id": official_task_id}))
        for trial, attempt in enumerate(result.attempts):
            simulations.append(
                SimulationRun(
                    id=f"{official_task_id}/trial-{trial}",
                    task_id=official_task_id,
                    timestamp=now,
                    start_time=now,
                    end_time=now,
                    duration=0.0,
                    termination_reason=_official_termination_reason(attempt.get("status")),
                    reward_info=RewardInfo(reward=float(attempt["reward"])),
                    trial=trial,
                    info={"local_attempt_status": attempt.get("status")},
                )
            )

    info = Info(
        git_commit="local-sglang-tau2-metrics-adapter",
        num_trials=k,
        max_steps=0,
        max_errors=0,
        user_info=UserInfo(implementation="tau2_user_simulator"),
        agent_info=AgentInfo(implementation="sglang_chat_completions"),
        environment_info=EnvironmentInfo(
            domain_name="multi_domain" if len({result.domain for result in results}) > 1 else (results[0].domain if results else "unknown"),
            policy="Policy is preserved in the run configuration and trajectory artifacts.",
        ),
    )
    return Results(info=info, tasks=official_tasks, simulations=simulations)


def _summarize(results: list[PassKResult], *, k: int) -> dict[str, Any]:
    """Compute leaderboard-aligned metrics through Tau2's official code."""
    from tau2.metrics.agent_metrics import compute_metrics

    if not results:
        return {
            "metric_implementation": "tau2.metrics.agent_metrics.compute_metrics",
            "total_tasks": 0,
            "total_simulations": 0,
            "infrastructure_error_count": 0,
            "avg_reward": 0.0,
            "pass_hat_ks": {},
            "pass^1": None,
            f"pass^{k}": None,
            "diagnostics": {"first_sample_success": 0.0, f"best_of_{k}_success": 0.0, "attempt_status_counts": {}},
        }

    metrics = compute_metrics(_official_results(results, k=k))
    attempts = [attempt for result in results for attempt in result.attempts]
    status_counts = dict(sorted(Counter(str(attempt.get("status", "unknown")) for attempt in attempts).items()))
    total = len(results)
    pass_hat_ks = {str(index): value for index, value in sorted(metrics.pass_hat_ks.items())}
    return {
        "metric_implementation": "tau2.metrics.agent_metrics.compute_metrics",
        "total_tasks": metrics.total_tasks,
        "total_simulations": metrics.total_simulations,
        "infrastructure_error_count": metrics.infra_error_count,
        "avg_reward": metrics.avg_reward,
        "pass_hat_ks": pass_hat_ks,
        "pass^1": metrics.pass_hat_ks.get(1),
        f"pass^{k}": metrics.pass_hat_ks.get(k),
        "diagnostics": {
            "first_sample_success": sum(
                _official_success(float(result.attempts[0]["reward"])) for result in results
            ) / total,
            f"best_of_{k}_success": sum(
                any(_official_success(float(attempt["reward"])) for attempt in result.attempts)
                for result in results
            ) / total,
            "attempt_status_counts": status_counts,
        },
    }


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically persist JSON so an interruption never leaves a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, ensure_ascii=False)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _resolve_run_paths(requested_output: str) -> tuple[Path, Path]:
    """Store every artifact for one invocation in a dedicated run directory."""
    requested_path = Path(requested_output).expanduser().resolve()
    run_dir = requested_path.parent / requested_path.stem
    return run_dir, run_dir / requested_path.name


def _add_run_log_handler(run_dir: Path) -> logging.Handler:
    """Mirror evaluator logs into the run directory while retaining console logs."""
    handler = logging.FileHandler(run_dir / "evaluation.log", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def _task_result_path(*, output_path: Path, spec: TaskSpec) -> Path:
    artifact_dir = output_path.parent / "task_results"
    safe_task_id = spec.task_id.replace("/", "_")
    return artifact_dir / f"{spec.domain}_task_{safe_task_id}.json"


def _checkpoint_path(output_path: Path) -> Path:
    return output_path.parent / "checkpoint.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read persisted evaluation state {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Persisted evaluation state {path} must be a JSON object")
    return payload


def _attempt_report(*, output_path: Path, spec: TaskSpec, sample_idx: int, attempt: AttemptResult) -> dict[str, Any]:
    report = asdict(attempt)
    report.pop("trajectory_context", None)
    report.pop("trajectory", None)
    report["trajectory_file"] = _write_trajectory(
        output_path=output_path,
        domain=spec.domain,
        task_id=spec.task_id,
        sample_idx=sample_idx,
        trajectory=attempt.trajectory,
    )
    return report


def _build_task_result(
    *, spec: TaskSpec, num_samples: int, attempts: list[dict[str, Any] | None], trajectory_context: dict[str, Any]
) -> PassKResult:
    if len(attempts) != num_samples or any(attempt is None for attempt in attempts):
        raise ValueError(f"Task {spec.key} does not have all {num_samples} completed samples")
    complete_attempts = [attempt for attempt in attempts if attempt is not None]
    best_idx = 0
    for index in range(1, len(complete_attempts)):
        candidate = complete_attempts[index]
        best = complete_attempts[best_idx]
        if bool(candidate["success"]) and not bool(best["success"]):
            best_idx = index
        elif bool(candidate["success"]) == bool(best["success"]) and float(candidate["partial_score"]) > float(best["partial_score"]):
            best_idx = index
    best = complete_attempts[best_idx]
    return PassKResult(
        domain=spec.domain,
        task_split=spec.task_split,
        task_index=spec.task_index,
        task_id=spec.task_id,
        num_samples=num_samples,
        best_success=bool(best["success"]),
        best_reward=float(best["reward"]),
        best_partial_score=float(best["partial_score"]),
        best_sample_idx=best_idx,
        trajectory_context=trajectory_context,
        attempts=complete_attempts,
        first_sample_success=1.0 if _official_success(float(complete_attempts[0]["reward"])) else 0.0,
        best_of_k_success=1.0 if any(_official_success(float(attempt["reward"])) for attempt in complete_attempts) else 0.0,
    )


def _task_progress_payload(
    *,
    spec: TaskSpec,
    num_samples: int,
    attempts: list[dict[str, Any] | None],
    trajectory_context: dict[str, Any],
    result: PassKResult | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "completed" if result is not None else "in_progress",
        "domain": spec.domain,
        "task_split": spec.task_split,
        "task_index": spec.task_index,
        "task_id": spec.task_id,
        "num_samples": num_samples,
        "trajectory_context": trajectory_context,
        "attempts": attempts,
        "result": asdict(result) if result is not None else None,
    }


def _load_task_progress(
    *,
    output_path: Path,
    spec: TaskSpec,
    num_samples: int,
    retry_statuses: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any] | None], dict[str, Any], PassKResult | None]:
    path = _task_result_path(output_path=output_path, spec=spec)
    if not path.exists():
        return [None] * num_samples, {}, None
    payload = _read_json(path)
    expected = {
        "domain": spec.domain,
        "task_split": spec.task_split,
        "task_index": spec.task_index,
        "task_id": spec.task_id,
        "num_samples": num_samples,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Persisted task result {path} does not match the requested task")
    attempts = payload.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != num_samples:
        raise ValueError(f"Persisted task result {path} has invalid attempts")
    if any(attempt is not None and not isinstance(attempt, dict) for attempt in attempts):
        raise ValueError(f"Persisted task result {path} has invalid attempt records")
    context = payload.get("trajectory_context")
    if not isinstance(context, dict):
        raise ValueError(f"Persisted task result {path} has invalid trajectory_context")
    retry_indices = [
        index
        for index, attempt in enumerate(attempts)
        if isinstance(attempt, dict) and attempt.get("status") in retry_statuses
    ]
    if retry_indices:
        # Keep successful and model-failure samples intact. The selected records
        # are replaced atomically one by one by _evaluate_task.
        attempts = attempts.copy()
        for index in retry_indices:
            attempts[index] = None
        return attempts, context, None
    raw_result = payload.get("result")
    if raw_result is None:
        return attempts, context, None
    if not isinstance(raw_result, dict) or payload.get("status") != "completed":
        raise ValueError(f"Persisted task result {path} has invalid completion state")
    # Read legacy checkpoints written before the metric names were corrected.
    raw_result = raw_result.copy()
    if "first_sample_success" not in raw_result and "pass_at_1" in raw_result:
        raw_result["first_sample_success"] = raw_result.pop("pass_at_1")
    if "best_of_k_success" not in raw_result and "pass_at_k" in raw_result:
        raw_result["best_of_k_success"] = raw_result.pop("pass_at_k")
    return attempts, context, PassKResult(**raw_result)


def _write_checkpoint(
    *,
    output_path: Path,
    run_configuration: dict[str, Any],
    args: argparse.Namespace,
    domains: list[str],
    specs: list[TaskSpec],
    completed: dict[str, PassKResult],
) -> None:
    _atomic_write_json(
        _checkpoint_path(output_path),
        {
            "schema_version": 1,
            "status": "in_progress",
            "configuration": run_configuration,
            "task_split": args.task_split,
            "domains": domains,
            "k": args.num_samples,
            "completed_task_count": len(completed),
            "results": [asdict(completed[spec.key]) for spec in specs if spec.key in completed],
        },
    )



def _mark_checkpoint_completed(*, output_path: Path) -> None:
    """Mark a checkpoint complete only after the aggregate report is durable."""
    path = _checkpoint_path(output_path)
    if not path.exists():
        return
    checkpoint = _read_json(path)
    checkpoint["status"] = "completed"
    checkpoint["report_file"] = output_path.name
    checkpoint["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(path, checkpoint)

async def _evaluate_task(
    *,
    client: SGLangClient,
    output_path: Path,
    spec: TaskSpec,
    sampling_params: dict[str, Any],
    repair_sampling_params: dict[str, Any],
    max_steps: int,
    num_samples: int,
    user_llm: str,
    user_llm_args: dict[str, Any],
    retry_statuses: frozenset[str],
) -> PassKResult:
    """Resume missing samples for one task and persist each completed sample."""
    attempts, trajectory_context, completed = _load_task_progress(
        output_path=output_path,
        spec=spec,
        num_samples=num_samples,
        retry_statuses=retry_statuses,
    )
    if completed is not None:
        return completed

    task_path = _task_result_path(output_path=output_path, spec=spec)
    for sample_idx, existing_attempt in enumerate(attempts):
        if existing_attempt is not None:
            continue
        try:
            attempt = await _run_one_attempt(
                client=client,
                domain=spec.domain,
                task_id=spec.task_id,
                sampling_params=sampling_params,
                repair_sampling_params=repair_sampling_params,
                max_steps=max_steps,
                user_llm=user_llm,
                user_llm_args=user_llm_args,
            )
        except Exception as exc:
            logger.exception("Task %s sample %d failed", spec.key, sample_idx)
            attempt = AttemptResult(
                success=False,
                reward=0.0,
                partial_score=0.0,
                partial_components={},
                steps=0,
                status="infrastructure_error",
                error=f"{type(exc).__name__}: {exc}",
            )
        if not trajectory_context and attempt.trajectory_context:
            trajectory_context = attempt.trajectory_context
        attempts[sample_idx] = _attempt_report(
            output_path=output_path, spec=spec, sample_idx=sample_idx, attempt=attempt
        )
        _atomic_write_json(
            task_path,
            _task_progress_payload(
                spec=spec,
                num_samples=num_samples,
                attempts=attempts,
                trajectory_context=trajectory_context,
                result=None,
            ),
        )
        logger.info("Persisted task=%s sample=%d/%d", spec.key, sample_idx + 1, num_samples)

    result = _build_task_result(
        spec=spec,
        num_samples=num_samples,
        attempts=attempts,
        trajectory_context=trajectory_context,
    )
    _atomic_write_json(
        task_path,
        _task_progress_payload(
            spec=spec,
            num_samples=num_samples,
            attempts=attempts,
            trajectory_context=trajectory_context,
            result=result,
        ),
    )
    return result


async def _evaluate_tasks_concurrently(
    *,
    client: SGLangClient,
    output_path: Path,
    args: argparse.Namespace,
    domains: list[str],
    sampling_params: dict[str, Any],
    repair_sampling_params: dict[str, Any],
    user_llm_args: dict[str, Any],
    run_configuration: dict[str, Any],
) -> list[PassKResult]:
    retry_statuses = frozenset(_parse_csv(args.retry_statuses))
    specs: list[TaskSpec] = []
    requested_task_ids = set(_parse_csv(args.task_ids)) if args.task_ids else None
    found_task_ids: set[str] = set()
    for domain in domains:
        task_ids = _load_tasks(domain, args.task_split)
        if requested_task_ids is not None:
            task_ids = [task_id for task_id in task_ids if task_id in requested_task_ids]
            found_task_ids.update(task_ids)
        if args.max_tasks_per_domain is not None:
            task_ids = task_ids[: args.max_tasks_per_domain]
        logger.info("Queued domain=%s split=%s tasks=%d k=%d", domain, args.task_split, len(task_ids), args.num_samples)
        specs.extend(
            TaskSpec(domain=domain, task_split=args.task_split, task_index=index, task_id=task_id)
            for index, task_id in enumerate(task_ids)
        )
    if requested_task_ids is not None:
        missing_task_ids = requested_task_ids.difference(found_task_ids)
        if missing_task_ids:
            raise ValueError(
                f"Requested task IDs are unavailable for selected domains/split: {', '.join(sorted(missing_task_ids))}"
            )

    completed: dict[str, PassKResult] = {}
    for spec in specs:
        _, _, existing = _load_task_progress(
            output_path=output_path,
            spec=spec,
            num_samples=args.num_samples,
            retry_statuses=retry_statuses,
        )
        if existing is not None:
            completed[spec.key] = existing
    _write_checkpoint(
        output_path=output_path,
        run_configuration=run_configuration,
        args=args,
        domains=domains,
        specs=specs,
        completed=completed,
    )
    if completed:
        logger.info("Resuming with %d/%d completed tasks", len(completed), len(specs))

    semaphore = asyncio.Semaphore(args.max_concurrency)
    checkpoint_lock = asyncio.Lock()

    async def run_spec(spec: TaskSpec) -> None:
        if spec.key in completed:
            return
        async with semaphore:
            result = await _evaluate_task(
                client=client,
                output_path=output_path,
                spec=spec,
                sampling_params=sampling_params,
                repair_sampling_params=repair_sampling_params,
                max_steps=args.max_steps,
                num_samples=args.num_samples,
                user_llm=args.user_model,
                user_llm_args=user_llm_args,
                retry_statuses=retry_statuses,
            )
        async with checkpoint_lock:
            completed[spec.key] = result
            _write_checkpoint(
                output_path=output_path,
                run_configuration=run_configuration,
                args=args,
                domains=domains,
                specs=specs,
                completed=completed,
            )
            logger.info("Completed task=%s progress=%d/%d", spec.key, len(completed), len(specs))

    await asyncio.gather(*(run_spec(spec) for spec in specs))
    return [completed[spec.key] for spec in specs]


def _write_trajectory(
    *,
    output_path: Path,
    domain: str,
    task_id: str,
    sample_idx: int,
    trajectory: list[dict[str, Any]] | None,
) -> str:
    """Write one example_trajectory.md-compatible role-based conversation."""
    artifact_dir = output_path.parent / "trajectories"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    safe_task_id = task_id.replace("/", "_")
    artifact_path = artifact_dir / f"{domain}_task_{safe_task_id}_sample_{sample_idx}.json"
    _atomic_write_json(artifact_path, trajectory or [])
    return str(artifact_path.relative_to(output_path.parent))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate tau2-bench with official Tau2 Pass^k metrics")
    parser.add_argument("--hf-checkpoint", help="Deprecated compatibility field; SGLang loads the model server-side.")
    parser.add_argument("--sglang-url", required=True)
    parser.add_argument("--sglang-model", default="qwen3-4b", help="Model name exposed by SGLang Chat Completions")
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=DEFAULT_POLICY_CONFIG_PATH,
        help="YAML file containing Qwen3-4B thinking and non-thinking sampling profiles.",
    )
    parser.add_argument(
        "--simulator-config",
        type=Path,
        default=DEFAULT_SIMULATOR_CONFIG_PATH,
        help="YAML file containing simulator model, sampling, and thinking settings.",
    )
    parser.add_argument("--output", required=True, help="Report filename; artifacts are grouped in a sibling run directory.")
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--task-split", default="test", choices=("train", "test", "base"))
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated task IDs to evaluate; filters each selected domain before --max-tasks-per-domain.",
    )
    parser.add_argument("--max-tasks-per-domain", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("TAU2_MAX_STEPS", "100")))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=16,
        help="Maximum number of tasks evaluated concurrently (default: 16).",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Qwen3 thinking mode (default: enabled; use --no-enable-thinking to disable).",
    )
    parser.add_argument("--temperature", type=float, default=None, help="Override the Qwen3 mode default.")
    parser.add_argument("--top-p", type=float, default=None, help="Override the Qwen3 mode default.")
    parser.add_argument("--top-k", type=int, default=None, help="Override the Qwen3 mode default.")
    parser.add_argument("--min-p", type=float, default=None, help="Override the Qwen3 mode default.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the active profile's main completion token budget.",
    )
    parser.add_argument(
        "--repair-max-new-tokens",
        type=int,
        default=None,
        help="Override the active profile's independent format-repair token budget.",
    )
    parser.add_argument("--user-model", default=None, help="Override the simulator model from --simulator-config.")
    parser.add_argument("--user-temperature", type=float, default=None, help="Override simulator temperature.")
    parser.add_argument(
        "--user-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override simulator reasoning (default: use --simulator-config; disabled there).",
    )
    parser.add_argument(
        "--retry-statuses",
        default="",
        help="Comma-separated persisted attempt statuses to rerun; e.g. infrastructure_error.",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")

    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be >= 1")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.repair_max_new_tokens is not None and args.repair_max_new_tokens < 1:
        parser.error("--repair-max-new-tokens must be >= 1")
    invalid_retry_statuses = set(_parse_csv(args.retry_statuses)).difference(
        {"aborted", "infrastructure_error", "parse_error", "truncated"}
    )
    if invalid_retry_statuses:
        parser.error(
            "--retry-statuses contains unsupported values: "
            + ", ".join(sorted(invalid_retry_statuses))
        )

def _load_policy_config(path: Path) -> dict[str, Any]:
    """Load and validate the small, versioned Qwen3 inference profile file."""
    try:
        with path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as exc:
        raise ValueError(f"Unable to read policy config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in policy config {path}: {exc}") from exc

    if not isinstance(config, dict) or not isinstance(config.get("profiles"), dict):
        raise ValueError("Policy config must contain a 'profiles' mapping")
    return config


def _load_simulator_config(path: Path) -> dict[str, Any]:
    """Load and validate simulator model and generation settings."""
    try:
        with path.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as exc:
        raise ValueError(f"Unable to read simulator config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in simulator config {path}: {exc}") from exc

    if not isinstance(config, dict) or not isinstance(config.get("model"), str):
        raise ValueError("Simulator config must contain a string 'model'")
    sampling = config.get("sampling")
    thinking = config.get("thinking")
    if not isinstance(sampling, dict) or not isinstance(sampling.get("temperature"), (int, float)):
        raise ValueError("Simulator config must contain sampling.temperature")
    if not isinstance(thinking, dict) or not isinstance(thinking.get("enabled"), bool):
        raise ValueError("Simulator config must contain thinking.enabled")
    if thinking["enabled"] and not isinstance(thinking.get("reasoning_effort"), str):
        raise ValueError("Enabled simulator thinking requires thinking.reasoning_effort")
    return config


def _simulator_settings(
    args: argparse.Namespace, simulator_config: dict[str, Any]
) -> tuple[str, float, bool, str | None]:
    """Resolve simulator settings, allowing explicit CLI values to override YAML."""
    model = args.user_model or simulator_config["model"]
    configured_temperature = float(simulator_config["sampling"]["temperature"])
    temperature = args.user_temperature if args.user_temperature is not None else configured_temperature
    enable_thinking = (
        args.user_enable_thinking
        if args.user_enable_thinking is not None
        else simulator_config["thinking"]["enabled"]
    )
    reasoning_effort = simulator_config["thinking"].get("reasoning_effort")
    if temperature <= 0:
        raise ValueError("Simulator temperature must be > 0")
    if enable_thinking and not reasoning_effort:
        raise ValueError("Enabled simulator thinking requires reasoning_effort")
    return model, temperature, enable_thinking, reasoning_effort


def _policy_sampling_params(args: argparse.Namespace, policy_config: dict[str, Any]) -> dict[str, Any]:
    """Build the mandated non-greedy Qwen3 profile, with explicit CLI overrides."""
    profile_name = "thinking" if args.enable_thinking else "non_thinking"
    profile = policy_config["profiles"].get(profile_name)
    if not isinstance(profile, dict) or not isinstance(profile.get("sampling"), dict):
        raise ValueError(f"Policy config is missing profiles.{profile_name}.sampling")
    if profile.get("enable_thinking") is not args.enable_thinking:
        raise ValueError(f"Policy config profiles.{profile_name}.enable_thinking must be {args.enable_thinking}")

    sampling = profile["sampling"].copy()
    required_sampling_fields = {"temperature", "top_p", "top_k", "min_p"}
    missing = required_sampling_fields.difference(sampling)
    if missing:
        raise ValueError(f"Policy config profiles.{profile_name}.sampling is missing: {', '.join(sorted(missing))}")
    for name in ("temperature", "top_p", "top_k", "min_p"):
        value = getattr(args, name)
        if value is not None:
            sampling[name] = value

    if sampling["temperature"] <= 0:
        raise ValueError("Qwen3 policy decoding must be non-greedy: temperature must be > 0")
    if not 0 < sampling["top_p"] <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if sampling["top_k"] < 1:
        raise ValueError("top_k must be >= 1")
    if not 0 <= sampling["min_p"] <= 1:
        raise ValueError("min_p must be in [0, 1]")

    configured_max_tokens = profile.get("max_new_tokens")
    if not isinstance(configured_max_tokens, int) or configured_max_tokens < 1:
        raise ValueError(f"Policy config profiles.{profile_name}.max_new_tokens must be a positive integer")
    max_tokens = args.max_new_tokens if args.max_new_tokens is not None else configured_max_tokens

    return {
        **sampling,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": max_tokens,
        # SGLang passes these to the Qwen3 chat template. This keeps native
        # reasoning separate in message.reasoning_content.
        "chat_template_kwargs": {"enable_thinking": profile["enable_thinking"]},
    }


def _repair_sampling_params(
    args: argparse.Namespace, policy_config: dict[str, Any], sampling_params: dict[str, Any]
) -> dict[str, Any]:
    """Build a separately configured budget for one format-repair completion."""
    profile_name = "thinking" if args.enable_thinking else "non_thinking"
    profile = policy_config["profiles"].get(profile_name)
    if not isinstance(profile, dict):
        raise ValueError(f"Policy config is missing profiles.{profile_name}")
    configured_budget = profile.get("repair_max_new_tokens")
    if not isinstance(configured_budget, int) or configured_budget < 1:
        raise ValueError(f"Policy config profiles.{profile_name}.repair_max_new_tokens must be a positive integer")
    budget = args.repair_max_new_tokens if args.repair_max_new_tokens is not None else configured_budget
    return {**sampling_params, "max_tokens": budget}


def _run_configuration(
    *,
    args: argparse.Namespace,
    client: SGLangClient,
    sampling_params: dict[str, Any],
    repair_sampling_params: dict[str, Any],
    policy_config_path: Path,
    simulator_config_path: Path,
    user_llm_args: dict[str, Any],
    domains: list[str],
) -> dict[str, Any]:
    """Persist all replay-relevant settings, but never secrets."""
    return _json_safe(
        {
            "trajectory_schema_version": 1,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy": {
                "api_endpoint": client.url,
                "model": client.model,
                "config_file": str(policy_config_path),
                "mode": "thinking" if args.enable_thinking else "non_thinking",
                "enable_thinking": args.enable_thinking,
                "tool_choice": "auto",
                "sampling": sampling_params,
                "repair_sampling": repair_sampling_params,
            },
            "simulator": {
                "api_endpoint": _configured_simulator_endpoint(),
                "model": args.user_model,
                "config_file": str(simulator_config_path),
                "enable_thinking": args.user_enable_thinking,
                "sampling": {
                    key: value
                    for key, value in user_llm_args.items()
                    if key not in {"api_base", "api_key", "extra_body", "reasoning_effort"}
                },
                "reasoning_effort": user_llm_args.get("reasoning_effort"),
                "extra_body": user_llm_args.get("extra_body"),
                "llm_args": user_llm_args,
            },
            "nl_assertion_judge": _configured_judge(),
            "evaluation": {
                "domains": domains,
                "task_split": args.task_split,
                "task_ids": _parse_csv(args.task_ids) if args.task_ids else None,
                "max_tasks_per_domain": args.max_tasks_per_domain,
                "max_steps": args.max_steps,
                "num_samples": args.num_samples,
                "max_concurrency": args.max_concurrency,
                "retry_statuses": _parse_csv(args.retry_statuses),
                "hf_checkpoint": args.hf_checkpoint,
            },
        }
    )


async def main_async() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _validate_args(args, parser)
    try:
        policy_config = _load_policy_config(args.policy_config)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        simulator_config = _load_simulator_config(args.simulator_config)
        simulator_model, simulator_temperature, simulator_thinking, simulator_reasoning_effort = _simulator_settings(
            args, simulator_config
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.user_model = simulator_model
    args.user_temperature = simulator_temperature
    args.user_enable_thinking = simulator_thinking

    domains = _parse_csv(args.domains)
    try:
        sampling_params = _policy_sampling_params(args, policy_config)
        repair_sampling_params = _repair_sampling_params(args, policy_config, sampling_params)
    except ValueError as exc:
        parser.error(str(exc))

    client = SGLangClient(args.sglang_url, model=args.sglang_model)
    run_log_handler: logging.Handler | None = None
    try:
        run_dir, output_path = _resolve_run_paths(args.output)
        run_dir.mkdir(parents=True, exist_ok=True)
        run_log_handler = _add_run_log_handler(run_dir)
        logger.info("Run artifacts directory: %s", run_dir)
        logger.info("Final report path: %s", output_path)
        user_llm_args = _get_user_llm_args(
            temperature=args.user_temperature,
            enable_thinking=args.user_enable_thinking,
            reasoning_effort=simulator_reasoning_effort,
        )
        run_configuration = _run_configuration(
            args=args,
            client=client,
            sampling_params=sampling_params,
            repair_sampling_params=repair_sampling_params,
            policy_config_path=args.policy_config,
            simulator_config_path=args.simulator_config,
            user_llm_args=user_llm_args,
            domains=domains,
        )
        run_configuration["artifacts"] = {
            "run_directory": str(run_dir),
            "report_file": output_path.name,
            "log_file": "evaluation.log",
            "task_results_directory": "task_results",
            "trajectories_directory": "trajectories",
        }
        all_results = await _evaluate_tasks_concurrently(
            client=client,
            output_path=output_path,
            args=args,
            domains=domains,
            sampling_params=sampling_params,
            repair_sampling_params=repair_sampling_params,
            user_llm_args=user_llm_args,
            run_configuration=run_configuration,
        )

        by_domain: dict[str, list[PassKResult]] = {}
        for r in all_results:
            by_domain.setdefault(r.domain, []).append(r)

        report = {
            "configuration": run_configuration,
            "hf_checkpoint": args.hf_checkpoint,
            "sglang_url": args.sglang_url,
            "sglang_model": args.sglang_model,
            "task_split": args.task_split,
            "domains": domains,
            "k": args.num_samples,
            "metric_note": OFFICIAL_METRIC_NOTE,
            "summary": _summarize(all_results, k=args.num_samples),
            "by_domain": {d: _summarize(rs, k=args.num_samples) for d, rs in sorted(by_domain.items())},
            "results": [asdict(r) for r in all_results],
        }

        _atomic_write_json(output_path, report)
        _mark_checkpoint_completed(output_path=output_path)

        logger.info("Wrote %d results to %s", len(all_results), output_path)
        logger.info(f"Overall: {report['summary']}")
        for d, s in report["by_domain"].items():
            logger.info(f"{d}: {s}")
    finally:
        await client.close()
        if run_log_handler is not None:
            root_logger = logging.getLogger()
            root_logger.removeHandler(run_log_handler)
            run_log_handler.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
