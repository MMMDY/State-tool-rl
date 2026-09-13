#!/usr/bin/env python3
"""Pass@K evaluation for tau2-bench using an SGLang-served policy."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from tau2_rl_pipeline.actions import env_action_from_parsed_action, followup_messages_for_observation, parse_action
from tau2_rl_pipeline.env import compute_partial_score_from_reward_info, parse_reward_info
from tau2_rl_pipeline.prompting import build_tau2_agent_system_prompt

logger = logging.getLogger(__name__)

DEFAULT_DOMAINS = ("airline", "retail", "telecom")
PASS_AT_K_NOTE = "pass@k = any success among k attempts (not pass^k leaderboard estimate)"
_SENSITIVE_FIELD_MARKERS = ("api_key", "authorization", "password", "secret")
_SENSITIVE_FIELD_NAMES = {"token", "access_token", "refresh_token", "id_token"}


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _get_user_llm_args(*, temperature: float) -> dict[str, Any]:
    args: dict[str, Any] = {"temperature": temperature}
    api_base = os.environ.get("TAU2_USER_API_BASE", "").strip()
    if api_base:
        args["api_base"] = api_base
        args["api_key"] = "dummy-key-for-local-server"
        args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
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


def _trace_event(trajectory: list[dict[str, Any]], event: str, **data: Any) -> None:
    trajectory.append({"event": event, **_json_safe(data)})


def _configured_simulator_endpoint() -> str | None:
    for name in ("TAU2_USER_API_BASE", "OPENAI_API_BASE", "OPENAI_BASE_URL"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


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
    trajectory: list[dict[str, Any]] | None = None


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
    attempts: list[dict[str, Any]]
    pass_at_1: float
    pass_at_k: float


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


def _action_from_chat_completion(response: dict[str, Any]) -> tuple[str, str | None]:
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
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments.strip() else {}
        if not isinstance(arguments, dict):
            raise ValueError("Structured tool-call arguments must be an object")
        return _canonical_tool_call(name, arguments), reasoning

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        content = content.strip()
        # Compatibility with a server response that did not run its tool parser.
        # The structured path above remains the normal path for this evaluator.
        if content.startswith("<tool_call>") and content.endswith("</tool_call>"):
            return content, reasoning
        return _canonical_tool_call("respond", {"content": content}), reasoning
    raise ValueError("Chat Completions response has neither tool_calls nor content")


def _load_tasks(domain: str, task_split: str) -> list[str]:
    from tau2.registry import registry

    return [t.id for t in registry.get_tasks_loader(domain)(task_split)]


async def _run_one_attempt(
    *,
    client: SGLangClient,
    domain: str,
    task_id: str,
    sampling_params: dict[str, Any],
    max_steps: int,
    user_llm: str,
    user_llm_args: dict[str, Any],
) -> AttemptResult:
    from tau2.gym.gym_agent import AgentGymEnv

    trajectory: list[dict[str, Any]] = []
    env = AgentGymEnv(
        domain=domain,
        task_id=task_id,
        max_steps=max_steps,
        solo_mode=False,
        user_llm=user_llm,
        user_llm_args=user_llm_args,
        all_messages_as_observation=False,
    )

    observation, info = env.reset()
    tools = info.get("tools", [])
    tools_openai = [t if isinstance(t, dict) else t.openai_schema for t in tools]
    policy = info.get("policy", "")
    _trace_event(
        trajectory,
        "environment_reset",
        observation=observation,
        policy=policy,
        tools=tools_openai,
    )

    system_prompt = build_tau2_agent_system_prompt(
        domain=domain,
        policy=policy,
        tools_openai=tools_openai,
        include_tool_schema=False,
        use_structured_tool_calls=True,
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(
        followup_messages_for_observation(
            observation=observation,
            last_action_call="(reset)",
            last_action_was_tool=False,
        )
    )

    reward = 0.0
    reward_info: dict[str, Any] = {}

    for step in range(max_steps):
        _trace_event(
            trajectory,
            "policy_request",
            step=step,
            phase="initial",
            model=client.model,
            endpoint=client.url,
            messages=messages,
            tools=tools_openai,
            tool_choice="auto",
            sampling_params=sampling_params,
        )
        out = await client.chat_completion(messages=messages, tools=tools_openai, sampling_params=sampling_params)
        choice = (out.get("choices") or [{}])[0]
        _trace_event(
            trajectory,
            "policy_response",
            step=step,
            phase="initial",
            finish_reason=choice.get("finish_reason"),
            response=out,
        )
        if choice.get("finish_reason") == "abort":
            _trace_event(trajectory, "attempt_end", status="aborted", reason="sglang_abort")
            return AttemptResult(
                success=False,
                reward=0.0,
                partial_score=0.0,
                partial_components={},
                steps=step,
                status="aborted",
                error="sglang_abort",
                trajectory=trajectory,
            )

        try:
            assistant_text, reasoning_content = _action_from_chat_completion(out)
            if reasoning_content:
                logger.debug("step=%d received %d reasoning characters", step, len(reasoning_content))
            parsed = parse_action(assistant_text)
            _trace_event(
                trajectory,
                "parsed_policy_action",
                step=step,
                phase="initial",
                canonical_action=assistant_text,
                function_name=parsed.name,
                arguments=parsed.arguments,
                reasoning_content=reasoning_content,
            )
        except Exception as exc:
            _trace_event(trajectory, "policy_parse_error", step=step, phase="initial", error=str(exc))
            messages.append(
                {
                    "role": "user",
                    "content": "FORMAT ERROR. Make exactly one valid tool call with JSON object arguments, "
                    "or provide a customer-facing response.",
                }
            )
            repair_params = {**sampling_params, "temperature": 0.0}
            _trace_event(
                trajectory,
                "policy_request",
                step=step,
                phase="repair",
                model=client.model,
                endpoint=client.url,
                messages=messages,
                tools=tools_openai,
                tool_choice="auto",
                sampling_params=repair_params,
            )
            out = await client.chat_completion(messages=messages, tools=tools_openai, sampling_params=repair_params)
            choice = (out.get("choices") or [{}])[0]
            _trace_event(
                trajectory,
                "policy_response",
                step=step,
                phase="repair",
                finish_reason=choice.get("finish_reason"),
                response=out,
            )
            try:
                assistant_text, reasoning_content = _action_from_chat_completion(out)
                if reasoning_content:
                    logger.debug("step=%d repair received %d reasoning characters", step, len(reasoning_content))
                parsed = parse_action(assistant_text)
                _trace_event(
                    trajectory,
                    "parsed_policy_action",
                    step=step,
                    phase="repair",
                    canonical_action=assistant_text,
                    function_name=parsed.name,
                    arguments=parsed.arguments,
                    reasoning_content=reasoning_content,
                )
            except Exception as repair_exc:
                partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)
                _trace_event(
                    trajectory,
                    "attempt_end",
                    status="parse_error",
                    error=f"{exc}; repair failed: {repair_exc}",
                )
                return AttemptResult(
                    success=False,
                    reward=float(reward),
                    partial_score=partial_score,
                    partial_components=partial_components,
                    steps=step + 1,
                    status="parse_error",
                    error=f"{exc}; repair failed: {repair_exc}",
                    reward_info=reward_info,
                    trajectory=trajectory,
                )

        messages.append({"role": "assistant", "content": assistant_text})

        env_action = env_action_from_parsed_action(parsed)
        observation, reward, terminated, truncated, info = env.step(env_action)
        _trace_event(
            trajectory,
            "environment_step",
            step=step,
            env_action=env_action,
            observation=observation,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        if terminated:
            reward_info = parse_reward_info(info)
            partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)
            _trace_event(
                trajectory,
                "attempt_end",
                status="completed",
                success=float(reward) >= 1.0,
                reward=float(reward),
                reward_info=reward_info,
            )
            return AttemptResult(
                success=float(reward) >= 1.0,
                reward=float(reward),
                partial_score=partial_score,
                partial_components=partial_components,
                steps=step + 1,
                status="completed",
                reward_info=reward_info,
                trajectory=trajectory,
            )

        messages.extend(
            followup_messages_for_observation(
                observation=observation,
                last_action_call=parsed.raw_action_call,
                last_action_was_tool=(parsed.name != "respond"),
            )
        )

    partial_score, partial_components = compute_partial_score_from_reward_info(reward_info)
    _trace_event(
        trajectory,
        "attempt_end",
        status="truncated",
        success=False,
        reward=float(reward),
        reward_info=reward_info,
    )
    return AttemptResult(
        success=False,
        reward=float(reward),
        partial_score=partial_score,
        partial_components=partial_components,
        steps=max_steps,
        status="truncated",
        reward_info=reward_info,
        trajectory=trajectory,
    )


def _summarize(results: list[PassKResult], *, k: int) -> dict[str, Any]:
    total = len(results)
    pass1 = sum(r.pass_at_1 for r in results) / total if total else 0.0
    passk = sum(r.pass_at_k for r in results) / total if total else 0.0
    return {"total": total, "pass_at_1": pass1, f"pass_at_{k}": passk}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate tau2-bench with Pass@K sampling")
    parser.add_argument("--hf-checkpoint", help="Deprecated compatibility field; SGLang loads the model server-side.")
    parser.add_argument("--sglang-url", required=True)
    parser.add_argument("--sglang-model", default="qwen3-4b", help="Model name exposed by SGLang Chat Completions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--task-split", default="test", choices=("train", "test", "base"))
    parser.add_argument("--max-tasks-per-domain", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("TAU2_MAX_STEPS", "100")))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--user-model", default=os.environ.get("TAU2_USER_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--user-temperature", type=float, default=float(os.environ.get("TAU2_USER_TEMPERATURE", "0.7")))
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1")


def _run_configuration(
    *,
    args: argparse.Namespace,
    client: SGLangClient,
    sampling_params: dict[str, Any],
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
                "tool_choice": "auto",
                "sampling": sampling_params,
            },
            "simulator": {
                "api_endpoint": _configured_simulator_endpoint(),
                "model": args.user_model,
                "llm_args": user_llm_args,
            },
            "evaluation": {
                "domains": domains,
                "task_split": args.task_split,
                "max_tasks_per_domain": args.max_tasks_per_domain,
                "max_steps": args.max_steps,
                "num_samples": args.num_samples,
                "hf_checkpoint": args.hf_checkpoint,
            },
        }
    )


async def main_async() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    domains = _parse_csv(args.domains)
    sampling_params: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.max_new_tokens,
    }
    if args.top_k > 0:
        sampling_params["top_k"] = args.top_k

    client = SGLangClient(args.sglang_url, model=args.sglang_model)
    try:
        user_llm_args = _get_user_llm_args(temperature=args.user_temperature)
        run_configuration = _run_configuration(
            args=args,
            client=client,
            sampling_params=sampling_params,
            user_llm_args=user_llm_args,
            domains=domains,
        )
        all_results: list[PassKResult] = []

        for domain in domains:
            task_ids = _load_tasks(domain, args.task_split)
            if args.max_tasks_per_domain is not None:
                task_ids = task_ids[: args.max_tasks_per_domain]

            logger.info(f"Evaluating domain={domain} split={args.task_split} tasks={len(task_ids)} k={args.num_samples}")
            for i, task_id in enumerate(task_ids):
                attempts: list[AttemptResult] = []
                for _ in range(args.num_samples):
                    attempts.append(
                        await _run_one_attempt(
                            client=client,
                            domain=domain,
                            task_id=task_id,
                            sampling_params=sampling_params,
                            max_steps=args.max_steps,
                            user_llm=args.user_model,
                            user_llm_args=user_llm_args,
                        )
                    )

                best_idx = 0
                for j in range(1, len(attempts)):
                    a = attempts[j]
                    b = attempts[best_idx]
                    if a.success and not b.success:
                        best_idx = j
                    elif a.success == b.success and a.partial_score > b.partial_score:
                        best_idx = j

                pass_at_1 = 1.0 if attempts and attempts[0].success else 0.0
                pass_at_k = 1.0 if any(a.success for a in attempts) else 0.0
                best = attempts[best_idx]
                all_results.append(
                    PassKResult(
                        domain=domain,
                        task_split=args.task_split,
                        task_index=i,
                        task_id=task_id,
                        num_samples=args.num_samples,
                        best_success=best.success,
                        best_reward=best.reward,
                        best_partial_score=best.partial_score,
                        best_sample_idx=best_idx,
                        attempts=[asdict(a) for a in attempts],
                        pass_at_1=pass_at_1,
                        pass_at_k=pass_at_k,
                    )
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
            "metric_note": PASS_AT_K_NOTE,
            "summary": _summarize(all_results, k=args.num_samples),
            "by_domain": {d: _summarize(rs, k=args.num_samples) for d, rs in sorted(by_domain.items())},
            "results": [asdict(r) for r in all_results],
        }

        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Wrote {len(all_results)} results to {args.output}")
        logger.info(f"Overall: {report['summary']}")
        for d, s in report["by_domain"].items():
            logger.info(f"{d}: {s}")
    finally:
        await client.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
