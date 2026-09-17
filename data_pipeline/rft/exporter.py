"""Deduplicate, stratify, quality-check, and export verified RFT trajectories."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from data_pipeline.rft.io import (
    atomic_write_json,
    existing_ids,
    sha256_file,
    sha256_json,
    utc_now,
)
from data_pipeline.rft.normalize import normalized_messages, trajectory_length

DIFFICULTIES = ("hard", "medium", "easy")
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_INTERNAL_PATH_RE = re.compile(r"(?:/root/|/home/|/workspace/|[A-Za-z]:\\\\)")
_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|password|secret|access[_-]?token)\s*[:=]\s*(?!\[redacted\])"
)


def difficulty_from_rate(rate: float) -> str:
    if rate < 0.20:
        return "hard"
    if rate <= 0.80:
        return "medium"
    return "easy"


def task_statistics(
    raw_rows: list[dict[str, Any]],
    verified_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    verified_ids = {str(row["trajectory_id"]) for row in verified_rows}
    all_checked = {
        str(row["trajectory_id"]): row for row in [*verified_rows, *rejected_rows]
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(str(row["domain"]), str(row["task_id"]))].append(row)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        first = [
            row
            for row in rows
            if int(row.get("sampling", {}).get("rollout_index", 10**9)) < 16
        ]
        first_pass = sum(str(row["trajectory_id"]) in verified_ids for row in first)
        rate = first_pass / len(first) if first else 0.0
        process_pass = sum(
            bool(
                all_checked.get(str(row["trajectory_id"]), {})
                .get("verifier", {})
                .get("process", {})
                .get("passed")
            )
            for row in rows
        )
        result[key] = {
            "domain": key[0],
            "task_id": key[1],
            "n_rollout": len(rows),
            "n_first_batch": len(first),
            "n_terminal_success": sum(
                bool(row.get("terminal", {}).get("success")) for row in rows
            ),
            "n_process_pass": process_pass,
            "n_both_pass": sum(
                str(row["trajectory_id"]) in verified_ids for row in rows
            ),
            "success_rate_first_16": rate,
            "difficulty": difficulty_from_rate(rate),
        }
    return result


def _candidate_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    quality = row.get("quality") or {}
    return (
        -float(quality.get("official_score", 0.0)),
        -int(quality.get("verifier_passes", 0)),
        int(quality.get("trajectory_chars", trajectory_length(row))),
        str(row.get("trajectory_id")),
    )


def deduplicate_and_cap(
    verified_rows: list[dict[str, Any]],
    stats: dict[tuple[str, str], dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in verified_rows:
        grouped[(str(row["domain"]), str(row["task_id"]))].append(row)
    selected: list[dict[str, Any]] = []
    before = len(verified_rows)
    fingerprint_removed = 0
    cap_removed = 0
    for key, rows in sorted(grouped.items()):
        unique_action: dict[str, dict[str, Any]] = {}
        for row in sorted(rows, key=_candidate_rank):
            fingerprint = row.get("fingerprints", {}).get("action_fingerprint")
            if fingerprint not in unique_action:
                unique_action[str(fingerprint)] = row
            else:
                fingerprint_removed += 1
        unique_semantic_state: dict[tuple[str, str], dict[str, Any]] = {}
        for row in sorted(unique_action.values(), key=_candidate_rank):
            fp = row.get("fingerprints") or {}
            composite = (
                str(fp.get("state_fingerprint")),
                str(fp.get("semantic_fingerprint")),
            )
            if composite not in unique_semantic_state:
                unique_semantic_state[composite] = row
            else:
                fingerprint_removed += 1
        difficulty = stats[key]["difficulty"]
        cap = int(config["export"]["max_per_task"][difficulty])
        kept = sorted(unique_semantic_state.values(), key=_candidate_rank)[:cap]
        cap_removed += max(0, len(unique_semantic_state) - len(kept))
        for row in kept:
            row = dict(row)
            row["difficulty"] = difficulty
            row["task_stats"] = stats[key]
            selected.append(row)
    return selected, {
        "verified_before_dedupe": before,
        "fingerprint_duplicates_removed": fingerprint_removed,
        "per_task_cap_removed": cap_removed,
        "selected": len(selected),
    }


def export_row(row: dict[str, Any]) -> dict[str, Any]:
    messages = normalized_messages(list(row.get("messages") or []))
    return {
        "task_id": row["task_id"],
        "trajectory_id": row["trajectory_id"],
        "domain": row["domain"],
        "split": "train",
        "prompt": messages,
        "messages": messages,
        "tools": row.get("tools", {}).get("schema") or [],
        "schema_hash": row.get("tools", {}).get("schema_hash"),
        "quality": {
            **(row.get("quality") or {}),
            "difficulty": row["difficulty"],
            "fingerprints": row.get("fingerprints") or {},
            "loss_mask_roles": ["assistant"],
        },
        "source_run_id": row["run_id"],
    }


def recovery_row(row: dict[str, Any], difficulty: str) -> dict[str, Any]:
    steps = row.get("steps") or []
    failure_index = len(steps) - 1 if steps else 0
    for index, step in enumerate(steps):
        observation = str(step.get("tool_observation") or "")
        if (
            step.get("flags")
            or "error" in observation.lower()
            or "invalid" in observation.lower()
        ):
            failure_index = index
            break
    failure = steps[failure_index] if steps else {}
    return {
        "task_id": row["task_id"],
        "trajectory_id": row["trajectory_id"],
        "domain": row["domain"],
        "split": "train",
        "difficulty": difficulty,
        "failed_prefix": steps[: failure_index + 1],
        "error_action": failure.get("tool_call") or failure.get("assistant_raw_text"),
        "error_observation": failure.get("tool_observation"),
        "recovery_context": normalized_messages(list(row.get("messages") or [])),
        "terminal": row.get("terminal") or {},
        "source_run_id": row["run_id"],
        "training_eligible": False,
    }


def validate_no_split_leakage(rows: Iterable[dict[str, Any]]) -> None:
    from tau2.registry import registry

    rows = list(rows)
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Export contains a non-train trajectory")
    domains = sorted({str(row["domain"]) for row in rows})
    held_out: set[tuple[str, str]] = set()
    for domain in domains:
        loader = registry.get_tasks_loader(domain)
        for split in ("test", "dev"):
            try:
                held_out_tasks = loader(split)
            except (FileNotFoundError, KeyError, ValueError):
                held_out_tasks = []
            held_out.update((domain, str(task.id)) for task in held_out_tasks)
    overlap = {(str(row["domain"]), str(row["task_id"])) for row in rows}.intersection(
        held_out
    )
    if overlap:
        raise ValueError(
            f"Export task leakage into held-out split: {sorted(overlap)[:10]}"
        )


def _percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _privacy_findings(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    findings = Counter()
    for row in rows:
        text = json.dumps(
            row.get("messages") or row.get("prompt") or [], ensure_ascii=False
        )
        findings["email_like_values"] += len(_EMAIL_RE.findall(text))
        findings["internal_paths"] += len(_INTERNAL_PATH_RE.findall(text))
        findings["unredacted_secret_patterns"] += len(_SECRET_RE.findall(text))
    return dict(findings)


def build_summary(
    *,
    raw_rows: list[dict[str, Any]],
    verified_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    exported_rows: list[dict[str, Any]],
    stats: dict[tuple[str, str], dict[str, Any]],
    dedupe_counts: dict[str, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    difficulty_counts = Counter(
        row.get("quality", {}).get("difficulty") for row in exported_rows
    )
    domain_counts = Counter(str(row.get("domain")) for row in exported_rows)
    rejection_flags = Counter(
        flag
        for row in rejected_rows
        for flag in row.get("verifier", {}).get("flags", [])
    )
    lengths = [
        sum(
            len(json.dumps(message, ensure_ascii=False))
            for message in row.get("messages") or []
        )
        for row in exported_rows
    ]
    tool_counts = [
        sum(
            len(message.get("tool_calls") or [])
            for message in row.get("messages") or []
        )
        for row in exported_rows
    ]
    semantic = [
        row.get("quality", {}).get("fingerprints", {}).get("semantic_fingerprint")
        for row in exported_rows
    ]
    duplicate_rate = 1 - len(set(semantic)) / len(semantic) if semantic else 0.0
    total_exported = len(exported_rows)
    actual_mix = {
        level: difficulty_counts[level] / total_exported if total_exported else 0.0
        for level in DIFFICULTIES
    }
    thresholds = config["export"]["warning_thresholds"]
    replay_success_rate = (
        sum(
            bool(row.get("verifier", {}).get("replay", {}).get("passed"))
            for row in verified_rows
        )
        / len(verified_rows)
        if verified_rows
        else 0.0
    )
    warnings: list[str] = []
    if replay_success_rate < float(thresholds["replay_success_rate"]):
        warnings.append("verified replay success rate is below threshold")
    if duplicate_rate >= float(thresholds["duplicate_rate"]):
        warnings.append("semantic duplicate rate is above threshold")
    if actual_mix["easy"] > float(thresholds["easy_share"]):
        warnings.append("easy-task share is above threshold")
    return {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "counts": {
            "raw": len(raw_rows),
            "terminal_success": sum(
                bool(row.get("terminal", {}).get("success")) for row in raw_rows
            ),
            "verified": len(verified_rows),
            "rejected": len(rejected_rows),
            "exported": total_exported,
            **dedupe_counts,
        },
        "by_domain": dict(sorted(domain_counts.items())),
        "by_difficulty": dict(sorted(difficulty_counts.items())),
        "target_mix": config["export"]["target_mix"],
        "actual_mix": actual_mix,
        "task_statistics": sorted(
            stats.values(), key=lambda item: (item["domain"], item["task_id"])
        ),
        "rejection_reasons": dict(rejection_flags.most_common()),
        "trajectory_chars": {
            "mean": statistics.fmean(lengths) if lengths else 0.0,
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
        },
        "tool_calls": {
            "mean": statistics.fmean(tool_counts) if tool_counts else 0.0,
            "p50": _percentile(tool_counts, 0.50),
            "p90": _percentile(tool_counts, 0.90),
        },
        "terminal_success_rate": (
            sum(bool(row.get("terminal", {}).get("success")) for row in raw_rows)
            / len(raw_rows)
            if raw_rows
            else 0.0
        ),
        "verified_replay_success_rate": replay_success_rate,
        "semantic_duplicate_rate": duplicate_rate,
        "privacy_findings": _privacy_findings(exported_rows),
        "warnings": warnings,
    }


def sample_review_rows(
    exported_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    *,
    per_domain: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in exported_rows:
        by_domain[str(row["domain"])].append(row)
    for domain, rows in sorted(by_domain.items()):
        rows.sort(key=lambda row: sha256_json(row["trajectory_id"]))
        for row in rows[:per_domain]:
            selected.append(
                {
                    "review_id": f"accepted/{row['trajectory_id']}",
                    "trajectory_id": row["trajectory_id"],
                    "domain": domain,
                    "difficulty": row.get("quality", {}).get("difficulty"),
                    "automatic_decision": "accepted",
                    "human_decision": None,
                    "notes": None,
                }
            )
    covered: set[tuple[str, str]] = set()
    for row in sorted(rejected_rows, key=lambda item: str(item.get("trajectory_id"))):
        for flag in row.get("verifier", {}).get("flags", []):
            key = (str(row.get("domain")), str(flag))
            if key in covered:
                continue
            covered.add(key)
            selected.append(
                {
                    "review_id": f"rejected/{row['trajectory_id']}/{flag}",
                    "trajectory_id": row["trajectory_id"],
                    "domain": row.get("domain"),
                    "difficulty": None,
                    "automatic_decision": "rejected",
                    "primary_flag": flag,
                    "human_decision": None,
                    "notes": None,
                }
            )
    return selected


def append_unique(path: Path, rows: Iterable[dict[str, Any]], *, key: str) -> int:
    from data_pipeline.rft.io import append_jsonl

    known = existing_ids(path, key=key)
    new_rows = [row for row in rows if str(row[key]) not in known]
    return append_jsonl(path, new_rows)


def update_manifest(manifest_path: Path, artifacts: dict[str, Path]) -> None:
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    manifest["updated_at_utc"] = utc_now()
    manifest["artifacts"] = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in artifacts.items()
        if path.exists()
    }
    atomic_write_json(manifest_path, manifest)
