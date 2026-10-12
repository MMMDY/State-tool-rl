"""Configuration, hashing, and append-only JSONL helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

SENSITIVE_MARKERS = ("api_key", "authorization", "password", "secret")
SENSITIVE_NAMES = {"token", "access_token", "refresh_token", "id_token"}
JSONL_SHARD_SIZE = 100


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_int(value: str, *, bits: int = 31) -> int:
    number = int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")
    return number & ((1 << bits) - 1)


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[redacted]"
            if _is_sensitive_name(str(key))
            else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value


def _is_sensitive_name(name: str) -> bool:
    normalized = name.lower()
    return normalized in SENSITIVE_NAMES or any(
        marker in normalized for marker in SENSITIVE_MARKERS
    )


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to read config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TypeError("RFT config must be a YAML mapping")
    validate_config(raw)
    return raw


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {
        "run",
        "model",
        "tasks",
        "sampling",
        "simulator",
        "prompt",
        "verification",
        "export",
    }
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Config is missing sections: {', '.join(sorted(missing))}")
    if config["tasks"].get("split") != "train":
        raise ValueError("RFT sampling is restricted to tasks.split=train")
    sampling = config["sampling"]
    required_sampling = {
        "initial_candidates",
        "additional_candidates",
        "max_candidates",
        "success_quota",
        "temperature",
        "top_p",
        "max_response_tokens",
        "max_steps",
        "base_seed",
    }
    missing_sampling = required_sampling.difference(sampling)
    if missing_sampling:
        raise ValueError(f"sampling is missing: {', '.join(sorted(missing_sampling))}")
    initial = sampling["initial_candidates"]
    additional = sampling["additional_candidates"]
    maximum = sampling["max_candidates"]
    quota = sampling["success_quota"]
    if not all(
        isinstance(item, int) and item > 0
        for item in (initial, additional, maximum, quota)
    ):
        raise ValueError("Candidate counts and success_quota must be positive integers")
    if initial > maximum or quota > maximum:
        raise ValueError(
            "initial_candidates and success_quota cannot exceed max_candidates"
        )
    if float(sampling["temperature"]) <= 0:
        raise ValueError("sampling.temperature must be > 0")
    if not 0 < float(sampling["top_p"]) <= 1:
        raise ValueError("sampling.top_p must be in (0, 1]")
    if sampling.get("thinking") is not False:
        raise ValueError("The default RFT pipeline requires sampling.thinking=false")


def resolve_config(
    config: dict[str, Any], *, config_path: Path, repo_root: Path
) -> dict[str, Any]:
    resolved = json.loads(json.dumps(config))
    resolved["_meta"] = {
        "config_source": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "resolved_at_utc": utc_now(),
        "code_commit": git_commit(repo_root),
        "dirty_worktree": git_dirty(repo_root),
    }
    template_path = repo_root / resolved["prompt"]["template"]
    resolved["prompt"]["template"] = str(template_path.resolve())
    resolved["prompt"]["template_sha256"] = sha256_file(template_path)
    resolved["run"]["root"] = str((repo_root / resolved["run"]["root"]).resolve())
    return redact_sensitive(resolved)


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty(repo_root: Path) -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output_file:
            json.dump(
                payload, output_file, ensure_ascii=False, indent=2, sort_keys=True
            )
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_yaml_once(path: Path, payload: Any) -> None:
    serialized = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        comparable_existing = _without_resolution_time(existing)
        comparable_payload = _without_resolution_time(payload)
        if comparable_existing != comparable_payload:
            raise ValueError(
                f"Run config already exists with different content: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def _without_resolution_time(value: Any) -> Any:
    value = json.loads(json.dumps(value))
    if isinstance(value, dict) and isinstance(value.get("_meta"), dict):
        value["_meta"].pop("resolved_at_utc", None)
        value["_meta"].pop("dirty_worktree", None)
    return value


def jsonl_input_paths(path: Path) -> list[Path]:
    """Return a logical JSONL input and its numbered shards, if present."""

    shards = sorted(path.parent.glob(f"{path.stem}.part-*{path.suffix}"))
    if path.exists() and shards:
        raise ValueError(f"Both unsplit and sharded JSONL files exist for {path}")
    if shards:
        return shards
    return [path] if path.exists() else []


def _iter_jsonl_file(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TypeError(f"JSONL row at {path}:{line_number} must be an object")
            yield row


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for input_path in jsonl_input_paths(path):
        yield from _iter_jsonl_file(input_path)


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(
                json.dumps(
                    redact_sensitive(row), ensure_ascii=False, separators=(",", ":")
                )
            )
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
            count += 1
    return count


def _next_shard_path(path: Path, paths: list[Path]) -> Path:
    indexes = []
    prefix = f"{path.stem}.part-"
    for candidate in paths:
        name = candidate.name
        if name.startswith(prefix) and name.endswith(path.suffix):
            index = name[len(prefix) : -len(path.suffix)]
            if index.isdigit():
                indexes.append(int(index))
    next_index = max(indexes, default=-1) + 1
    return path.with_name(f"{path.stem}.part-{next_index:03d}{path.suffix}")


def _split_jsonl_file(path: Path, *, shard_size: int) -> None:
    """Migrate a legacy unsplit JSONL file without leaving partial shards."""

    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{path.stem}.", dir=path.parent))
    temporary_paths: list[Path] = []
    moved_paths: list[Path] = []
    try:
        output_file = None
        row_count = 0
        try:
            with path.open(encoding="utf-8") as input_file:
                for line in input_file:
                    if not line.strip():
                        continue
                    if row_count % shard_size == 0:
                        if output_file is not None:
                            output_file.flush()
                            os.fsync(output_file.fileno())
                            output_file.close()
                        temporary_path = temporary_dir / (
                            f"part-{len(temporary_paths):03d}{path.suffix}"
                        )
                        temporary_paths.append(temporary_path)
                        output_file = temporary_path.open("w", encoding="utf-8")
                    output_file.write(line if line.endswith("\n") else f"{line}\n")
                    row_count += 1
        finally:
            if output_file is not None and not output_file.closed:
                output_file.flush()
                os.fsync(output_file.fileno())
                output_file.close()
        for temporary_path in temporary_paths:
            final_path = path.with_name(
                f"{path.stem}.part-{len(moved_paths):03d}{path.suffix}"
            )
            os.replace(temporary_path, final_path)
            moved_paths.append(final_path)
        path.unlink()
    except BaseException:
        for moved_path in moved_paths:
            moved_path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


def append_jsonl_sharded(
    path: Path,
    rows: Iterable[dict[str, Any]],
    *,
    shard_size: int = JSONL_SHARD_SIZE,
) -> int:
    """Append rows to numbered JSONL shards, starting a new file every N rows."""

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    existing_shards = sorted(path.parent.glob(f"{path.stem}.part-*{path.suffix}"))
    if path.exists():
        if existing_shards:
            raise ValueError(f"Both unsplit and sharded JSONL files exist for {path}")
        _split_jsonl_file(path, shard_size=shard_size)
    pending = list(rows)
    written = 0
    while pending:
        paths = sorted(path.parent.glob(f"{path.stem}.part-*{path.suffix}"))
        target = paths[-1] if paths else _next_shard_path(path, paths)
        current_count = sum(1 for _ in _iter_jsonl_file(target)) if target.exists() else 0
        if current_count >= shard_size:
            target = _next_shard_path(path, paths)
            current_count = 0
        available = shard_size - current_count
        batch = pending[:available]
        written += append_jsonl(target, batch)
        del pending[: len(batch)]
    return written


def existing_ids(path: Path, key: str = "trajectory_id") -> set[str]:
    return {str(row[key]) for row in iter_jsonl(path) if row.get(key) is not None}


def stage_is_cached(
    meta_path: Path, *, input_hashes: dict[str, str], config_hash: str
) -> bool:
    if not meta_path.exists():
        return False
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("input_hashes") == input_hashes
        and metadata.get("config_hash") == config_hash
    )


def write_stage_metadata(
    meta_path: Path,
    *,
    input_hashes: dict[str, str],
    config_hash: str,
    counts: dict[str, int],
) -> None:
    atomic_write_json(
        meta_path,
        {
            "completed_at_utc": utc_now(),
            "input_hashes": input_hashes,
            "config_hash": config_hash,
            "counts": counts,
        },
    )
