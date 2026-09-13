"""Reproducible, secret-safe run metadata logging."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SECRET_WORDS = ("key", "token", "secret", "password", "authorization", "cookie", "header")
_SAFE_EXACT_ENV = {
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
    "TAU2_DATA_DIR",
    "TAU2_MAX_STEPS",
    "TAU2_REWARD_ALPHA",
    "TAU2_ROOT",
    "TAU2_USER_API_BASE",
    "TAU2_USER_MODEL",
}


def _is_secret_name(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in _SECRET_WORDS)


def _safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if not _is_secret_name(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_environment() -> dict[str, str]:
    result: dict[str, str] = {}
    for name, value in os.environ.items():
        if _is_secret_name(name):
            continue
        if name in _SAFE_EXACT_ENV or name.startswith(("TAU2_", "CUDA_", "RAY_")):
            result[name] = value
    return result


def _package_versions() -> dict[str, str | None]:
    names = ("tau2", "ray", "slime", "megatron-core", "torch", "sglang", "litellm", "openai")
    installed = {
        distribution.metadata.get("Name", "").lower(): distribution.version
        for distribution in importlib.metadata.distributions()
    }
    return {name: installed.get(name.lower()) for name in names}


def write_versions(run_dir: str | os.PathLike[str], config: Mapping[str, Any]) -> dict[str, Path]:
    """Write versions, resolved config, and a filtered environment allowlist.

    Values under keys containing credential-like words are omitted recursively.
    The returned paths make it easy for callers to include the files in an
    experiment manifest without knowing the storage layout.
    """

    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    versions_path = directory / "versions.json"
    config_path = directory / "resolved_config.json"
    environment_path = directory / "environment.json"

    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    versions_path.write_text(json.dumps(versions, indent=2, ensure_ascii=False) + "\n")
    config_path.write_text(json.dumps(_safe_value(config), indent=2, ensure_ascii=False) + "\n")
    environment_path.write_text(json.dumps(_safe_environment(), indent=2, ensure_ascii=False) + "\n")
    return {
        "versions": versions_path,
        "config": config_path,
        "environment": environment_path,
    }

