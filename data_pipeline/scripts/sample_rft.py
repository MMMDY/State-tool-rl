#!/usr/bin/env python3
"""Sample adaptive RFT rollouts without running verification or export."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.rft.io import load_config, resolve_config
from data_pipeline.rft.sampler import (
    default_run_id,
    initialize_run,
    load_task_specs,
    run_sampling,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample Qwen3-4B trajectories for RFT")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_config = load_config(args.config)
    config = resolve_config(source_config, config_path=args.config, repo_root=REPO_ROOT)
    run_id = args.run_id or default_run_id(config, REPO_ROOT)
    specs = load_task_specs(config)
    run_root = initialize_run(
        repo_root=REPO_ROOT, run_id=run_id, resolved_config=config, specs=specs
    )
    counts = asyncio.run(
        run_sampling(repo_root=REPO_ROOT, run_id=run_id, config=config, specs=specs)
    )
    print(
        f"run={run_root} tasks={counts['tasks']} new_rollouts={counts['new_rollouts']} reused={counts['reused_rollouts']}"
    )


if __name__ == "__main__":
    main()
