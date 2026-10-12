#!/usr/bin/env python3
"""Strictly verify raw RFT trajectories and partition accepted/rejected rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.rft.io import (
    append_jsonl,
    existing_ids,
    iter_jsonl,
    jsonl_input_paths,
    sha256_file,
    sha256_json,
    stage_is_cached,
    write_stage_metadata,
)
from data_pipeline.rft.verifier import verify_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify raw RFT trajectories with strict tau2 replay"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Resolved config; inferred from the run directory",
    )
    return parser


def _config_path(args: argparse.Namespace) -> Path:
    if args.config:
        return args.config
    candidate = args.input.parent.parent / "config.resolved.yaml"
    if not candidate.exists():
        raise ValueError("Unable to infer config.resolved.yaml; pass --config")
    return candidate


def main() -> None:
    args = build_parser().parse_args()
    config_path = _config_path(args)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rejected_path = args.rejected or args.output.with_name("rejected.jsonl")
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    input_hashes = {
        str(path.resolve()): sha256_file(path)
        for path in jsonl_input_paths(args.input)
    }
    config_hash = sha256_json(config)
    if stage_is_cached(meta_path, input_hashes=input_hashes, config_hash=config_hash):
        print("verify_rft: cached input/config; nothing to do")
        return
    processed = existing_ids(args.output) | existing_ids(rejected_path)
    accepted_new = 0
    rejected_new = 0
    for row in iter_jsonl(args.input):
        trajectory_id = str(row.get("trajectory_id"))
        if trajectory_id in processed:
            continue
        verified = verify_record(row, config)
        destination = args.output if verified["status"] == "verified" else rejected_path
        append_jsonl(destination, [verified])
        if verified["status"] == "verified":
            accepted_new += 1
        else:
            rejected_new += 1
    counts = {
        "accepted_new": accepted_new,
        "rejected_new": rejected_new,
        "processed_total": len(processed) + accepted_new + rejected_new,
    }
    write_stage_metadata(
        meta_path, input_hashes=input_hashes, config_hash=config_hash, counts=counts
    )
    print(
        f"accepted_new={counts['accepted_new']} rejected_new={counts['rejected_new']} "
        f"processed_total={counts['processed_total']}"
    )


if __name__ == "__main__":
    main()
