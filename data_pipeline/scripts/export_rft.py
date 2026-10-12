#!/usr/bin/env python3
"""Export verified RFT samples, recovery candidates, manifest, and QA reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_pipeline.rft.exporter import (
    append_unique,
    build_summary,
    deduplicate_and_cap,
    export_row,
    recovery_row,
    sample_review_rows,
    task_statistics,
    update_manifest,
    validate_no_split_leakage,
)
from data_pipeline.rft.io import (
    atomic_write_json,
    iter_jsonl,
    sha256_file,
    sha256_json,
    stage_is_cached,
    write_stage_metadata,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export verified RFT trajectories to SFT JSONL"
    )
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--rejected", type=Path, default=None)
    parser.add_argument("--recovery-output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = args.verified.parent.parent
    raw_path = args.raw or run_root / "rollouts" / "raw.jsonl"
    rejected_path = args.rejected or run_root / "rollouts" / "rejected.jsonl"
    recovery_path = args.recovery_output or args.output.with_name(
        "rft_recovery_candidates.jsonl"
    )
    report_path = args.report or run_root / "reports" / "summary.json"
    review_path = run_root / "reports" / "sample_review.jsonl"
    config_path = args.config or run_root / "config.resolved.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_paths = [raw_path, args.verified, rejected_path]
    input_hashes = {
        str(path.resolve()): sha256_file(path) for path in input_paths if path.exists()
    }
    config_hash = sha256_json(config)
    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    if stage_is_cached(meta_path, input_hashes=input_hashes, config_hash=config_hash):
        print("export_rft: cached input/config; nothing to do")
        return

    raw_rows = list(iter_jsonl(raw_path))
    verified_rows = list(iter_jsonl(args.verified))
    rejected_rows = list(iter_jsonl(rejected_path))
    if any(not row.get("verifier", {}).get("passed") for row in verified_rows):
        raise ValueError("--verified contains a row that did not pass all verifiers")
    validate_no_split_leakage(verified_rows)
    stats = task_statistics(raw_rows, verified_rows, rejected_rows)
    selected, dedupe_counts = deduplicate_and_cap(verified_rows, stats, config)
    existing_exported = list(iter_jsonl(args.output))
    existing_ids = {str(row["trajectory_id"]) for row in existing_exported}
    existing_per_task: dict[tuple[str, str], int] = {}
    for row in existing_exported:
        key = (str(row["domain"]), str(row["task_id"]))
        existing_per_task[key] = existing_per_task.get(key, 0) + 1
    appendable = []
    for row in selected:
        if str(row["trajectory_id"]) in existing_ids:
            continue
        key = (str(row["domain"]), str(row["task_id"]))
        difficulty = stats[key]["difficulty"]
        cap = int(config["export"]["max_per_task"][difficulty])
        if existing_per_task.get(key, 0) >= cap:
            continue
        appendable.append(row)
        existing_per_task[key] = existing_per_task.get(key, 0) + 1
    sft_rows = [export_row(row) for row in appendable]
    append_unique(args.output, sft_rows, key="trajectory_id")

    recovery_rows = []
    verified_ids = {str(row["trajectory_id"]) for row in verified_rows}
    for row in raw_rows:
        key = (str(row["domain"]), str(row["task_id"]))
        task_stats = stats[key]
        if (
            task_stats["difficulty"] == "hard"
            and str(row["trajectory_id"]) not in verified_ids
        ):
            recovery_rows.append(recovery_row(row, "hard"))
    append_unique(recovery_path, recovery_rows, key="trajectory_id")

    exported_rows = list(iter_jsonl(args.output))
    summary = build_summary(
        raw_rows=raw_rows,
        verified_rows=verified_rows,
        rejected_rows=rejected_rows,
        exported_rows=exported_rows,
        stats=stats,
        dedupe_counts=dedupe_counts,
        config=config,
    )
    atomic_write_json(report_path, summary)
    reviews = sample_review_rows(
        exported_rows,
        rejected_rows,
        per_domain=int(config["export"].get("review_per_domain", 50)),
    )
    append_unique(review_path, reviews, key="review_id")
    update_manifest(
        run_root / "manifest.json",
        {
            "raw": raw_path,
            "verified": args.verified,
            "rejected": rejected_path,
            "rft_train": args.output,
            "recovery_candidates": recovery_path,
            "summary": report_path,
            "sample_review": review_path,
        },
    )
    counts = {
        "selected": len(selected),
        "exported_total": len(exported_rows),
        "recovery_candidates": len(recovery_rows),
    }
    write_stage_metadata(
        meta_path, input_hashes=input_hashes, config_hash=config_hash, counts=counts
    )
    print(
        f"selected={counts['selected']} exported_total={counts['exported_total']} "
        f"recovery_candidates={counts['recovery_candidates']} report={report_path}"
    )


if __name__ == "__main__":
    main()
