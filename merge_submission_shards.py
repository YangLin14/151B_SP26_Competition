#!/usr/bin/env python3
"""Merge partial private submission CSV shards into one Kaggle CSV."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge submission shard CSVs")
    parser.add_argument("--private-path", default="data/private.jsonl")
    parser.add_argument("--pattern", default="results/submission_part_*.csv")
    parser.add_argument("--output-path", default="results/submission_final.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_ids = [
        int(json.loads(line)["id"])
        for line in Path(args.private_path).open(encoding="utf-8")
        if line.strip()
    ]
    required_set = set(required_ids)

    shard_paths = sorted(glob.glob(args.pattern))
    if not shard_paths:
        raise SystemExit(f"No shard CSVs matched pattern: {args.pattern}")

    rows: dict[int, str] = {}
    duplicates: list[int] = []
    for shard_path in shard_paths:
        with Path(shard_path).open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                qid = int(row["id"])
                if qid in rows:
                    duplicates.append(qid)
                    continue
                rows[qid] = row["response"]
        print(f"Loaded {shard_path}: cumulative rows={len(rows)}")

    missing = sorted(required_set - set(rows))
    extra = sorted(set(rows) - required_set)
    empty = sorted(qid for qid, response in rows.items() if not response.strip())

    print("\n=== Merge Validation ===")
    print(f"Required ids: {len(required_ids)}")
    print(f"Merged rows: {len(rows)}")
    print(f"Missing: {len(missing)}")
    print(f"Extra: {len(extra)}")
    print(f"Duplicates skipped: {len(duplicates)}")
    print(f"Empty responses: {len(empty)}")
    if missing:
        print("Missing examples:", missing[:20])
    if extra:
        print("Extra examples:", extra[:20])
    if duplicates:
        print("Duplicate examples:", sorted(set(duplicates))[:20])
    if empty:
        print("Empty examples:", empty[:20])

    if missing or extra or empty:
        raise SystemExit("Merged submission is incomplete; fix shards before submitting.")

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for qid in required_ids:
            writer.writerow([qid, rows[qid]])

    print(f"\nSaved final submission: {out_path}")


if __name__ == "__main__":
    main()
