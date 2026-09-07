#!/usr/bin/env python3
"""Filter OpenSVBench trial files by condition label."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a trial file by one or more condition labels")
    parser.add_argument("--input", required=True, help="Input trial file")
    parser.add_argument("--output", required=True, help="Filtered output trial file")
    parser.add_argument("--conditions", nargs="+", required=True, help="Condition labels to keep")
    parser.add_argument("--output-stats", default=None, help="Optional JSON stats path")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    keep = set(args.conditions)

    total = 0
    kept = 0
    label_counts = Counter()
    condition_counts = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r") as src, output_path.open("w") as dst:
        for line in src:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            total += 1
            condition = parts[3]
            if condition not in keep:
                continue
            dst.write(line)
            kept += 1
            label_counts[parts[2]] += 1
            condition_counts[condition] += 1

    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "conditions_kept": sorted(keep),
        "total_input_trials": total,
        "total_output_trials": kept,
        "label_counts": dict(label_counts),
        "condition_counts": dict(condition_counts),
    }

    if args.output_stats:
        stats_path = Path(args.output_stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n")

    print(
        "[Trials] Filtered "
        f"{input_path} -> {output_path}: kept={kept}/{total} "
        f"conditions={sorted(condition_counts.items())}"
    )


if __name__ == "__main__":
    main()
