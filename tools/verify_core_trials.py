#!/usr/bin/env python3
"""Verify local trial files against the frozen OpenSVBench core fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_catalog import iter_scenarios


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify OpenSVBench core trial fingerprints")
    parser.add_argument("--data", required=True, help="Benchmark data root")
    parser.add_argument(
        "--fingerprints",
        default=str(ROOT / "resources" / "core_trial_fingerprints.tsv"),
        help="Frozen core trial fingerprint TSV",
    )
    return parser.parse_args()


def sha256_and_rows(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    rows = 0
    with path.open("rb") as f:
        for line in f:
            rows += 1
            h.update(line)
    return h.hexdigest(), rows


def load_expected(path: Path) -> dict[str, dict[str, str]]:
    expected = {}
    with path.open("r") as f:
        header = f.readline().strip().split("\t")
        if header != ["scenario_id", "submission_filename", "rows", "sha256"]:
            raise ValueError(f"Unexpected fingerprint header: {header}")
        for line in f:
            scenario_id, submission_filename, rows, digest = line.strip().split("\t")
            expected[scenario_id] = {
                "submission_filename": submission_filename,
                "rows": rows,
                "sha256": digest,
            }
    return expected


def main() -> None:
    args = get_args()
    data_root = Path(args.data).resolve()
    expected = load_expected(Path(args.fingerprints))
    failures = []

    for spec in iter_scenarios(collection="core"):
        trial_path = spec.trial_path(data_root)
        item = expected.get(spec.scenario_id)
        if item is None:
            failures.append(f"{spec.scenario_id}: missing frozen fingerprint entry")
            continue
        if not trial_path.is_file():
            failures.append(f"{spec.scenario_id}: missing trial file {trial_path}")
            continue
        digest, rows = sha256_and_rows(trial_path)
        if str(rows) != item["rows"] or digest != item["sha256"]:
            failures.append(
                f"{spec.scenario_id}: local trial file differs from frozen core "
                f"(rows {rows} vs {item['rows']}, sha256 {digest[:12]} vs {item['sha256'][:12]})"
            )

    if failures:
        print("[OpenSVBench] Core trial fingerprint check failed:")
        for failure in failures:
            print(f"  - {failure}")
        print("Regenerate the affected dataset with the current submission kit before packaging scores.")
        raise SystemExit(2)

    print(f"[OpenSVBench] Core trial fingerprint check passed: {len(expected)} trial sets.")


if __name__ == "__main__":
    main()
