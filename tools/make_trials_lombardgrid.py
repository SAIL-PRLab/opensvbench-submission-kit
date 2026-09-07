#!/usr/bin/env python3
"""Generate Lombard Grid plain-to-Lombard speaker verification trials."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LombardRecord:
    utt_id: str
    speaker_id: str
    style: str
    text: str
    duration_sec: float


@dataclass(frozen=True)
class TargetPair:
    enroll: LombardRecord
    test: LombardRecord
    condition: str


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Lombard Grid plain-to-Lombard SV trials")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-pair-metadata", default=None)
    parser.add_argument("--output-stats", default=None)
    parser.add_argument("--max-target-trials", type=int, default=0)
    parser.add_argument("--nontarget-ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def canonical_pair(utt1: str, utt2: str) -> tuple[str, str]:
    return (utt1, utt2) if utt1 <= utt2 else (utt2, utt1)


def load_manifest(path: Path) -> list[LombardRecord]:
    records: list[LombardRecord] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                LombardRecord(
                    utt_id=row["utt_id"],
                    speaker_id=row["speaker_id"],
                    style=row["style"],
                    text=row["actual_text"] or row["text"],
                    duration_sec=float(row["duration_sec"] or 0.0),
                )
            )
    if not records:
        raise ValueError(f"No Lombard Grid records found in {path}")
    return records


def round_robin_limit(grouped: dict[str, list[TargetPair]], limit: int, rng: random.Random) -> list[TargetPair]:
    prepared: dict[str, list[TargetPair]] = {}
    active: list[str] = []
    for key, items in grouped.items():
        copied = list(items)
        rng.shuffle(copied)
        if copied:
            prepared[key] = copied
            active.append(key)

    if limit <= 0:
        selected: list[TargetPair] = []
        for key in sorted(prepared):
            selected.extend(prepared[key])
        return selected

    selected = []
    while active and len(selected) < limit:
        next_active = []
        for key in active:
            if len(selected) >= limit:
                break
            items = prepared[key]
            if items:
                selected.append(items.pop())
            if items:
                next_active.append(key)
        active = next_active
    return selected


def condition_for(enroll: LombardRecord, test: LombardRecord) -> str:
    if enroll.text == test.text:
        return "plain_to_lombard_same_text"
    return "plain_to_lombard_diff_text"


def build_target_pool(records: list[LombardRecord]) -> tuple[dict[str, list[TargetPair]], dict]:
    by_speaker_style: dict[str, dict[str, list[LombardRecord]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_speaker_style[record.speaker_id][record.style].append(record)

    grouped_targets: dict[str, list[TargetPair]] = defaultdict(list)
    condition_counts = Counter()
    eligible_speakers = 0

    for speaker_id, style_map in by_speaker_style.items():
        plain_records = sorted(style_map.get("plain", []), key=lambda item: item.utt_id)
        lombard_records = sorted(style_map.get("lombard", []), key=lambda item: item.utt_id)
        if not plain_records or not lombard_records:
            continue
        eligible_speakers += 1
        lombard_by_text: dict[str, list[LombardRecord]] = defaultdict(list)
        for record in lombard_records:
            lombard_by_text[record.text].append(record)
        for idx, enroll in enumerate(plain_records):
            if lombard_by_text.get(enroll.text):
                candidates = sorted(lombard_by_text[enroll.text], key=lambda item: item.utt_id)
                test = candidates[idx % len(candidates)]
            else:
                test = lombard_records[idx % len(lombard_records)]
            condition = condition_for(enroll, test)
            grouped_targets[speaker_id].append(TargetPair(enroll=enroll, test=test, condition=condition))
            condition_counts[condition] += 1

    stats = {
        "eligible_speakers": eligible_speakers,
        "target_candidates": sum(len(items) for items in grouped_targets.values()),
        "target_candidate_conditions": dict(sorted(condition_counts.items())),
    }
    return grouped_targets, stats


def build_impostor_pool(records: list[LombardRecord]) -> tuple[dict[str, list[LombardRecord]], list[LombardRecord]]:
    by_text: dict[str, list[LombardRecord]] = defaultdict(list)
    all_lombard = []
    for record in records:
        if record.style != "lombard":
            continue
        by_text[record.text].append(record)
        all_lombard.append(record)
    return by_text, all_lombard


def ordered_impostors(
    target: TargetPair,
    by_text: dict[str, list[LombardRecord]],
    all_lombard: list[LombardRecord],
    rng: random.Random,
) -> list[tuple[LombardRecord, str]]:
    same_text = [
        record
        for record in by_text.get(target.enroll.text, [])
        if record.speaker_id != target.enroll.speaker_id
    ]
    other_text = [
        record
        for record in all_lombard
        if record.speaker_id != target.enroll.speaker_id
        and record.utt_id not in {candidate.utt_id for candidate in same_text}
    ]
    rng.shuffle(same_text)
    rng.shuffle(other_text)
    return [(record, "same_text_nontarget") for record in same_text] + [
        (record, "other_text_nontarget") for record in other_text
    ]


def write_trials(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    with path.open("w") as f:
        for utt1, utt2, label, condition in rows:
            f.write(f"{utt1} {utt2} {label} {condition}\n")


def write_pair_metadata(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "utt1",
        "utt2",
        "label",
        "speaker1",
        "speaker2",
        "style1",
        "style2",
        "text1",
        "text2",
        "source",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_pair_metadata(rows: list[dict], utt1: LombardRecord, utt2: LombardRecord, label: str, source: str) -> None:
    rows.append(
        {
            "utt1": utt1.utt_id,
            "utt2": utt2.utt_id,
            "label": label,
            "speaker1": utt1.speaker_id,
            "speaker2": utt2.speaker_id,
            "style1": utt1.style,
            "style2": utt2.style,
            "text1": utt1.text,
            "text2": utt2.text,
            "source": source,
        }
    )


def main() -> None:
    args = get_args()
    rng = random.Random(args.seed)
    manifest_path = Path(args.manifest).resolve()
    records = load_manifest(manifest_path)

    grouped_targets, prep_stats = build_target_pool(records)
    selected_targets = round_robin_limit(grouped_targets, args.max_target_trials, rng)
    by_text, all_lombard = build_impostor_pool(records)

    rows: list[tuple[str, str, str, str]] = []
    pair_meta_rows: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    condition_counts = Counter()
    nontarget_source_counts = Counter()

    for target in sorted(selected_targets, key=lambda item: (item.enroll.speaker_id, item.enroll.utt_id)):
        pair_key = canonical_pair(target.enroll.utt_id, target.test.utt_id)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        rows.append((target.enroll.utt_id, target.test.utt_id, "target", target.condition))
        condition_counts[target.condition] += 1
        add_pair_metadata(pair_meta_rows, target.enroll, target.test, "target", "plain_to_lombard_target")

    for target in sorted(selected_targets, key=lambda item: (item.enroll.speaker_id, item.enroll.utt_id)):
        added = 0
        for impostor, source in ordered_impostors(target, by_text, all_lombard, rng):
            pair_key = canonical_pair(target.enroll.utt_id, impostor.utt_id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            condition = condition_for(target.enroll, impostor)
            rows.append((target.enroll.utt_id, impostor.utt_id, "nontarget", condition))
            condition_counts[condition] += 1
            nontarget_source_counts[source] += 1
            add_pair_metadata(pair_meta_rows, target.enroll, impostor, "nontarget", source)
            added += 1
            if added >= args.nontarget_ratio:
                break
        if added < args.nontarget_ratio:
            raise RuntimeError(
                f"Could not find {args.nontarget_ratio} impostors for {target.enroll.utt_id}; found {added}"
            )

    rows.sort()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trials(output_path, rows)

    if args.output_pair_metadata:
        pair_metadata_path = Path(args.output_pair_metadata).resolve()
        pair_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        write_pair_metadata(pair_metadata_path, pair_meta_rows)

    label_counts = Counter(row[2] for row in rows)
    stats = {
        "manifest": str(manifest_path),
        "num_trials": len(rows),
        "num_target_trials": label_counts.get("target", 0),
        "num_nontarget_trials": label_counts.get("nontarget", 0),
        "num_trial_utterances": len({row[0] for row in rows} | {row[1] for row in rows}),
        "condition_counts": dict(sorted(condition_counts.items())),
        "nontarget_source_counts": dict(sorted(nontarget_source_counts.items())),
        "max_target_trials": args.max_target_trials,
        "nontarget_ratio": args.nontarget_ratio,
        "seed": args.seed,
        **prep_stats,
    }
    if args.output_stats:
        Path(args.output_stats).resolve().write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n")

    print("=== [LombardGrid] Trial generation done ===")
    print(f"  trials    : {stats['num_trials']}")
    print(f"  target    : {stats['num_target_trials']}")
    print(f"  nontarget : {stats['num_nontarget_trials']}")
    print(f"  conditions: {stats['condition_counts']}")


if __name__ == "__main__":
    main()
