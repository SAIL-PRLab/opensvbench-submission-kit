#!/usr/bin/env python3
"""Generate CHiME-6 domestic far-field verification trials."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    segment_id: str
    speaker_id: str
    session_id: str
    location: str
    ref_array: str
    array_channel: str
    start: float
    end: float
    duration: float
    overlap_ratio: float
    text: str
    wearable_utt: str
    array_utt: str


@dataclass(frozen=True)
class TargetPair:
    enroll: Segment
    test: Segment

    @property
    def condition(self) -> str:
        return self.test.location


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CHiME-6 domestic far-field trials")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-pair-metadata", default=None)
    parser.add_argument("--output-stats", default=None)
    parser.add_argument("--max-target-trials", type=int, default=5000)
    parser.add_argument("--nontarget-ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def canonical_pair(utt1: str, utt2: str) -> tuple[str, str]:
    return (utt1, utt2) if utt1 <= utt2 else (utt2, utt1)


def load_manifest(path: Path) -> list[Segment]:
    segments: list[Segment] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append(
                Segment(
                    segment_id=row["segment_id"],
                    speaker_id=row["speaker_id"],
                    session_id=row["session_id"],
                    location=row["location"],
                    ref_array=row["ref_array"],
                    array_channel=row["array_channel"],
                    start=float(row["start"]),
                    end=float(row["end"]),
                    duration=float(row["duration"]),
                    overlap_ratio=float(row["overlap_ratio"]),
                    text=row["text"],
                    wearable_utt=row["wearable_utt"],
                    array_utt=row["array_utt"],
                )
            )
    if not segments:
        raise ValueError(f"No CHiME-6 segments found in {path}")
    return segments


def round_robin_limit(grouped: dict[str, list[TargetPair]], limit: int, rng: random.Random) -> list[TargetPair]:
    prepared: dict[str, list[TargetPair]] = {}
    active: list[str] = []
    for key, items in grouped.items():
        copied = list(items)
        rng.shuffle(copied)
        if copied:
            prepared[key] = copied
            active.append(key)
    rng.shuffle(active)

    total = sum(len(items) for items in prepared.values())
    if limit <= 0 or total <= limit:
        selected: list[TargetPair] = []
        for key in sorted(prepared):
            selected.extend(prepared[key])
        return selected

    selected: list[TargetPair] = []
    while active and len(selected) < limit:
        next_active: list[str] = []
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


def build_target_pool(segments: list[Segment]) -> tuple[dict[str, list[TargetPair]], dict]:
    by_speaker_location: dict[tuple[str, str], list[Segment]] = defaultdict(list)
    for segment in segments:
        by_speaker_location[(segment.speaker_id, segment.location)].append(segment)

    grouped: dict[str, list[TargetPair]] = defaultdict(list)
    candidates = 0
    for (speaker_id, location), items in by_speaker_location.items():
        sorted_items = sorted(items, key=lambda item: (item.session_id, item.start, item.segment_id))
        if len(sorted_items) < 2:
            continue
        for enroll in sorted_items:
            for test in sorted_items:
                if enroll.segment_id == test.segment_id:
                    continue
                grouped[f"{speaker_id}|{location}"].append(TargetPair(enroll=enroll, test=test))
                candidates += 1

    stats = {
        "target_candidates": candidates,
        "target_candidate_groups": len(grouped),
    }
    return grouped, stats


def ordered_impostors(target: TargetPair, segments: list[Segment], rng: random.Random) -> list[tuple[Segment, str]]:
    def different_speaker(segment: Segment) -> bool:
        return segment.speaker_id != target.enroll.speaker_id

    same_session_same_condition = [
        segment
        for segment in segments
        if different_speaker(segment)
        and segment.session_id == target.test.session_id
        and segment.location == target.test.location
        and segment.ref_array == target.test.ref_array
        and segment.array_channel == target.test.array_channel
    ]
    same_session_same_location = [
        segment
        for segment in segments
        if different_speaker(segment)
        and segment.session_id == target.test.session_id
        and segment.location == target.test.location
        and segment.segment_id not in {item.segment_id for item in same_session_same_condition}
    ]
    same_location_other_session = [
        segment
        for segment in segments
        if different_speaker(segment)
        and segment.session_id != target.test.session_id
        and segment.location == target.test.location
    ]

    for items in (same_session_same_condition, same_session_same_location, same_location_other_session):
        rng.shuffle(items)

    ordered: list[tuple[Segment, str]] = []
    ordered.extend((segment, "same_session_location_ref_array") for segment in same_session_same_condition)
    ordered.extend((segment, "same_session_location") for segment in same_session_same_location)
    ordered.extend((segment, "same_location_other_session") for segment in same_location_other_session)
    return ordered


def write_trials(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    with path.open("w") as f:
        for utt1, utt2, label, condition in rows:
            f.write(f"{utt1} {utt2} {label} {condition}\n")


def pair_metadata_row(utt1: Segment, utt2: Segment, label: str, condition: str, source: str) -> dict:
    return {
        "utt1": utt1.wearable_utt,
        "utt2": utt2.array_utt,
        "label": label,
        "condition": condition,
        "speaker1": utt1.speaker_id,
        "speaker2": utt2.speaker_id,
        "session1": utt1.session_id,
        "session2": utt2.session_id,
        "location1": utt1.location,
        "location2": utt2.location,
        "channel_type1": "wearable",
        "channel_type2": "array",
        "array_id1": "none",
        "array_id2": utt2.ref_array,
        "array_channel1": "none",
        "array_channel2": utt2.array_channel,
        "overlap_ratio1": f"{utt1.overlap_ratio:.6f}",
        "overlap_ratio2": f"{utt2.overlap_ratio:.6f}",
        "segment1": utt1.segment_id,
        "segment2": utt2.segment_id,
        "source": source,
    }


def write_pair_metadata(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "utt1",
        "utt2",
        "label",
        "condition",
        "speaker1",
        "speaker2",
        "session1",
        "session2",
        "location1",
        "location2",
        "channel_type1",
        "channel_type2",
        "array_id1",
        "array_id2",
        "array_channel1",
        "array_channel2",
        "overlap_ratio1",
        "overlap_ratio2",
        "segment1",
        "segment2",
        "source",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = get_args()
    rng = random.Random(args.seed)
    segments = load_manifest(Path(args.manifest).resolve())
    grouped_targets, prep_stats = build_target_pool(segments)
    selected_targets = round_robin_limit(grouped_targets, args.max_target_trials, rng)

    rows: list[tuple[str, str, str, str]] = []
    meta_rows: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    condition_counts = Counter()
    nontarget_source_counts = Counter()
    kept_target_count = 0
    skipped_duplicate_targets = 0
    skipped_insufficient_impostors = 0

    for target in sorted(selected_targets, key=lambda item: (item.condition, item.enroll.segment_id, item.test.segment_id)):
        key = canonical_pair(target.enroll.wearable_utt, target.test.array_utt)
        if key in seen_pairs:
            skipped_duplicate_targets += 1
            continue

        nontarget_candidates: list[tuple[Segment, str, tuple[str, str]]] = []
        for impostor, source in ordered_impostors(target, segments, rng):
            impostor_key = canonical_pair(target.enroll.wearable_utt, impostor.array_utt)
            if impostor_key in seen_pairs:
                continue
            nontarget_candidates.append((impostor, source, impostor_key))
            if len(nontarget_candidates) >= args.nontarget_ratio:
                break
        if len(nontarget_candidates) < args.nontarget_ratio:
            skipped_insufficient_impostors += 1
            continue

        seen_pairs.add(key)
        rows.append((target.enroll.wearable_utt, target.test.array_utt, "target", target.condition))
        condition_counts[target.condition] += 1
        meta_rows.append(pair_metadata_row(target.enroll, target.test, "target", target.condition, "wearable_to_array_target"))
        kept_target_count += 1

        for impostor, source, impostor_key in nontarget_candidates:
            seen_pairs.add(impostor_key)
            rows.append((target.enroll.wearable_utt, impostor.array_utt, "nontarget", target.condition))
            condition_counts[target.condition] += 1
            nontarget_source_counts[source] += 1
            meta_rows.append(pair_metadata_row(target.enroll, impostor, "nontarget", target.condition, source))

    rows.sort()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trials(output_path, rows)

    if args.output_pair_metadata:
        meta_path = Path(args.output_pair_metadata).resolve()
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        write_pair_metadata(meta_path, meta_rows)

    stats = {
        "manifest": str(Path(args.manifest).resolve()),
        "num_trials": len(rows),
        "labels": dict(sorted(Counter(row[2] for row in rows).items())),
        "conditions": dict(sorted(condition_counts.items())),
        "target_count": kept_target_count,
        "nontarget_count": kept_target_count * args.nontarget_ratio,
        "nontarget_ratio_requested": args.nontarget_ratio,
        "selected_target_candidates": len(selected_targets),
        "skipped_duplicate_targets": skipped_duplicate_targets,
        "skipped_insufficient_impostors": skipped_insufficient_impostors,
        "num_speakers": len({segment.speaker_id for segment in segments}),
        "num_source_segments": len(segments),
        "unique_audio_count": len({row[0] for row in rows} | {row[1] for row in rows}),
        "nontarget_source_counts": dict(sorted(nontarget_source_counts.items())),
        **prep_stats,
    }
    if args.output_stats:
        stats_path = Path(args.output_stats).resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n")

    print("=== [CHiME-6] Trial generation done ===")
    print(f"  source segments : {stats['num_source_segments']}")
    print(f"  speakers        : {stats['num_speakers']}")
    print(f"  targets         : {stats['target_count']}")
    print(f"  labels          : {stats['labels']}")
    print(f"  conditions      : {stats['conditions']}")


if __name__ == "__main__":
    main()
