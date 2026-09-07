#!/usr/bin/env python3
"""Generate VOiCES noise/reverb robustness trials for OpenSVBench."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CANONICAL_ROOM = "rm1"
CANONICAL_NOISE = "none"
CANONICAL_MIC = "mc01"


@dataclass(frozen=True)
class VoiceRecord:
    utt_id: str
    speaker_id: str
    gender: str
    room: str
    noise: str
    mic: str
    mic_type: str
    mic_location: str
    duration_sec: float
    source_segment: str

    @property
    def condition_key(self) -> str:
        return f"{self.room}_{self.noise}_{self.mic}"


@dataclass(frozen=True)
class TargetPair:
    enroll: VoiceRecord
    test: VoiceRecord
    condition_bucket: str


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VOiCES noise/reverb robustness trials")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-pair-metadata", default=None)
    parser.add_argument("--output-stats", default=None)
    parser.add_argument("--max-target-trials", type=int, default=5000)
    parser.add_argument("--nontarget-ratio", type=int, default=10)
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def canonical_pair(utt1: str, utt2: str) -> tuple[str, str]:
    return (utt1, utt2) if utt1 <= utt2 else (utt2, utt1)


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
        "gender1",
        "gender2",
        "room1",
        "room2",
        "noise1",
        "noise2",
        "mic1",
        "mic2",
        "mic_type1",
        "mic_type2",
        "mic_location1",
        "mic_location2",
        "source_segment1",
        "source_segment2",
        "source",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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
        limit = sum(len(items) for items in prepared.values())

    total = sum(len(items) for items in prepared.values())
    if total <= limit:
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


def condition_bucket(record: VoiceRecord) -> str | None:
    room_shift = record.room != CANONICAL_ROOM
    noise_shift = record.noise != CANONICAL_NOISE
    mic_shift = record.mic != CANONICAL_MIC

    if not room_shift and not noise_shift and not mic_shift:
        return None
    if not room_shift and noise_shift and not mic_shift:
        return "noise_only"
    if room_shift and not noise_shift and not mic_shift:
        return "room_only"
    if not room_shift and not noise_shift and mic_shift:
        return "mic_only"
    if room_shift and noise_shift and not mic_shift:
        return "room_noise"
    if room_shift and not noise_shift and mic_shift:
        return "room_mic"
    if not room_shift and noise_shift and mic_shift:
        return "mic_noise"
    return "room_mic_noise"


def load_manifest(path: Path) -> list[VoiceRecord]:
    records: list[VoiceRecord] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                VoiceRecord(
                    utt_id=row["utt_id"],
                    speaker_id=row["speaker_id"],
                    gender=row["gender"],
                    room=row["room"],
                    noise=row["noise"],
                    mic=row["mic"],
                    mic_type=row["mic_type"],
                    mic_location=row["mic_location"],
                    duration_sec=float(row["duration_sec"]),
                    source_segment=row["source_segment"],
                )
            )
    if not records:
        raise ValueError(f"No VOiCES records found in {path}")
    return records


def build_target_pool(records: list[VoiceRecord], min_duration: float) -> tuple[dict[str, list[TargetPair]], dict]:
    by_speaker_segment: dict[str, dict[str, list[VoiceRecord]]] = defaultdict(lambda: defaultdict(list))
    easy_lookup: dict[tuple[str, str], VoiceRecord] = {}

    for record in records:
        if record.duration_sec < min_duration:
            continue
        by_speaker_segment[record.speaker_id][record.source_segment].append(record)
        if (
            record.room == CANONICAL_ROOM
            and record.noise == CANONICAL_NOISE
            and record.mic == CANONICAL_MIC
        ):
            easy_lookup[(record.speaker_id, record.source_segment)] = record

    grouped_targets: dict[str, list[TargetPair]] = defaultdict(list)
    eligible_speakers = 0
    eligible_segments = 0
    target_candidates = 0
    bucket_counts = Counter()

    for speaker_id, source_map in by_speaker_segment.items():
        segments = sorted(seg for seg in source_map if (speaker_id, seg) in easy_lookup)
        if len(segments) < 2:
            continue
        eligible_speakers += 1
        eligible_segments += len(segments)
        for enroll_segment in segments:
            enroll_record = easy_lookup[(speaker_id, enroll_segment)]
            for test_segment in segments:
                if test_segment == enroll_segment:
                    continue
                for test_record in sorted(source_map[test_segment], key=lambda item: item.utt_id):
                    bucket = condition_bucket(test_record)
                    if bucket is None:
                        continue
                    grouped_targets[f"{bucket}|{speaker_id}"].append(
                        TargetPair(enroll=enroll_record, test=test_record, condition_bucket=bucket)
                    )
                    bucket_counts[bucket] += 1
                    target_candidates += 1

    stats = {
        "eligible_speakers": eligible_speakers,
        "eligible_source_segments": eligible_segments,
        "target_candidates": target_candidates,
        "target_candidate_buckets": dict(sorted(bucket_counts.items())),
    }
    return grouped_targets, stats


def ordered_impostors(
    target: TargetPair,
    by_bucket: dict[str, list[VoiceRecord]],
    by_condition: dict[tuple[str, str], list[VoiceRecord]],
    rng: random.Random,
) -> list[tuple[VoiceRecord, str]]:
    exact = [
        record
        for record in by_condition[(target.condition_bucket, target.test.condition_key)]
        if record.speaker_id != target.enroll.speaker_id
    ]
    bucket_all = [
        record
        for record in by_bucket[target.condition_bucket]
        if record.speaker_id != target.enroll.speaker_id
        and record.utt_id not in {candidate.utt_id for candidate in exact}
    ]

    exact_same_gender = [record for record in exact if record.gender == target.enroll.gender]
    exact_other_gender = [record for record in exact if record.gender != target.enroll.gender]
    bucket_same_gender = [record for record in bucket_all if record.gender == target.enroll.gender]
    bucket_other_gender = [record for record in bucket_all if record.gender != target.enroll.gender]

    for items in (exact_same_gender, exact_other_gender, bucket_same_gender, bucket_other_gender):
        rng.shuffle(items)

    ordered: list[tuple[VoiceRecord, str]] = []
    ordered.extend((record, "same_condition_nontarget") for record in exact_same_gender)
    ordered.extend((record, "same_condition_nontarget") for record in exact_other_gender)
    ordered.extend((record, "same_bucket_nontarget") for record in bucket_same_gender)
    ordered.extend((record, "same_bucket_nontarget") for record in bucket_other_gender)
    return ordered


def add_pair_metadata(rows: list[dict], utt1: VoiceRecord, utt2: VoiceRecord, label: str, source: str) -> None:
    rows.append(
        {
            "utt1": utt1.utt_id,
            "utt2": utt2.utt_id,
            "label": label,
            "speaker1": utt1.speaker_id,
            "speaker2": utt2.speaker_id,
            "gender1": utt1.gender,
            "gender2": utt2.gender,
            "room1": utt1.room,
            "room2": utt2.room,
            "noise1": utt1.noise,
            "noise2": utt2.noise,
            "mic1": utt1.mic,
            "mic2": utt2.mic,
            "mic_type1": utt1.mic_type,
            "mic_type2": utt2.mic_type,
            "mic_location1": utt1.mic_location,
            "mic_location2": utt2.mic_location,
            "source_segment1": utt1.source_segment,
            "source_segment2": utt2.source_segment,
            "source": source,
        }
    )


def main() -> None:
    args = get_args()
    rng = random.Random(args.seed)

    manifest_path = Path(args.manifest).resolve()
    records = load_manifest(manifest_path)
    grouped_targets, prep_stats = build_target_pool(records, args.min_duration)
    selected_targets = round_robin_limit(grouped_targets, args.max_target_trials, rng)

    by_bucket: dict[str, list[VoiceRecord]] = defaultdict(list)
    by_condition: dict[tuple[str, str], list[VoiceRecord]] = defaultdict(list)
    for record in records:
        if record.duration_sec < args.min_duration:
            continue
        bucket = condition_bucket(record)
        if bucket is None:
            continue
        by_bucket[bucket].append(record)
        by_condition[(bucket, record.condition_key)].append(record)

    rows: list[tuple[str, str, str, str]] = []
    pair_meta_rows: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    nontarget_source_counts = Counter()
    condition_counts = Counter()

    for target in sorted(selected_targets, key=lambda item: (item.condition_bucket, item.enroll.utt_id, item.test.utt_id)):
        pair_key = canonical_pair(target.enroll.utt_id, target.test.utt_id)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        rows.append((target.enroll.utt_id, target.test.utt_id, "target", target.condition_bucket))
        condition_counts[target.condition_bucket] += 1
        add_pair_metadata(pair_meta_rows, target.enroll, target.test, "target", "voices_easy_to_hard_target")

    nontarget_total = 0
    for target in sorted(selected_targets, key=lambda item: (item.condition_bucket, item.enroll.utt_id, item.test.utt_id)):
        ordered = ordered_impostors(target, by_bucket, by_condition, rng)
        added = 0
        for impostor, source in ordered:
            pair_key = canonical_pair(target.enroll.utt_id, impostor.utt_id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            rows.append((target.enroll.utt_id, impostor.utt_id, "nontarget", target.condition_bucket))
            condition_counts[target.condition_bucket] += 1
            nontarget_total += 1
            nontarget_source_counts[source] += 1
            add_pair_metadata(pair_meta_rows, target.enroll, impostor, "nontarget", source)
            added += 1
            if added >= args.nontarget_ratio:
                break
        if added < args.nontarget_ratio:
            raise RuntimeError(
                f"Could not find {args.nontarget_ratio} impostors for target "
                f"{target.enroll.utt_id} -> {target.test.utt_id}; only found {added}"
            )

    rows.sort()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trials(output_path, rows)

    if args.output_pair_metadata:
        pair_metadata_path = Path(args.output_pair_metadata).resolve()
        pair_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        write_pair_metadata(pair_metadata_path, pair_meta_rows)

    stats = {
        "manifest": str(manifest_path),
        "num_trials": len(rows),
        "labels": dict(sorted(Counter(row[2] for row in rows).items())),
        "conditions": dict(sorted(Counter(row[3] for row in rows).items())),
        "target_count": sum(1 for row in rows if row[2] == "target"),
        "nontarget_count": nontarget_total,
        "nontarget_ratio_requested": args.nontarget_ratio,
        "min_duration": args.min_duration,
        "max_target_trials": args.max_target_trials,
        "unique_audio_count": len({utt for row in rows for utt in row[:2]}),
        "num_speakers": len({record.speaker_id for record in records}),
        "num_utterances": len(records),
        "nontarget_source_counts": dict(sorted(nontarget_source_counts.items())),
        **prep_stats,
    }
    if args.output_stats:
        stats_path = Path(args.output_stats).resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n")

    print("=== [VOiCES] Trial generation done ===")
    print(f"  utterances       : {stats['num_utterances']}")
    print(f"  speakers         : {stats['num_speakers']}")
    print(f"  target candidates: {stats['target_candidates']}")
    print(f"  targets          : {stats['target_count']}")
    print(f"  nontargets       : {stats['nontarget_count']}")
    print(f"  conditions       : {stats['conditions']}")
    print(f"  nontarget source : {stats['nontarget_source_counts']}")


if __name__ == "__main__":
    main()
