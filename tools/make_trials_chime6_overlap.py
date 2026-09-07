#!/usr/bin/env python3
"""Generate CHiME-6 overlap-robustness trials with same-channel pairs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


TARGET_BUCKETS = ("overlap_light", "overlap_mid", "overlap_heavy")
CHANNEL_PAIRS = ("ww", "aa")


@dataclass(frozen=True)
class Segment:
    segment_id: str
    speaker_id: str
    session_id: str
    location: str
    ref_array: str
    array_channel: str
    overlap_ratio: float
    wearable_utt: str
    array_utt: str


@dataclass(frozen=True)
class TargetPair:
    enroll: Segment
    test: Segment
    condition: str
    channel_pair: str

    @property
    def enroll_utt(self) -> str:
        return self.enroll.wearable_utt if self.channel_pair == "ww" else self.enroll.array_utt

    @property
    def test_utt(self) -> str:
        return self.test.wearable_utt if self.channel_pair == "ww" else self.test.array_utt

    @property
    def channel_type(self) -> str:
        return "wearable" if self.channel_pair == "ww" else "array"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CHiME-6 overlap-robustness trials")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-pair-metadata", required=True)
    parser.add_argument("--output-stats", required=True)
    parser.add_argument("--clean-max-overlap", type=float, default=0.05)
    parser.add_argument("--light-max-overlap", type=float, default=0.20)
    parser.add_argument("--mid-max-overlap", type=float, default=0.50)
    parser.add_argument("--max-target-per-bucket", type=int, default=1200)
    parser.add_argument("--nontarget-ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def canonical_pair(utt1: str, utt2: str) -> tuple[str, str]:
    return (utt1, utt2) if utt1 <= utt2 else (utt2, utt1)


def bucket_overlap(ratio: float, clean_max: float, light_max: float, mid_max: float) -> str:
    if ratio < clean_max:
        return "clean"
    if ratio < light_max:
        return "overlap_light"
    if ratio < mid_max:
        return "overlap_mid"
    return "overlap_heavy"


def load_manifest(path: Path) -> list[Segment]:
    rows: list[Segment] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                Segment(
                    segment_id=row["segment_id"],
                    speaker_id=row["speaker_id"],
                    session_id=row["session_id"],
                    location=row["location"],
                    ref_array=row["ref_array"],
                    array_channel=row["array_channel"],
                    overlap_ratio=float(row["overlap_ratio"]),
                    wearable_utt=row["wearable_utt"],
                    array_utt=row["array_utt"],
                )
            )
    if not rows:
        raise ValueError(f"No CHiME-6 rows in {path}")
    return rows


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

    if limit <= 0:
        limit = sum(len(v) for v in prepared.values())

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


def build_target_candidates(clean_segments: list[Segment], bucket_segments: list[Segment], condition: str, channel_pair: str) -> dict[str, list[TargetPair]]:
    clean_by_spk: dict[str, list[Segment]] = defaultdict(list)
    test_by_spk: dict[str, list[Segment]] = defaultdict(list)
    for seg in clean_segments:
        clean_by_spk[seg.speaker_id].append(seg)
    for seg in bucket_segments:
        test_by_spk[seg.speaker_id].append(seg)

    grouped: dict[str, list[TargetPair]] = defaultdict(list)
    speakers = sorted(set(clean_by_spk) & set(test_by_spk))
    for speaker in speakers:
        for enroll in clean_by_spk[speaker]:
            for test in test_by_spk[speaker]:
                if enroll.segment_id == test.segment_id:
                    continue
                grouped[speaker].append(
                    TargetPair(
                        enroll=enroll,
                        test=test,
                        condition=condition,
                        channel_pair=channel_pair,
                    )
                )
    return grouped


def ordered_impostors(target: TargetPair, bucket_segments: list[Segment], rng: random.Random) -> list[tuple[Segment, str]]:
    def diff_speaker(seg: Segment) -> bool:
        return seg.speaker_id != target.enroll.speaker_id

    same_session_location = [
        seg
        for seg in bucket_segments
        if diff_speaker(seg)
        and seg.session_id == target.test.session_id
        and seg.location == target.test.location
    ]
    same_location_other_session = [
        seg
        for seg in bucket_segments
        if diff_speaker(seg)
        and seg.location == target.test.location
        and seg.segment_id not in {x.segment_id for x in same_session_location}
    ]
    global_fallback = [
        seg
        for seg in bucket_segments
        if diff_speaker(seg)
        and seg.segment_id not in {x.segment_id for x in same_session_location}
        and seg.segment_id not in {x.segment_id for x in same_location_other_session}
    ]

    for items in (same_session_location, same_location_other_session, global_fallback):
        rng.shuffle(items)

    ordered: list[tuple[Segment, str]] = []
    ordered.extend((seg, "same_session_location") for seg in same_session_location)
    ordered.extend((seg, "same_location_other_session") for seg in same_location_other_session)
    ordered.extend((seg, "global_fallback") for seg in global_fallback)
    return ordered


def target_utt(seg: Segment, channel_pair: str) -> str:
    return seg.wearable_utt if channel_pair == "ww" else seg.array_utt


def pair_metadata_row(target: TargetPair, test_segment: Segment, label: str, source: str) -> dict:
    if target.channel_pair == "ww":
        array_id_1 = "none"
        array_id_2 = "none"
        array_ch_1 = "none"
        array_ch_2 = "none"
        channel_1 = "wearable"
        channel_2 = "wearable"
    else:
        array_id_1 = target.enroll.ref_array
        array_id_2 = test_segment.ref_array
        array_ch_1 = target.enroll.array_channel
        array_ch_2 = test_segment.array_channel
        channel_1 = "array"
        channel_2 = "array"

    return {
        "utt1": target.enroll_utt,
        "utt2": target_utt(test_segment, target.channel_pair),
        "label": label,
        "condition": target.condition,
        "speaker1": target.enroll.speaker_id,
        "speaker2": test_segment.speaker_id,
        "session1": target.enroll.session_id,
        "session2": test_segment.session_id,
        "location1": target.enroll.location,
        "location2": test_segment.location,
        "channel_type1": channel_1,
        "channel_type2": channel_2,
        "array_id1": array_id_1,
        "array_id2": array_id_2,
        "array_channel1": array_ch_1,
        "array_channel2": array_ch_2,
        "overlap_ratio1": f"{target.enroll.overlap_ratio:.6f}",
        "overlap_ratio2": f"{test_segment.overlap_ratio:.6f}",
        "segment1": target.enroll.segment_id,
        "segment2": test_segment.segment_id,
        "channel_pair": target.channel_pair,
        "source": source,
    }


def write_trials(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    with path.open("w") as f:
        for utt1, utt2, label, condition in rows:
            f.write(f"{utt1} {utt2} {label} {condition}\n")


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
        "channel_pair",
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

    bucketed: dict[str, list[Segment]] = defaultdict(list)
    for seg in segments:
        bucket = bucket_overlap(seg.overlap_ratio, args.clean_max_overlap, args.light_max_overlap, args.mid_max_overlap)
        bucketed[bucket].append(seg)

    clean_segments = bucketed.get("clean", [])
    if not clean_segments:
        raise ValueError("No clean segments available for CHiME-6 overlap trials.")

    max_target_per_bucket = max(1, args.max_target_per_bucket)
    max_target_per_channel = max(1, max_target_per_bucket // 2)
    if max_target_per_bucket % 2 != 0:
        max_target_per_channel += 1

    selected_targets: list[TargetPair] = []
    candidate_counts = Counter()
    selected_counts = Counter()

    for condition in TARGET_BUCKETS:
        bucket_segments = bucketed.get(condition, [])
        if not bucket_segments:
            continue
        for channel_pair in CHANNEL_PAIRS:
            grouped = build_target_candidates(clean_segments, bucket_segments, condition, channel_pair)
            candidate_counts[f"{condition}:{channel_pair}"] = sum(len(v) for v in grouped.values())
            picked = round_robin_limit(grouped, max_target_per_channel, rng)
            selected_targets.extend(picked)
            selected_counts[f"{condition}:{channel_pair}"] = len(picked)

    rows: list[tuple[str, str, str, str]] = []
    pair_rows: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    condition_counts = Counter()
    channel_counts = Counter()
    nontarget_source_counts = Counter()
    kept_targets = 0
    skipped_dupe_target = 0
    skipped_short_impostor = 0

    bucket_lookup = {condition: bucketed.get(condition, []) for condition in TARGET_BUCKETS}

    for target in sorted(selected_targets, key=lambda x: (x.condition, x.channel_pair, x.enroll.segment_id, x.test.segment_id)):
        target_key = canonical_pair(target.enroll_utt, target.test_utt)
        if target_key in seen_pairs:
            skipped_dupe_target += 1
            continue

        nontarget_candidates: list[tuple[Segment, str, tuple[str, str]]] = []
        for impostor, source in ordered_impostors(target, bucket_lookup[target.condition], rng):
            impostor_utt = target_utt(impostor, target.channel_pair)
            key = canonical_pair(target.enroll_utt, impostor_utt)
            if key in seen_pairs:
                continue
            nontarget_candidates.append((impostor, source, key))
            if len(nontarget_candidates) >= args.nontarget_ratio:
                break
        if len(nontarget_candidates) < args.nontarget_ratio:
            skipped_short_impostor += 1
            continue

        seen_pairs.add(target_key)
        rows.append((target.enroll_utt, target.test_utt, "target", target.condition))
        pair_rows.append(
            pair_metadata_row(
                target=target,
                test_segment=target.test,
                label="target",
                source=f"clean_to_overlap_target_{target.channel_pair}",
            )
        )
        condition_counts[target.condition] += 1
        channel_counts[target.channel_pair] += 1
        kept_targets += 1

        for impostor, source, key in nontarget_candidates:
            seen_pairs.add(key)
            rows.append((target.enroll_utt, target_utt(impostor, target.channel_pair), "nontarget", target.condition))
            pair_rows.append(
                pair_metadata_row(
                    target=target,
                    test_segment=impostor,
                    label="nontarget",
                    source=f"{source}_{target.channel_pair}",
                )
            )
            condition_counts[target.condition] += 1
            channel_counts[target.channel_pair] += 1
            nontarget_source_counts[source] += 1

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_trials(output_path, rows)

    pair_meta_path = Path(args.output_pair_metadata).resolve()
    pair_meta_path.parent.mkdir(parents=True, exist_ok=True)
    write_pair_metadata(pair_meta_path, pair_rows)

    stats = {
        "manifest": str(Path(args.manifest).resolve()),
        "num_trials": len(rows),
        "labels": dict(sorted(Counter(label for _, _, label, _ in rows).items())),
        "conditions": dict(sorted(condition_counts.items())),
        "channel_pairs": dict(sorted(channel_counts.items())),
        "target_count": kept_targets,
        "nontarget_count": kept_targets * args.nontarget_ratio,
        "nontarget_ratio_requested": args.nontarget_ratio,
        "candidate_counts": dict(sorted(candidate_counts.items())),
        "selected_target_counts": dict(sorted(selected_counts.items())),
        "skipped_duplicate_targets": skipped_dupe_target,
        "skipped_insufficient_impostors": skipped_short_impostor,
        "clean_segment_count": len(clean_segments),
        "bucket_segment_counts": {k: len(v) for k, v in sorted(bucketed.items())},
        "num_source_segments": len(segments),
        "num_speakers": len({s.speaker_id for s in segments}),
        "unique_audio_count": len({utt for row in rows for utt in row[:2]}),
        "nontarget_source_counts": dict(sorted(nontarget_source_counts.items())),
        "thresholds": {
            "clean_max_overlap": args.clean_max_overlap,
            "light_max_overlap": args.light_max_overlap,
            "mid_max_overlap": args.mid_max_overlap,
        },
    }

    stats_path = Path(args.output_stats).resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n")

    print("=== [CHiME-6 overlap] Trial generation done ===")
    print(f"  source segments : {stats['num_source_segments']}")
    print(f"  clean segments  : {stats['clean_segment_count']}")
    print(f"  targets         : {stats['target_count']}")
    print(f"  labels          : {stats['labels']}")
    print(f"  conditions      : {stats['conditions']}")
    print(f"  channel_pairs   : {stats['channel_pairs']}")


if __name__ == "__main__":
    main()
