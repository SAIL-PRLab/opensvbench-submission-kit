#!/usr/bin/env python3
"""Generate AliMeeting overlap-robustness trial sets.

This script keeps the task as speaker verification:
  - enroll side: clean far-field AliMeeting segments
  - test side  : far-field AliMeeting segments bucketed by overlap ratio

It writes three trial files:
  - overlap_light
  - overlap_mid
  - overlap_heavy
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prepare_alimeeting_data import find_dir, get_spk_from_tier_far, parse_textgrid


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AliMeeting overlap-robustness trials")
    parser.add_argument("--raw-dir", required=True, help="AliMeeting raw-data root")
    parser.add_argument("--utt2spk", required=True)
    parser.add_argument("--utt2channel", required=True)
    parser.add_argument("--output-light", required=True)
    parser.add_argument("--output-mid", required=True)
    parser.add_argument("--output-heavy", required=True)
    parser.add_argument("--output-pair-metadata", default=None)
    parser.add_argument("--output-utt2session", default=None)
    parser.add_argument("--output-stats", default=None)
    parser.add_argument("--min-dur", type=float, default=3.0)
    parser.add_argument("--clean-max-overlap", type=float, default=0.05)
    parser.add_argument("--light-max-overlap", type=float, default=0.20)
    parser.add_argument("--mid-max-overlap", type=float, default=0.50)
    parser.add_argument("--max-target-trials", type=int, default=5000)
    parser.add_argument("--nontarget-ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def read_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def parse_alimeeting_utt_id(utt_id: str) -> tuple[str, str, str, int]:
    parts = utt_id.strip().split("_")
    if len(parts) < 6 or parts[0] != "alimeeting":
        raise ValueError(f"Unexpected AliMeeting utterance id: {utt_id}")
    channel = parts[1]
    speaker_id = parts[2]
    session_id = "_".join(parts[3:-1])
    seg_idx = int(parts[-1])
    return channel, speaker_id, session_id, seg_idx


def overlap_ratio(start: float, end: float, others: list[tuple[float, float]]) -> float:
    duration = end - start
    if duration <= 0.0:
        return 0.0
    intersections: list[tuple[float, float]] = []
    for other_start, other_end in others:
        seg_start = max(start, other_start)
        seg_end = min(end, other_end)
        if seg_end > seg_start:
            intersections.append((seg_start, seg_end))
    if not intersections:
        return 0.0
    intersections.sort()
    merged: list[tuple[float, float]] = []
    current_start, current_end = intersections[0]
    for seg_start, seg_end in intersections[1:]:
        if seg_start <= current_end:
            current_end = max(current_end, seg_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = seg_start, seg_end
    merged.append((current_start, current_end))
    return sum(seg_end - seg_start for seg_start, seg_end in merged) / duration


def bucket_name(
    ratio: float,
    clean_max_overlap: float,
    light_max_overlap: float,
    mid_max_overlap: float,
) -> str:
    if ratio < clean_max_overlap:
        return "clean"
    if ratio < light_max_overlap:
        return "overlap_light"
    if ratio < mid_max_overlap:
        return "overlap_mid"
    return "overlap_heavy"


def locate_far_textgrid_dir(raw_dir: Path) -> Path:
    far_tg = find_dir(
        raw_dir,
        "Test_Ali/Test_Ali_far/textgrid_dir",
        "Test_Ali_far/textgrid_dir",
    )
    if far_tg is None:
        raise FileNotFoundError(f"Cannot find AliMeeting far-field TextGrid dir under {raw_dir}")
    return far_tg


def build_far_overlap_index(
    raw_dir: Path,
    min_dur: float,
    clean_max_overlap: float,
    light_max_overlap: float,
    mid_max_overlap: float,
) -> dict[str, dict[str, Any]]:
    far_tg_dir = locate_far_textgrid_dir(raw_dir)
    utt_info: dict[str, dict[str, Any]] = {}
    tg_files = sorted(far_tg_dir.glob("*.TextGrid"))
    print(f"[AliMeeting overlap] Scanning far-field TextGrids: {len(tg_files)} files ...")

    for tg_path in tg_files:
        session_id = tg_path.stem
        segments = parse_textgrid(tg_path)
        per_tier: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for tier_name, start, end, text in segments:
            if not text.strip():
                continue
            per_tier[tier_name].append((float(start), float(end)))

        for tier_name, tier_intervals in per_tier.items():
            speaker_id = get_spk_from_tier_far(tier_name)
            other_intervals = [
                (start, end)
                for other_tier_name, other_tier_intervals in per_tier.items()
                if other_tier_name != tier_name
                for start, end in other_tier_intervals
            ]

            seg_idx = 0
            for start, end in tier_intervals:
                duration = float(end - start)
                if duration < min_dur:
                    continue
                utt_id = f"alimeeting_far_{speaker_id}_{session_id}_{seg_idx:04d}"
                ratio = overlap_ratio(start, end, other_intervals)
                utt_info[utt_id] = {
                    "utt_id": utt_id,
                    "speaker_id": speaker_id,
                    "session_id": session_id,
                    "start": float(start),
                    "end": float(end),
                    "duration": duration,
                    "overlap_ratio": ratio,
                    "overlap_bucket": bucket_name(
                        ratio,
                        clean_max_overlap=clean_max_overlap,
                        light_max_overlap=light_max_overlap,
                        mid_max_overlap=mid_max_overlap,
                    ),
                }
                seg_idx += 1

    print(f"[AliMeeting overlap] Indexed {len(utt_info)} far-field utterances")
    return utt_info


def round_robin_sample(
    grouped_items: dict[str, list[tuple[str, str]]],
    limit: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    prepared: dict[str, list[tuple[str, str]]] = {}
    active_groups: list[str] = []
    for group, items in grouped_items.items():
        copied = list(items)
        rng.shuffle(copied)
        if copied:
            prepared[group] = copied
            active_groups.append(group)

    if limit <= 0:
        limit = sum(len(items) for items in prepared.values())

    total_available = sum(len(items) for items in prepared.values())
    if total_available <= limit:
        selected: list[tuple[str, str]] = []
        for group in sorted(prepared):
            selected.extend(prepared[group])
        return selected

    selected: list[tuple[str, str]] = []
    while active_groups and len(selected) < limit:
        next_active: list[str] = []
        for group in active_groups:
            if len(selected) >= limit:
                break
            items = prepared[group]
            if items:
                selected.append(items.pop())
            if items:
                next_active.append(group)
        active_groups = next_active
    return selected


def write_trial_file(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for utt1, utt2, label, condition in rows:
            f.write(f"{utt1} {utt2} {label} {condition}\n")


def write_pair_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    fieldnames = [
        "utt1",
        "utt2",
        "speaker1",
        "speaker2",
        "session1",
        "session2",
        "source",
        "overlap_bucket",
        "test_overlap_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_utt2session(path: Path, utt_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for utt_id in sorted(utt_ids):
            try:
                _channel, _speaker_id, session_id, _seg_idx = parse_alimeeting_utt_id(utt_id)
            except ValueError:
                continue
            f.write(f"{utt_id} {session_id}\n")


def main() -> None:
    args = get_args()
    rng = random.Random(args.seed)

    utt2spk = read_mapping(Path(args.utt2spk))
    utt2channel = read_mapping(Path(args.utt2channel))
    all_utt_ids = sorted(set(utt2spk) & set(utt2channel))
    far_utt_ids = sorted(utt_id for utt_id in all_utt_ids if utt2channel.get(utt_id) == "far")

    if args.output_utt2session:
        write_utt2session(Path(args.output_utt2session), all_utt_ids)

    utt_info = build_far_overlap_index(
        raw_dir=Path(args.raw_dir),
        min_dur=args.min_dur,
        clean_max_overlap=args.clean_max_overlap,
        light_max_overlap=args.light_max_overlap,
        mid_max_overlap=args.mid_max_overlap,
    )

    missing_far = [utt_id for utt_id in far_utt_ids if utt_id not in utt_info]
    if missing_far:
        raise ValueError(
            f"{len(missing_far)} far-field utterances were not found in overlap index; "
            f"examples: {missing_far[:5]}"
        )

    clean_by_speaker: dict[str, list[str]] = defaultdict(list)
    clean_all: list[str] = []
    test_by_bucket_speaker: dict[str, dict[str, list[str]]] = {
        "overlap_light": defaultdict(list),
        "overlap_mid": defaultdict(list),
        "overlap_heavy": defaultdict(list),
    }
    test_by_bucket_session: dict[str, dict[str, dict[str, list[str]]]] = {
        "overlap_light": defaultdict(lambda: defaultdict(list)),
        "overlap_mid": defaultdict(lambda: defaultdict(list)),
        "overlap_heavy": defaultdict(lambda: defaultdict(list)),
    }
    clean_by_session: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    bucket_counts = Counter()
    for utt_id in far_utt_ids:
        info = utt_info[utt_id]
        speaker_id = info["speaker_id"]
        session_id = info["session_id"]
        bucket = info["overlap_bucket"]
        bucket_counts[bucket] += 1
        if bucket == "clean":
            clean_by_speaker[speaker_id].append(utt_id)
            clean_by_session[session_id][speaker_id].append(utt_id)
            clean_all.append(utt_id)
        else:
            test_by_bucket_speaker[bucket][speaker_id].append(utt_id)
            test_by_bucket_session[bucket][session_id][speaker_id].append(utt_id)

    print("[AliMeeting overlap] Far utterance buckets:")
    for bucket_name_key in ("clean", "overlap_light", "overlap_mid", "overlap_heavy"):
        print(f"  {bucket_name_key}: {bucket_counts[bucket_name_key]}")

    outputs = {
        "overlap_light": Path(args.output_light),
        "overlap_mid": Path(args.output_mid),
        "overlap_heavy": Path(args.output_heavy),
    }

    pair_metadata_rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "clean_threshold_max": args.clean_max_overlap,
        "light_threshold_max": args.light_max_overlap,
        "mid_threshold_max": args.mid_max_overlap,
        "min_duration": args.min_dur,
        "bucket_counts": dict(bucket_counts),
        "trials": {},
    }

    for bucket, output_path in outputs.items():
        grouped_targets: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for speaker_id, clean_utts in clean_by_speaker.items():
            bucket_utts = test_by_bucket_speaker[bucket].get(speaker_id, [])
            if not bucket_utts:
                continue
            for enroll_utt in clean_utts:
                for test_utt in bucket_utts:
                    grouped_targets[speaker_id].append((enroll_utt, test_utt))

        target_pairs = round_robin_sample(grouped_targets, args.max_target_trials, rng)
        required_nontargets = len(target_pairs) * args.nontarget_ratio

        seen_pairs = {tuple(sorted(pair)) for pair in target_pairs}
        nontarget_pairs: list[tuple[str, str, str]] = []

        grouped_same_session_nt: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for session_id, clean_spk_map in clean_by_session.items():
            bucket_spk_map = test_by_bucket_session[bucket].get(session_id, {})
            if not bucket_spk_map:
                continue
            for enroll_speaker_id, clean_utts in clean_spk_map.items():
                for test_speaker_id, test_utts in bucket_spk_map.items():
                    if enroll_speaker_id == test_speaker_id:
                        continue
                    for enroll_utt in clean_utts:
                        for test_utt in test_utts:
                            pair_key = tuple(sorted((enroll_utt, test_utt)))
                            if pair_key in seen_pairs:
                                continue
                            grouped_same_session_nt[session_id].append((enroll_utt, test_utt))

        sampled_same_session = round_robin_sample(grouped_same_session_nt, required_nontargets, rng)
        for enroll_utt, test_utt in sampled_same_session:
            pair_key = tuple(sorted((enroll_utt, test_utt)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            nontarget_pairs.append((enroll_utt, test_utt, "same_session"))

        bucket_test_all = [
            utt_id
            for speaker_map in test_by_bucket_speaker[bucket].values()
            for utt_id in speaker_map
        ]
        attempts = 0
        max_attempts = max(required_nontargets * 200, 1000)
        while len(nontarget_pairs) < required_nontargets and attempts < max_attempts:
            attempts += 1
            enroll_utt = rng.choice(clean_all)
            test_utt = rng.choice(bucket_test_all)
            if utt2spk[enroll_utt] == utt2spk[test_utt]:
                continue
            if utt_info[enroll_utt]["session_id"] == utt_info[test_utt]["session_id"]:
                continue
            pair_key = tuple(sorted((enroll_utt, test_utt)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            nontarget_pairs.append((enroll_utt, test_utt, "cross_session"))

        if len(nontarget_pairs) < required_nontargets:
            raise RuntimeError(
                f"Could not sample enough nontarget pairs for {bucket}: "
                f"{len(nontarget_pairs)} < {required_nontargets}"
            )

        rows: list[tuple[str, str, str, str]] = []
        for enroll_utt, test_utt in sorted(target_pairs):
            rows.append((enroll_utt, test_utt, "target", bucket))
            pair_metadata_rows.append(
                {
                    "utt1": enroll_utt,
                    "utt2": test_utt,
                    "speaker1": utt2spk[enroll_utt],
                    "speaker2": utt2spk[test_utt],
                    "session1": utt_info[enroll_utt]["session_id"],
                    "session2": utt_info[test_utt]["session_id"],
                    "source": "same_speaker_clean_to_overlap",
                    "overlap_bucket": bucket,
                    "test_overlap_ratio": f"{utt_info[test_utt]['overlap_ratio']:.6f}",
                }
            )
        for enroll_utt, test_utt, source in sorted(nontarget_pairs):
            rows.append((enroll_utt, test_utt, "nontarget", bucket))
            pair_metadata_rows.append(
                {
                    "utt1": enroll_utt,
                    "utt2": test_utt,
                    "speaker1": utt2spk[enroll_utt],
                    "speaker2": utt2spk[test_utt],
                    "session1": utt_info[enroll_utt]["session_id"],
                    "session2": utt_info[test_utt]["session_id"],
                    "source": f"{source}_nontarget",
                    "overlap_bucket": bucket,
                    "test_overlap_ratio": f"{utt_info[test_utt]['overlap_ratio']:.6f}",
                }
            )

        write_trial_file(output_path, rows)
        target_count = len(target_pairs)
        nontarget_count = len(nontarget_pairs)
        same_session_count = sum(1 for _u1, _u2, source in nontarget_pairs if source == "same_session")
        cross_session_count = nontarget_count - same_session_count
        eligible_speakers = sum(
            1
            for speaker_id in clean_by_speaker
            if clean_by_speaker[speaker_id] and test_by_bucket_speaker[bucket].get(speaker_id)
        )
        stats["trials"][bucket] = {
            "output_path": str(output_path),
            "target_count": target_count,
            "nontarget_count": nontarget_count,
            "same_session_nontarget_count": same_session_count,
            "cross_session_nontarget_count": cross_session_count,
            "eligible_speakers": eligible_speakers,
        }
        print(
            f"[AliMeeting overlap] {bucket}: "
            f"target={target_count} nontarget={nontarget_count} "
            f"(same_session={same_session_count}, cross_session={cross_session_count})"
        )

    if args.output_pair_metadata:
        write_pair_metadata(Path(args.output_pair_metadata), pair_metadata_rows)
    if args.output_stats:
        stats_path = Path(args.output_stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
