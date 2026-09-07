#!/usr/bin/env python3
"""Prepare CHiME-6 dev utterance manifests for OpenSVBench."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


LOCATIONS = {"kitchen", "dining", "living"}


@dataclass(frozen=True)
class JsonSegment:
    session_id: str
    speaker_id: str
    start: float
    end: float
    words: str
    location: str
    ref: str
    index: int
    overlap_ratio: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class PreparedSegment:
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
    wearable_wav: str
    array_wav: str
    source_wearable_wav: str
    source_array_wav: str


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CHiME-6 dev data for OpenSVBench")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--prep-dir", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--overlap-threshold", type=float, default=0.20)
    parser.add_argument("--array-channel", type=int, default=1)
    parser.add_argument("--max-segments", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def parse_time(value: str) -> float:
    value = str(value).strip()
    if ":" not in value:
        return float(value)
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unexpected CHiME-6 time value: {value}")
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def speech_like(words: str) -> bool:
    text = re.sub(r"\[[^\]]+\]", " ", words or "")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).strip()
    return bool(text)


def find_split_dir(raw_dir: Path, split: str, parent_name: str) -> Path:
    candidates = [
        raw_dir / parent_name / split,
        raw_dir / "CHiME6" / parent_name / split,
        raw_dir / "CHiME5" / parent_name / split,
    ]
    if parent_name == "audio":
        candidates.extend(
            [
                raw_dir / "CHiME6_dev" / "CHiME6" / "audio" / split,
                raw_dir / "CHiME6_train" / "CHiME6" / "audio" / split,
            ]
        )
    elif parent_name == "transcriptions":
        candidates.extend(
            [
                raw_dir / "transcriptions" / "transcriptions" / split,
                raw_dir / "CHiME6_transcriptions" / "transcriptions" / split,
            ]
        )
    for path in candidates:
        if path.is_dir():
            return path.resolve()

    matches = sorted(
        path for path in raw_dir.rglob(split) if path.is_dir() and path.parent.name == parent_name
    )
    if not matches and parent_name == "audio":
        matches = sorted(
            path
            for path in raw_dir.rglob(split)
            if path.is_dir() and path.parent.name == "audio" and any(path.rglob("*.wav"))
        )
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(f"Cannot find CHiME-6 {parent_name}/{split} under {raw_dir}")


def load_json_segments(trans_dir: Path) -> dict[str, list[dict]]:
    sessions: dict[str, list[dict]] = {}
    for json_path in sorted(trans_dir.glob("*.json")):
        data = json.loads(json_path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"Unexpected JSON structure in {json_path}")
        session_id = json_path.stem
        sessions[session_id] = data
    if not sessions:
        raise ValueError(f"No CHiME-6 JSON files found under {trans_dir}")
    return sessions


def merge_intervals(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        total += max(0.0, cur_end - cur_start)
        cur_start, cur_end = start, end
    total += max(0.0, cur_end - cur_start)
    return total


def compute_overlap_ratio(start: float, end: float, speaker: str, intervals: list[tuple[str, float, float]]) -> float:
    overlaps: list[tuple[float, float]] = []
    for other_speaker, other_start, other_end in intervals:
        if other_speaker == speaker:
            continue
        inter_start = max(start, other_start)
        inter_end = min(end, other_end)
        if inter_end > inter_start:
            overlaps.append((inter_start, inter_end))
    duration = end - start
    if duration <= 0:
        return 0.0
    return merge_intervals(overlaps) / duration


def build_candidate_segments(
    sessions: dict[str, list[dict]],
    min_duration: float,
    max_duration: float,
    overlap_threshold: float,
) -> tuple[list[JsonSegment], dict]:
    candidates: list[JsonSegment] = []
    reject_counts = Counter()

    for session_id, rows in sessions.items():
        speech_intervals: list[tuple[str, float, float]] = []
        parsed_rows: list[tuple[int, dict, float, float]] = []
        for idx, row in enumerate(rows):
            try:
                start = parse_time(row["start_time"])
                end = parse_time(row["end_time"])
            except Exception:
                reject_counts["bad_time"] += 1
                continue
            if end <= start:
                reject_counts["bad_time"] += 1
                continue
            parsed_rows.append((idx, row, start, end))
            if row.get("speaker") and speech_like(row.get("words", "")):
                speech_intervals.append((str(row["speaker"]), start, end))

        for idx, row, start, end in parsed_rows:
            missing = [key for key in ("speaker", "location", "ref", "words") if not row.get(key)]
            if missing:
                reject_counts["missing_required_fields"] += 1
                continue
            location = str(row["location"]).strip().lower()
            if location not in LOCATIONS:
                reject_counts["unsupported_location"] += 1
                continue
            words = str(row["words"])
            if not speech_like(words):
                reject_counts["non_speech_text"] += 1
                continue
            duration = end - start
            if duration < min_duration:
                reject_counts["too_short"] += 1
                continue
            if duration > max_duration:
                reject_counts["too_long"] += 1
                continue
            speaker = str(row["speaker"]).strip()
            overlap_ratio = compute_overlap_ratio(start, end, speaker, speech_intervals)
            if overlap_ratio >= overlap_threshold:
                reject_counts["overlap"] += 1
                continue
            candidates.append(
                JsonSegment(
                    session_id=session_id,
                    speaker_id=speaker,
                    start=start,
                    end=end,
                    words=words.strip(),
                    location=location,
                    ref=str(row["ref"]).strip(),
                    index=idx,
                    overlap_ratio=overlap_ratio,
                )
            )

    stats = {
        "raw_json_utterances": sum(len(rows) for rows in sessions.values()),
        "reject_counts": dict(sorted(reject_counts.items())),
        "candidate_segments": len(candidates),
    }
    return candidates, stats


def balanced_limit(segments: list[JsonSegment], limit: int, seed: int) -> list[JsonSegment]:
    if limit <= 0 or len(segments) <= limit:
        return sorted(segments, key=lambda item: (item.session_id, item.speaker_id, item.index))

    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[JsonSegment]] = defaultdict(list)
    for segment in segments:
        grouped[(segment.speaker_id, segment.location)].append(segment)

    prepared: dict[tuple[str, str], list[JsonSegment]] = {}
    active: list[tuple[str, str]] = []
    for key, items in grouped.items():
        copied = list(items)
        rng.shuffle(copied)
        prepared[key] = copied
        active.append(key)
    rng.shuffle(active)

    selected: list[JsonSegment] = []
    while active and len(selected) < limit:
        next_active: list[tuple[str, str]] = []
        for key in active:
            if len(selected) >= limit:
                break
            items = prepared[key]
            if items:
                selected.append(items.pop())
            if items:
                next_active.append(key)
        active = next_active
    return sorted(selected, key=lambda item: (item.session_id, item.speaker_id, item.index))


def index_audio(audio_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(audio_dir.rglob("*.wav")):
        mapping[path.name] = path.resolve()
    if not mapping:
        raise ValueError(f"No wav files found under {audio_dir}")
    return mapping


def sox_extract(src: Path, dst: Path, start: float, duration: float, remix_channel: int | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["sox", str(src), "-c", "1", str(dst)]
    if remix_channel is not None:
        cmd.extend(["remix", str(remix_channel)])
    cmd.extend(["trim", f"{start:.3f}", f"{duration:.3f}"])
    subprocess.run(cmd, check=True, capture_output=True)


def write_mapping(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w") as f:
        for key, value in sorted(pairs):
            f.write(f"{key} {value}\n")


def write_spk2utt(path: Path, utt2spk: list[tuple[str, str]]) -> None:
    grouped: dict[str, list[str]] = defaultdict(list)
    for utt, spk in utt2spk:
        grouped[spk].append(utt)
    with path.open("w") as f:
        for spk in sorted(grouped):
            f.write(f"{spk} {' '.join(sorted(grouped[spk]))}\n")


def write_manifest(path: Path, segments: list[PreparedSegment]) -> None:
    fieldnames = list(asdict(segments[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for segment in segments:
            writer.writerow(asdict(segment))


def main() -> None:
    args = get_args()
    raw_dir = Path(args.raw_dir).resolve()
    prep_dir = Path(args.prep_dir).resolve()
    prep_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = find_split_dir(raw_dir, args.split, "audio")
    trans_dir = find_split_dir(raw_dir, args.split, "transcriptions")
    sessions = load_json_segments(trans_dir)
    audio_index = index_audio(audio_dir)

    candidates, source_stats = build_candidate_segments(
        sessions=sessions,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        overlap_threshold=args.overlap_threshold,
    )

    available: list[JsonSegment] = []
    missing_audio = Counter()
    for segment in candidates:
        wearable_name = f"{segment.session_id}_{segment.speaker_id}.wav"
        array_name = f"{segment.session_id}_{segment.ref}.CH{args.array_channel}.wav"
        if wearable_name not in audio_index:
            missing_audio["wearable"] += 1
            continue
        if array_name not in audio_index:
            missing_audio["array"] += 1
            continue
        available.append(segment)

    selected = balanced_limit(available, args.max_segments, args.seed)
    out_segment_dir = prep_dir / "segments"
    prepared: list[PreparedSegment] = []

    for segment in selected:
        segment_id = f"chime6_{args.split}_{segment.session_id}_{segment.speaker_id}_{segment.index:05d}"
        wearable_utt = f"{segment_id}_wearable"
        array_utt = f"{segment_id}_array"
        wearable_src = audio_index[f"{segment.session_id}_{segment.speaker_id}.wav"]
        array_src = audio_index[f"{segment.session_id}_{segment.ref}.CH{args.array_channel}.wav"]
        wearable_dst = out_segment_dir / "wearable" / f"{wearable_utt}.wav"
        array_dst = out_segment_dir / "array" / f"{array_utt}.wav"
        duration = segment.duration

        sox_extract(wearable_src, wearable_dst, segment.start, duration, remix_channel=1)
        sox_extract(array_src, array_dst, segment.start, duration, remix_channel=None)

        prepared.append(
            PreparedSegment(
                segment_id=segment_id,
                speaker_id=segment.speaker_id,
                session_id=segment.session_id,
                location=segment.location,
                ref_array=segment.ref,
                array_channel=f"CH{args.array_channel}",
                start=segment.start,
                end=segment.end,
                duration=duration,
                overlap_ratio=segment.overlap_ratio,
                text=segment.words,
                wearable_utt=wearable_utt,
                array_utt=array_utt,
                wearable_wav=str(wearable_dst.resolve()),
                array_wav=str(array_dst.resolve()),
                source_wearable_wav=str(wearable_src),
                source_array_wav=str(array_src),
            )
        )

    if not prepared:
        raise ValueError("No CHiME-6 segments were prepared.")

    wav_pairs: list[tuple[str, str]] = []
    utt2spk: list[tuple[str, str]] = []
    utt2lang: list[tuple[str, str]] = []
    utt2session: list[tuple[str, str]] = []
    utt2location: list[tuple[str, str]] = []
    utt2channel: list[tuple[str, str]] = []
    utt2channel_type: list[tuple[str, str]] = []
    utt2array_id: list[tuple[str, str]] = []
    utt2array_channel: list[tuple[str, str]] = []
    utt2overlap_ratio: list[tuple[str, str]] = []
    utt2text: list[tuple[str, str]] = []
    utt2duration: list[tuple[str, str]] = []
    utt2segment: list[tuple[str, str]] = []

    for segment in prepared:
        for utt, wav, channel_type, array_id, array_channel in (
            (segment.wearable_utt, segment.wearable_wav, "wearable", "none", "none"),
            (segment.array_utt, segment.array_wav, "array", segment.ref_array, segment.array_channel),
        ):
            wav_pairs.append((utt, wav))
            utt2spk.append((utt, segment.speaker_id))
            utt2lang.append((utt, "en"))
            utt2session.append((utt, segment.session_id))
            utt2location.append((utt, segment.location))
            utt2channel.append((utt, channel_type))
            utt2channel_type.append((utt, channel_type))
            utt2array_id.append((utt, array_id))
            utt2array_channel.append((utt, array_channel))
            utt2overlap_ratio.append((utt, f"{segment.overlap_ratio:.6f}"))
            utt2text.append((utt, segment.text))
            utt2duration.append((utt, f"{segment.duration:.6f}"))
            utt2segment.append((utt, segment.segment_id))

    write_mapping(prep_dir / "wav.scp", wav_pairs)
    write_mapping(prep_dir / "wav_all.scp", wav_pairs)
    write_mapping(prep_dir / "utt2spk", utt2spk)
    write_mapping(prep_dir / "utt2lang", utt2lang)
    write_mapping(prep_dir / "utt2session", utt2session)
    write_mapping(prep_dir / "utt2location", utt2location)
    write_mapping(prep_dir / "utt2channel", utt2channel)
    write_mapping(prep_dir / "utt2channel_type", utt2channel_type)
    write_mapping(prep_dir / "utt2array_id", utt2array_id)
    write_mapping(prep_dir / "utt2array_channel", utt2array_channel)
    write_mapping(prep_dir / "utt2overlap_ratio", utt2overlap_ratio)
    write_mapping(prep_dir / "utt2text", utt2text)
    write_mapping(prep_dir / "utt2duration", utt2duration)
    write_mapping(prep_dir / "utt2segment", utt2segment)
    write_spk2utt(prep_dir / "spk2utt", utt2spk)
    write_manifest(prep_dir / "chime6_manifest.csv", prepared)

    stats = {
        "split": args.split,
        "audio_dir": str(audio_dir),
        "transcriptions_dir": str(trans_dir),
        "num_source_segments": len(prepared),
        "num_utterances": len(wav_pairs),
        "num_speakers": len({segment.speaker_id for segment in prepared}),
        "sessions": dict(sorted(Counter(segment.session_id for segment in prepared).items())),
        "locations": dict(sorted(Counter(segment.location for segment in prepared).items())),
        "ref_arrays": dict(sorted(Counter(segment.ref_array for segment in prepared).items())),
        "channel_types": {"array": len(prepared), "wearable": len(prepared)},
        "duration_min": min(segment.duration for segment in prepared),
        "duration_max": max(segment.duration for segment in prepared),
        "overlap_threshold": args.overlap_threshold,
        "max_segments": args.max_segments,
        "missing_audio": dict(sorted(missing_audio.items())),
        **source_stats,
    }
    (prep_dir / "stats_chime6_source.json").write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n")

    print("=== [CHiME-6] Source manifest preparation done ===")
    print(f"  source segments : {stats['num_source_segments']}")
    print(f"  utterances      : {stats['num_utterances']}")
    print(f"  speakers        : {stats['num_speakers']}")
    print(f"  sessions        : {stats['sessions']}")
    print(f"  locations       : {stats['locations']}")


if __name__ == "__main__":
    main()
