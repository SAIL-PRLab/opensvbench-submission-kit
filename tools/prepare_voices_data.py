#!/usr/bin/env python3
"""Prepare VOiCES devkit manifests for OpenSVBench."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


QUERY_RE = re.compile(
    r"^Lab41-SRI-VOiCES-"
    r"(?P<room>rm\d)-"
    r"(?P<noise>[a-z]+)-"
    r"sp(?P<speaker>\d+)-"
    r"ch(?P<chapter>\d+)-"
    r"s(?:eg|g)(?P<segment>\d+)-"
    r"mc(?P<mic>\d+)-"
    r"(?P<mic_type>[a-z]+)-"
    r"(?P<mic_location>[a-z]+)-"
    r"dg(?P<degrees>\d+)$"
)


@dataclass(frozen=True)
class VoiceRecord:
    utt_id: str
    query_name: str
    speaker_id: str
    gender: str
    room: str
    noise: str
    mic: str
    mic_type: str
    mic_location: str
    degrees: int
    duration_sec: float
    source_segment: str
    source_relpath: str
    transcript: str
    wav_path: str


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VOiCES devkit manifests for OpenSVBench")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--prep-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max-utts", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def resolve_dataset_root(raw_dir: Path) -> Path:
    nested = raw_dir / "VOiCES_devkit"
    if (nested / "references" / "test_index.csv").is_file():
        return nested
    if (raw_dir / "references" / "test_index.csv").is_file():
        return raw_dir
    raise FileNotFoundError(f"Cannot find VOiCES_devkit under {raw_dir}")


def resolve_relpath(dataset_root: Path, relpath: str) -> Path:
    path = dataset_root / relpath
    if path.is_file():
        return path.resolve()
    if path.suffix != ".wav" and path.with_suffix(".wav").is_file():
        return path.with_suffix(".wav").resolve()
    raise FileNotFoundError(f"Missing VOiCES file referenced by index: {relpath}")


def parse_query_name(query_name: str) -> dict[str, str]:
    match = QUERY_RE.match(query_name)
    if not match:
        raise ValueError(f"Unexpected VOiCES query_name: {query_name}")
    return match.groupdict()


def load_records(dataset_root: Path, split: str) -> list[VoiceRecord]:
    index_path = dataset_root / "references" / f"{split}_index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing VOiCES index file: {index_path}")

    records: list[VoiceRecord] = []
    with index_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_name = row["query_name"].strip()
            parsed = parse_query_name(query_name)
            wav_path = resolve_relpath(dataset_root, row["filename"].strip())
            source_path = resolve_relpath(dataset_root, row["source"].strip())
            speaker_id = f"sp{int(row['speaker']):04d}"
            if speaker_id != f"sp{int(parsed['speaker']):04d}":
                raise ValueError(
                    f"Speaker mismatch for {query_name}: index={row['speaker']} filename={parsed['speaker']}"
                )
            records.append(
                VoiceRecord(
                    utt_id=query_name,
                    query_name=query_name,
                    speaker_id=speaker_id,
                    gender=row["gender"].strip(),
                    room=parsed["room"],
                    noise=parsed["noise"],
                    mic=f"mc{int(parsed['mic']):02d}",
                    mic_type=parsed["mic_type"],
                    mic_location=parsed["mic_location"],
                    degrees=int(parsed["degrees"]),
                    duration_sec=float(row["noisy_time"]),
                    source_segment=source_path.stem,
                    source_relpath=row["source"].strip(),
                    transcript=row["transcript"].strip(),
                    wav_path=str(wav_path),
                )
            )
    if not records:
        raise ValueError(f"No VOiCES records found in {index_path}")
    return records


def maybe_cap_records(records: list[VoiceRecord], max_utts: int, seed: int) -> list[VoiceRecord]:
    if max_utts <= 0 or len(records) <= max_utts:
        return records

    rng = random.Random(seed)
    grouped: dict[str, list[VoiceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.speaker_id].append(record)

    for items in grouped.values():
        rng.shuffle(items)

    active = sorted(grouped)
    selected: list[VoiceRecord] = []
    while active and len(selected) < max_utts:
        next_active: list[str] = []
        for speaker_id in active:
            if len(selected) >= max_utts:
                break
            items = grouped[speaker_id]
            if items:
                selected.append(items.pop())
            if items:
                next_active.append(speaker_id)
        active = next_active
    return selected


def write_mapping(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w") as f:
        for key, value in sorted(pairs):
            f.write(f"{key} {value}\n")


def write_spk2utt(path: Path, records: list[VoiceRecord]) -> None:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record.speaker_id].append(record.utt_id)
    with path.open("w") as f:
        for speaker_id in sorted(grouped):
            f.write(f"{speaker_id} {' '.join(sorted(grouped[speaker_id]))}\n")


def write_manifest(path: Path, records: list[VoiceRecord]) -> None:
    fieldnames = list(asdict(records[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item.utt_id):
            writer.writerow(asdict(record))


def main() -> None:
    args = get_args()
    raw_dir = Path(args.raw_dir).resolve()
    prep_dir = Path(args.prep_dir).resolve()
    prep_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = resolve_dataset_root(raw_dir)
    records = load_records(dataset_root, args.split)
    records = maybe_cap_records(records, args.max_utts, args.seed)

    write_mapping(prep_dir / "wav.scp", [(record.utt_id, record.wav_path) for record in records])
    write_mapping(prep_dir / "wav_all.scp", [(record.utt_id, record.wav_path) for record in records])
    write_mapping(prep_dir / "utt2spk", [(record.utt_id, record.speaker_id) for record in records])
    write_mapping(prep_dir / "utt2lang", [(record.utt_id, "en") for record in records])
    write_mapping(prep_dir / "utt2gender", [(record.utt_id, record.gender) for record in records])
    write_mapping(prep_dir / "utt2room", [(record.utt_id, record.room) for record in records])
    write_mapping(prep_dir / "utt2noise", [(record.utt_id, record.noise) for record in records])
    write_mapping(prep_dir / "utt2channel", [(record.utt_id, record.mic) for record in records])
    write_mapping(prep_dir / "utt2mic", [(record.utt_id, record.mic) for record in records])
    write_mapping(prep_dir / "utt2mic_type", [(record.utt_id, record.mic_type) for record in records])
    write_mapping(prep_dir / "utt2mic_location", [(record.utt_id, record.mic_location) for record in records])
    write_mapping(prep_dir / "utt2source_segment", [(record.utt_id, record.source_segment) for record in records])
    write_mapping(prep_dir / "utt2text", [(record.utt_id, record.transcript) for record in records])
    write_mapping(prep_dir / "utt2duration", [(record.utt_id, f"{record.duration_sec:.6f}") for record in records])
    write_spk2utt(prep_dir / "spk2utt", records)
    write_manifest(prep_dir / "voices_manifest.csv", records)

    canonical_easy = sum(
        1 for record in records if record.room == "rm1" and record.noise == "none" and record.mic == "mc01"
    )
    stats = {
        "split": args.split,
        "dataset_root": str(dataset_root),
        "num_utterances": len(records),
        "num_speakers": len({record.speaker_id for record in records}),
        "num_source_segments": len({record.source_segment for record in records}),
        "canonical_easy_records": canonical_easy,
        "rooms": dict(sorted(Counter(record.room for record in records).items())),
        "noises": dict(sorted(Counter(record.noise for record in records).items())),
        "mics": dict(sorted(Counter(record.mic for record in records).items())),
        "mic_types": dict(sorted(Counter(record.mic_type for record in records).items())),
        "mic_locations": dict(sorted(Counter(record.mic_location for record in records).items())),
        "genders": dict(sorted(Counter(record.gender for record in records).items())),
        "max_utts": args.max_utts,
        "seed": args.seed,
    }
    (prep_dir / "stats_voices_source.json").write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n")

    print("=== [VOiCES] Source manifest preparation done ===")
    print(f"  utterances         : {stats['num_utterances']}")
    print(f"  speakers           : {stats['num_speakers']}")
    print(f"  source segments    : {stats['num_source_segments']}")
    print(f"  canonical easy utts: {stats['canonical_easy_records']}")
    print(f"  rooms              : {stats['rooms']}")
    print(f"  noises             : {stats['noises']}")
    print(f"  mics               : {stats['mics']}")


if __name__ == "__main__":
    main()
