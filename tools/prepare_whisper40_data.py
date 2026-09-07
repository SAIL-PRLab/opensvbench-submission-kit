#!/usr/bin/env python3
"""Prepare Whisper40 manifests for OpenSVBench."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


STYLES = ("nature", "whsp")
SPLITS = ("enroll", "test")


@dataclass(frozen=True)
class UtteranceMeta:
    utt_id: str
    speaker_id: str
    style: str
    text_id: str
    split: str
    wav_path: Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Whisper40 manifests for OpenSVBench")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--prep-dir", required=True)
    parser.add_argument("--max-utts", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def resolve_dataset_root(raw_dir: Path) -> Path:
    nested = raw_dir / "whisper40"
    if (nested / "nature").is_dir() and (nested / "whsp").is_dir():
        return nested
    if (raw_dir / "nature").is_dir() and (raw_dir / "whsp").is_dir():
        return raw_dir
    raise FileNotFoundError(f"Cannot find Whisper40 nature/whsp folders under {raw_dir}")


def parse_wav_name(wav_path: Path, style: str, split: str) -> UtteranceMeta:
    stem = wav_path.stem.lower()
    if stem.endswith("_dao"):
        raise ValueError(f"DAO variant should be filtered before parsing: {wav_path.name}")
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected Whisper40 wav filename: {wav_path.name}")
    speaker_id = parts[0]
    suffix_style = parts[-1]
    text_id = "_".join(parts[1:-1])
    if suffix_style != style:
        raise ValueError(f"Style mismatch for {wav_path.name}: dir={style} file={suffix_style}")
    if not text_id:
        raise ValueError(f"Missing text id in Whisper40 filename: {wav_path.name}")
    return UtteranceMeta(
        utt_id=f"whisper40_{split}_{stem}",
        speaker_id=speaker_id,
        style=style,
        text_id=text_id,
        split=split,
        wav_path=wav_path.resolve(),
    )


def read_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def scan_split_from_manifest(split_dir: Path, style: str, split: str) -> list[UtteranceMeta]:
    wav_scp = split_dir / "wav.scp"
    utt2spk_path = split_dir / "utt2spk"
    if not wav_scp.is_file() or not utt2spk_path.is_file():
        return []

    utt2spk = read_mapping(utt2spk_path)
    records: list[UtteranceMeta] = []
    missing_paths: list[str] = []
    for utt_id, raw_path in read_mapping(wav_scp).items():
        if utt_id.endswith("_dao"):
            continue
        basename = Path(raw_path).name
        local_path = split_dir / utt2spk[utt_id] / basename
        if not local_path.is_file():
            missing_paths.append(str(local_path))
            continue
        parsed = parse_wav_name(local_path, style=style, split=split)
        if parsed.speaker_id != utt2spk[utt_id]:
            raise ValueError(
                f"Speaker mismatch for {utt_id}: wav name gives {parsed.speaker_id}, "
                f"utt2spk gives {utt2spk[utt_id]}"
            )
        records.append(parsed)
    if missing_paths:
        preview = "\n".join(missing_paths[:10])
        raise FileNotFoundError(
            f"Whisper40 manifest listed missing local wav files under {split_dir}:\n{preview}"
        )
    return records


def scan_split_from_files(split_dir: Path, style: str, split: str) -> list[UtteranceMeta]:
    records: list[UtteranceMeta] = []
    for speaker_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        for wav_path in sorted(speaker_dir.glob("*.wav")):
            if wav_path.stem.lower().endswith("_dao"):
                continue
            records.append(parse_wav_name(wav_path, style=style, split=split))
    return records


def scan_records(dataset_root: Path) -> list[UtteranceMeta]:
    records: list[UtteranceMeta] = []
    for style in STYLES:
        for split in SPLITS:
            split_dir = dataset_root / style / split
            if not split_dir.is_dir():
                continue
            manifest_records = scan_split_from_manifest(split_dir, style=style, split=split)
            if manifest_records:
                records.extend(manifest_records)
                continue
            records.extend(scan_split_from_files(split_dir, style=style, split=split))
    if not records:
        raise ValueError(f"No Whisper40 wav files found under {dataset_root}")
    return records


def maybe_cap_records(records: list[UtteranceMeta], max_utts: int, seed: int) -> list[UtteranceMeta]:
    if max_utts <= 0 or len(records) <= max_utts:
        return records

    rng = random.Random(seed)
    grouped: dict[tuple[str, str, str], list[UtteranceMeta]] = defaultdict(list)
    for record in records:
        grouped[(record.speaker_id, record.style, record.split)].append(record)

    prepared: dict[tuple[str, str, str], list[UtteranceMeta]] = {}
    active: list[tuple[str, str, str]] = []
    for key, items in grouped.items():
        copied = list(items)
        rng.shuffle(copied)
        if copied:
            prepared[key] = copied
            active.append(key)

    selected: list[UtteranceMeta] = []
    while active and len(selected) < max_utts:
        next_active: list[tuple[str, str, str]] = []
        for key in active:
            if len(selected) >= max_utts:
                break
            items = prepared[key]
            if items:
                selected.append(items.pop())
            if items:
                next_active.append(key)
        active = next_active
    return sorted(selected, key=lambda item: item.utt_id)


def write_mapping(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w") as f:
        for key, value in sorted(pairs):
            f.write(f"{key} {value}\n")


def write_spk2utt(path: Path, records: list[UtteranceMeta]) -> None:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record.speaker_id].append(record.utt_id)
    with path.open("w") as f:
        for speaker_id in sorted(grouped):
            f.write(f"{speaker_id} {' '.join(sorted(grouped[speaker_id]))}\n")


def main() -> None:
    args = get_args()
    raw_dir = Path(args.raw_dir).resolve()
    prep_dir = Path(args.prep_dir).resolve()
    prep_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = resolve_dataset_root(raw_dir)
    records = scan_records(dataset_root)
    records = maybe_cap_records(records, args.max_utts, args.seed)

    write_mapping(prep_dir / "wav.scp", [(record.utt_id, str(record.wav_path)) for record in records])
    write_mapping(prep_dir / "wav_all.scp", [(record.utt_id, str(record.wav_path)) for record in records])
    write_mapping(prep_dir / "utt2spk", [(record.utt_id, record.speaker_id) for record in records])
    write_mapping(prep_dir / "utt2lang", [(record.utt_id, "zh") for record in records])
    write_mapping(prep_dir / "utt2style", [(record.utt_id, record.style) for record in records])
    write_mapping(prep_dir / "utt2text", [(record.utt_id, record.text_id) for record in records])
    write_mapping(prep_dir / "utt2split", [(record.utt_id, record.split) for record in records])
    write_spk2utt(prep_dir / "spk2utt", records)

    stats = {
        "num_utterances": len(records),
        "num_speakers": len({record.speaker_id for record in records}),
        "styles": dict(sorted(Counter(record.style for record in records).items())),
        "splits": dict(sorted(Counter(record.split for record in records).items())),
        "style_split_counts": {
            f"{style}_{split}": sum(1 for record in records if record.style == style and record.split == split)
            for style in STYLES
            for split in SPLITS
        },
        "max_utts": args.max_utts,
        "seed": args.seed,
    }
    (prep_dir / "stats_whisper40_source.json").write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n")

    print("=== [Whisper40] Source manifest preparation done ===")
    print(f"  utterances : {stats['num_utterances']}")
    print(f"  speakers   : {stats['num_speakers']}")
    print(f"  styles     : {stats['styles']}")
    print(f"  splits     : {stats['splits']}")


if __name__ == "__main__":
    main()
