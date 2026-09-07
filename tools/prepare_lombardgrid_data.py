#!/usr/bin/env python3
"""Prepare Lombard Grid audio manifests for OpenSVBench."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import wave
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


SIMPLE_RE = re.compile(
    r"^(?P<speaker>s\d+)_(?P<style>[lp])_(?P<text>[a-z0-9]+)(?:_WRONG_(?P<actual>[a-z0-9]+))?$",
    re.IGNORECASE,
)
EXTENDED_RE = re.compile(
    r"^(?P<speaker>s\d+)_(?P<session>\d+)_(?P<index>\d+)_(?P<subindex>\d+)_(?P<style>[lrp])_"
    r"(?P<text>[a-z0-9]+)(?:_WRONG_(?P<actual>[a-z0-9]+))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LombardRecord:
    utt_id: str
    speaker_id: str
    style: str
    style_code: str
    text: str
    actual_text: str
    session: str
    index: str
    subindex: str
    duration_sec: float
    wav_path: str
    source_relpath: str


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Lombard Grid manifests for OpenSVBench")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--prep-dir", required=True)
    parser.add_argument("--max-utts", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def parse_style(style_code: str) -> str:
    value = style_code.lower()
    if value == "l":
        return "lombard"
    if value in {"p", "r"}:
        return "plain"
    raise ValueError(f"Unexpected Lombard Grid style code: {style_code}")


def parse_audio_name(path: Path) -> dict[str, str] | None:
    stem = path.stem
    match = SIMPLE_RE.match(stem) or EXTENDED_RE.match(stem)
    if not match:
        return None
    data = {key: (value or "") for key, value in match.groupdict().items()}
    data["style"] = parse_style(data["style"])
    data["style_code"] = match.group("style").lower()
    data["actual_text"] = data.get("actual") or data["text"]
    data.setdefault("session", "")
    data.setdefault("index", "")
    data.setdefault("subindex", "")
    return data


def probe_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            sample_rate = wav.getframerate()
            if sample_rate > 0:
                return frames / float(sample_rate)
    except Exception:
        pass
    return 0.0


def collect_records(raw_dir: Path) -> list[LombardRecord]:
    records: list[LombardRecord] = []
    skipped = 0
    for path in sorted(raw_dir.rglob("*.wav")):
        parsed = parse_audio_name(path)
        if parsed is None:
            skipped += 1
            continue
        utt_id = path.stem
        records.append(
            LombardRecord(
                utt_id=utt_id,
                speaker_id=parsed["speaker"].lower(),
                style=parsed["style"],
                style_code=parsed["style_code"],
                text=parsed["text"].lower(),
                actual_text=parsed["actual_text"].lower(),
                session=parsed.get("session", ""),
                index=parsed.get("index", ""),
                subindex=parsed.get("subindex", ""),
                duration_sec=probe_duration(path),
                wav_path=str(path.resolve()),
                source_relpath=path.relative_to(raw_dir).as_posix(),
            )
        )
    if not records:
        raise ValueError(
            f"No parseable Lombard Grid wav files found under {raw_dir}. "
            "Expected filenames like s8_p_sbbi9p.wav or s8_l_sbbi9p.wav."
        )
    if skipped:
        print(f"[LombardGrid] Skipped {skipped} wav files with unexpected names.")
    return records


def maybe_cap_records(records: list[LombardRecord], max_utts: int, seed: int) -> list[LombardRecord]:
    if max_utts <= 0 or len(records) <= max_utts:
        return records
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[LombardRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.speaker_id, record.style)].append(record)
    for items in grouped.values():
        rng.shuffle(items)

    active = sorted(grouped)
    selected: list[LombardRecord] = []
    while active and len(selected) < max_utts:
        next_active = []
        for key in active:
            if len(selected) >= max_utts:
                break
            items = grouped[key]
            if items:
                selected.append(items.pop())
            if items:
                next_active.append(key)
        active = next_active
    return selected


def write_mapping(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w") as f:
        for key, value in sorted(pairs):
            f.write(f"{key} {value}\n")


def write_spk2utt(path: Path, records: list[LombardRecord]) -> None:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        grouped[record.speaker_id].append(record.utt_id)
    with path.open("w") as f:
        for speaker_id in sorted(grouped):
            f.write(f"{speaker_id} {' '.join(sorted(grouped[speaker_id]))}\n")


def write_manifest(path: Path, records: list[LombardRecord]) -> None:
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

    records = collect_records(raw_dir)
    records = maybe_cap_records(records, args.max_utts, args.seed)

    write_mapping(prep_dir / "wav.scp", [(record.utt_id, record.wav_path) for record in records])
    write_mapping(prep_dir / "wav_all.scp", [(record.utt_id, record.wav_path) for record in records])
    write_mapping(prep_dir / "utt2spk", [(record.utt_id, record.speaker_id) for record in records])
    write_mapping(prep_dir / "utt2lang", [(record.utt_id, "en") for record in records])
    write_mapping(prep_dir / "utt2style", [(record.utt_id, record.style) for record in records])
    write_mapping(prep_dir / "utt2text", [(record.utt_id, record.actual_text) for record in records])
    write_mapping(prep_dir / "utt2duration", [(record.utt_id, f"{record.duration_sec:.6f}") for record in records])
    write_mapping(prep_dir / "utt2session", [(record.utt_id, record.session) for record in records if record.session])
    write_spk2utt(prep_dir / "spk2utt", records)
    write_manifest(prep_dir / "lombardgrid_manifest.csv", records)

    stats = {
        "raw_dir": str(raw_dir),
        "num_utterances": len(records),
        "num_speakers": len({record.speaker_id for record in records}),
        "styles": dict(sorted(Counter(record.style for record in records).items())),
        "speakers_by_style": {
            style: len({record.speaker_id for record in records if record.style == style})
            for style in sorted({record.style for record in records})
        },
        "max_utts": args.max_utts,
        "seed": args.seed,
    }
    (prep_dir / "stats_lombardgrid_source.json").write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n")

    print("=== [LombardGrid] Source manifest preparation done ===")
    print(f"  utterances: {stats['num_utterances']}")
    print(f"  speakers  : {stats['num_speakers']}")
    print(f"  styles    : {stats['styles']}")


if __name__ == "__main__":
    main()
