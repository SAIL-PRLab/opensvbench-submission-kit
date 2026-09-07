#!/usr/bin/env python3
"""Prepare FFSVC 2022 dev trials for OpenSVBench.

The public dev release provides one official trial file and a metadata table.
OpenSVBench keeps the official pair list, converts labels to the shared
target/nontarget convention, and exports several ranked robustness views.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


META_RE = re.compile(
    r"^(?P<visit>[FST])"
    r"(?P<speaker_a>\d+)_"
    r"(?P<speaker_b>\d+)"
    r"(?P<device>PAD|I)"
    r"(?P<distance>-?\d+(?:\.\d+)?M|[LR]\d+M)_"
    r"(?P<session>\d+)_"
    r"(?P<text>\d+)_"
    r"(?P<style>[A-Za-z]+)$"
)

DEVICE_LABELS = {
    "I": "phone",
    "PAD": "tablet",
}

DISTANCE_LABELS = {
    "0.25M": "close_0p25m",
    "1M": "far_1m",
    "3M": "far_3m",
    "5M": "far_5m",
    "L3M": "far_left_3m",
    "R3M": "far_right_3m",
    "-1.5M": "far_back_1p5m",
}

LABELS = {
    "1": "target",
    "0": "nontarget",
    "target": "target",
    "nontarget": "nontarget",
}


@dataclass(frozen=True)
class UtteranceMeta:
    utt_id: str
    original_name: str
    speaker_id: str
    visit: str
    device: str
    device_label: str
    distance: str
    distance_label: str
    capture_domain: str
    session: str
    text_id: str
    style: str
    wav_path: Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FFSVC 2022 dev-set manifests and ranked trials")
    parser.add_argument("--raw-dir", required=True, help="FFSVC2022 raw-data root, containing dev wavs")
    parser.add_argument("--download-dir", required=True, help="Directory containing trials_dev_keys and dev_meta_list")
    parser.add_argument("--prep-dir", required=True, help="Output prepared directory")
    parser.add_argument("--split", default="dev", help="FFSVC split to prepare; currently dev is expected")
    parser.add_argument("--trial-file", default=None, help="Optional explicit trials_dev_keys path")
    parser.add_argument("--meta-file", default=None, help="Optional explicit dev_meta_list path")
    return parser.parse_args()


def strip_wav_suffix(utt_id: str) -> str:
    return utt_id[:-4] if utt_id.endswith(".wav") else utt_id


def parse_meta_line(original_name: str, ffsvc_name: str, wav_path: Path) -> UtteranceMeta:
    match = META_RE.match(original_name)
    if not match:
        raise ValueError(f"Cannot parse FFSVC original_name: {original_name}")
    groups = match.groupdict()
    speaker_a = int(groups["speaker_a"])
    speaker_b = int(groups["speaker_b"])
    if speaker_a != speaker_b:
        raise ValueError(
            f"Speaker id mismatch in {original_name}: {groups['speaker_a']} vs {groups['speaker_b']}"
        )

    device = groups["device"]
    distance = groups["distance"]
    return UtteranceMeta(
        utt_id=strip_wav_suffix(ffsvc_name),
        original_name=original_name,
        speaker_id=f"spk{speaker_a:04d}",
        visit=groups["visit"],
        device=device,
        device_label=DEVICE_LABELS.get(device, device.lower()),
        distance=distance,
        distance_label=DISTANCE_LABELS.get(distance, distance.lower().replace(".", "p")),
        capture_domain="close" if distance == "0.25M" else "far",
        session=groups["session"],
        text_id=groups["text"],
        style=groups["style"].lower(),
        wav_path=wav_path,
    )


def find_wav_path(raw_dir: Path, split: str, utt_id: str) -> Path | None:
    candidates = [
        raw_dir / split / f"{utt_id}.wav",
        raw_dir / f"{utt_id}.wav",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def read_metadata(meta_path: Path, raw_dir: Path, split: str) -> dict[str, UtteranceMeta]:
    metadata: dict[str, UtteranceMeta] = {}
    missing_wav = 0
    with meta_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter=" ")
        for row in reader:
            original_name = row.get("Original_Name", "").strip()
            ffsvc_name = row.get("FFSVC2022_Name", "").strip()
            if not original_name or not ffsvc_name:
                continue
            utt_id = strip_wav_suffix(ffsvc_name)
            wav_path = find_wav_path(raw_dir, split, utt_id)
            if wav_path is None:
                missing_wav += 1
                continue
            metadata[utt_id] = parse_meta_line(original_name, ffsvc_name, wav_path)
    if missing_wav:
        print(f"[FFSVC2022] Skipped {missing_wav} metadata rows with missing wav files.")
    if not metadata:
        raise ValueError(f"No usable metadata rows found in {meta_path}")
    return metadata


def read_trials(trial_path: Path, metadata: dict[str, UtteranceMeta]) -> tuple[list[dict], int]:
    rows: list[dict] = []
    skipped = 0
    with trial_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            raw_label, utt1_raw, utt2_raw = parts[:3]
            utt1 = strip_wav_suffix(utt1_raw)
            utt2 = strip_wav_suffix(utt2_raw)
            if raw_label not in LABELS:
                raise ValueError(f"Unexpected FFSVC label {raw_label!r} in {trial_path}")
            if utt1 not in metadata or utt2 not in metadata:
                skipped += 1
                continue
            rows.append({"utt1": utt1, "utt2": utt2, "label": LABELS[raw_label]})
    if not rows:
        raise ValueError(f"No usable trials found in {trial_path}")
    return rows, skipped


def write_mapping(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w") as f:
        for key, value in sorted(pairs):
            f.write(f"{key} {value}\n")


def write_spk2utt(path: Path, metadata: dict[str, UtteranceMeta]) -> None:
    grouped: dict[str, list[str]] = defaultdict(list)
    for meta in metadata.values():
        grouped[meta.speaker_id].append(meta.utt_id)
    with path.open("w") as f:
        for spk_id in sorted(grouped):
            f.write(f"{spk_id} {' '.join(sorted(grouped[spk_id]))}\n")


def write_trial_file(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
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
        "visit1",
        "visit2",
        "device1",
        "device2",
        "distance1",
        "distance2",
        "text1",
        "text2",
        "style1",
        "style2",
        "condition_cross_channel",
        "condition_cross_domain",
        "text_relation",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def label_counts(rows: list[tuple[str, str, str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row[2] for row in rows).items()))


def condition_counts(rows: list[tuple[str, str, str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row[3] for row in rows).items()))


def build_trials(
    official_rows: list[dict],
    metadata: dict[str, UtteranceMeta],
) -> tuple[dict[str, list[tuple[str, str, str, str]]], list[dict]]:
    outputs: dict[str, list[tuple[str, str, str, str]]] = {
        "cross_channel": [],
        "cross_domain": [],
    }
    pair_meta_rows: list[dict] = []

    for row in official_rows:
        utt1 = row["utt1"]
        utt2 = row["utt2"]
        label = row["label"]
        m1 = metadata[utt1]
        m2 = metadata[utt2]

        cross_channel_condition = ""
        if m1.device_label == "phone" and m2.device_label in {"phone", "tablet"}:
            cross_channel_condition = f"enroll_phone_test_{m2.device_label}"
            outputs["cross_channel"].append((utt1, utt2, label, cross_channel_condition))

        cross_domain_condition = ""
        if m1.distance == "0.25M" and m2.distance != "0.25M":
            cross_domain_condition = f"close_to_{m2.distance_label}"
            outputs["cross_domain"].append((utt1, utt2, label, cross_domain_condition))

        pair_meta_rows.append(
            {
                "utt1": utt1,
                "utt2": utt2,
                "label": label,
                "speaker1": m1.speaker_id,
                "speaker2": m2.speaker_id,
                "visit1": m1.visit,
                "visit2": m2.visit,
                "device1": m1.device_label,
                "device2": m2.device_label,
                "distance1": m1.distance_label,
                "distance2": m2.distance_label,
                "text1": m1.text_id,
                "text2": m2.text_id,
                "style1": m1.style,
                "style2": m2.style,
                "condition_cross_channel": cross_channel_condition,
                "condition_cross_domain": cross_domain_condition,
                "text_relation": "same" if m1.text_id == m2.text_id else "different",
            }
        )

    return outputs, pair_meta_rows


def main() -> None:
    args = get_args()
    raw_dir = Path(args.raw_dir).resolve()
    download_dir = Path(args.download_dir).resolve()
    prep_dir = Path(args.prep_dir).resolve()
    prep_dir.mkdir(parents=True, exist_ok=True)

    trial_path = Path(args.trial_file).resolve() if args.trial_file else download_dir / "trials_dev_keys"
    meta_path = Path(args.meta_file).resolve() if args.meta_file else download_dir / "dev_meta_list"
    if not trial_path.is_file():
        raise FileNotFoundError(f"Missing FFSVC dev trial file: {trial_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing FFSVC dev metadata file: {meta_path}")

    metadata = read_metadata(meta_path, raw_dir, args.split)
    official_rows, skipped_trials = read_trials(trial_path, metadata)
    outputs, pair_meta_rows = build_trials(official_rows, metadata)

    write_mapping(prep_dir / "wav.scp", [(m.utt_id, str(m.wav_path)) for m in metadata.values()])
    write_mapping(prep_dir / "wav_all.scp", [(m.utt_id, str(m.wav_path)) for m in metadata.values()])
    write_mapping(prep_dir / "utt2spk", [(m.utt_id, m.speaker_id) for m in metadata.values()])
    write_mapping(prep_dir / "utt2visit", [(m.utt_id, m.visit) for m in metadata.values()])
    write_mapping(prep_dir / "utt2session", [(m.utt_id, m.visit) for m in metadata.values()])
    write_mapping(prep_dir / "utt2device", [(m.utt_id, m.device_label) for m in metadata.values()])
    write_mapping(prep_dir / "utt2distance", [(m.utt_id, m.distance_label) for m in metadata.values()])
    write_mapping(prep_dir / "utt2text", [(m.utt_id, m.text_id) for m in metadata.values()])
    write_mapping(prep_dir / "utt2style", [(m.utt_id, m.style) for m in metadata.values()])
    write_mapping(prep_dir / "utt2domain", [(m.utt_id, m.capture_domain) for m in metadata.values()])
    write_spk2utt(prep_dir / "spk2utt", metadata)

    trial_outputs = {
        "cross_channel": prep_dir / "trials_ffsvc2022_cross_channel.txt",
        "cross_domain": prep_dir / "trials_ffsvc2022_cross_domain.txt",
    }
    for name, rows in outputs.items():
        write_trial_file(trial_outputs[name], rows)

    write_pair_metadata(prep_dir / "pair_metadata_ffsvc2022.csv", pair_meta_rows)

    stats = {
        "source_trial_file": str(trial_path),
        "source_meta_file": str(meta_path),
        "split": args.split,
        "num_utterances": len(metadata),
        "num_speakers": len({m.speaker_id for m in metadata.values()}),
        "skipped_trials_missing_audio_or_meta": skipped_trials,
        "metadata": {
            "visits": dict(sorted(Counter(m.visit for m in metadata.values()).items())),
            "devices": dict(sorted(Counter(m.device_label for m in metadata.values()).items())),
            "distances": dict(sorted(Counter(m.distance_label for m in metadata.values()).items())),
            "styles": dict(sorted(Counter(m.style for m in metadata.values()).items())),
        },
        "trials": {
            name: {
                "path": str(trial_outputs[name]),
                "total": len(rows),
                "labels": label_counts(rows),
                "conditions": condition_counts(rows),
            }
            for name, rows in outputs.items()
        },
    }
    (prep_dir / "stats_ffsvc2022.json").write_text(json.dumps(stats, indent=2, ensure_ascii=True) + "\n")

    print("=== [FFSVC2022] Preparation Done ===")
    print(f"  utterances : {stats['num_utterances']}")
    print(f"  speakers   : {stats['num_speakers']}")
    if skipped_trials:
        print(f"  skipped official trials due to missing metadata/audio: {skipped_trials}")
    for name, rows in outputs.items():
        print(
            f"  {name:<14}: {len(rows)} trials "
            f"(labels={label_counts(rows)}, conditions={condition_counts(rows)})"
        )


if __name__ == "__main__":
    main()
