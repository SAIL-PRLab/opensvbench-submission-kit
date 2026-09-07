#!/usr/bin/env python3
"""Check a few prepared-audio fingerprints used for submission debugging.

This is intentionally small: it does not validate the whole benchmark, only a
few utterances that are useful when diagnosing score mismatches between two
otherwise identical submissions.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REFERENCE_UTTERANCES = [
    {
        "dataset": "cnceleb",
        "utt_id": "eval-enroll-id00844-enroll-id00844-enroll",
        "sha256": "fa527170274842c97be04781b281b11ea6c127d7259ef9b50817da1ea8c55bd0",
        "duration": 20.547187,
    },
    {
        "dataset": "cnceleb",
        "utt_id": "eval-enroll-id00914-enroll-id00914-enroll",
        "sha256": "caf1ee5323d7c146c3adb056698d091ef64661a9918c2792862cd7bccc50187c",
        "duration": 20.456875,
    },
    {
        "dataset": "cnceleb",
        "utt_id": "eval-enroll-id00800-enroll-id00800-enroll",
        "sha256": "3c94e03c2b1830f27c70247a30f7c81dc5c5aa7735570843825bf68641417109",
        "duration": 47.067000,
    },
    {
        "dataset": "cnceleb",
        "utt_id": "eval-test-id00852-speech-id00852-speech-01-003",
        "sha256": "0b67be9928c8f3eb9f064e1f2b276bb1bebea709840f6b0b7a01b79cf0661ba9",
        "duration": 1.718312,
    },
    {
        "dataset": "whisper40",
        "utt_id": "whisper40_test_apz01_14_nature",
        "sha256": "d0c4c99d4fb3c7358b920da6d0f3623426c14bb4a0a6c78633854c47f5217677",
        "duration": 21.343750,
    },
    {
        "dataset": "whisper40",
        "utt_id": "whisper40_test_rhr22_14_whsp",
        "sha256": "1bc36eaaeef207d0158bedfd16e7f9332e1f454dffa5ed08b26b98a0274c65d3",
        "duration": 20.693750,
    },
    {
        "dataset": "voxceleb",
        "utt_id": "id10305-nJbBcMdxQU4-00010",
        "sha256": "e31c64200436d49ad25890bb70011c25e99a14495d23ef9a39363ad23ec7b32f",
        "duration": 20.200063,
    },
]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="Prepared benchmark data root")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Optional dataset id filter. Can be passed multiple times.",
    )
    return parser.parse_args()


def read_wav_scp(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.is_file():
        return mapping
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audio_duration(path: Path) -> float | None:
    try:
        import soundfile as sf
    except Exception:
        return None
    try:
        info = sf.info(str(path))
    except Exception:
        return None
    if not info.samplerate:
        return None
    return float(info.frames) / float(info.samplerate)


def main() -> None:
    args = get_args()
    data_root = Path(args.data).expanduser().resolve()
    dataset_filter = set(args.dataset or [])
    rows = [
        row
        for row in REFERENCE_UTTERANCES
        if not dataset_filter or row["dataset"] in dataset_filter
    ]

    print(
        "\t".join(
            [
                "dataset",
                "utt_id",
                "status",
                "hash_ok",
                "duration_ok",
                "sha256",
                "expected_sha256",
                "duration",
                "expected_duration",
                "path",
            ]
        )
    )
    mapping_cache: dict[str, dict[str, str]] = {}
    for row in rows:
        dataset = row["dataset"]
        if dataset not in mapping_cache:
            prep_dir = data_root / dataset / "prepared"
            mapping_cache[dataset] = read_wav_scp(prep_dir / "wav_all.scp") or read_wav_scp(
                prep_dir / "wav.scp"
            )
        mapping = mapping_cache[dataset]
        wav_value = mapping.get(row["utt_id"])
        if not wav_value:
            print(
                "\t".join(
                    [
                        dataset,
                        row["utt_id"],
                        "missing_wav_scp_entry",
                        "NA",
                        "NA",
                        "NA",
                        row["sha256"],
                        "NA",
                        f"{row['duration']:.6f}",
                        "NA",
                    ]
                )
            )
            continue
        if wav_value.endswith("|"):
            status = "pipeline_not_hashable"
            wav_path = wav_value
            sha = "NA"
            dur = None
        else:
            wav_path_obj = Path(wav_value).expanduser()
            if not wav_path_obj.is_absolute():
                wav_path_obj = (data_root / dataset / "prepared" / wav_path_obj).resolve()
            wav_path = str(wav_path_obj)
            if wav_path_obj.is_file():
                status = "ok"
                sha = file_sha256(wav_path_obj)
                dur = audio_duration(wav_path_obj)
            else:
                status = "missing_audio_file"
                sha = "NA"
                dur = None
        hash_ok = str(sha == row["sha256"]) if sha != "NA" else "NA"
        duration_ok = (
            str(abs(dur - row["duration"]) <= 0.002) if dur is not None else "NA"
        )
        print(
            "\t".join(
                [
                    dataset,
                    row["utt_id"],
                    status,
                    hash_ok,
                    duration_ok,
                    sha,
                    row["sha256"],
                    f"{dur:.6f}" if dur is not None else "NA",
                    f"{row['duration']:.6f}",
                    wav_path,
                ]
            )
        )


if __name__ == "__main__":
    main()
