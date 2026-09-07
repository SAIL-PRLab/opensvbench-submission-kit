#!/usr/bin/env python3
"""Install the frozen GLOBE core manifest and trial file.

GLOBE is hosted as converted Parquet shards on Hugging Face. We found that
regenerating the speaker-balanced subset from the live repository can drift
from the leaderboard's frozen core trial set when upstream files or row order
change. This helper keeps the public submission kit aligned with the canonical
OpenSVBench GLOBE subset while still requiring users to fetch audio upstream.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply frozen GLOBE manifests")
    parser.add_argument("--manifest", required=True, help="TSV with utt_id, rel_path, speaker_id, accent")
    parser.add_argument("--trials", required=True, help="Frozen trials_accent.txt")
    parser.add_argument("--raw-dir", required=True, help="Local GLOBE raw_data directory")
    parser.add_argument("--prep-dir", required=True, help="Local GLOBE prepared directory")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    manifest_path = Path(args.manifest)
    trials_path = Path(args.trials)
    raw_dir = Path(args.raw_dir).resolve()
    prep_dir = Path(args.prep_dir).resolve()
    prep_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with manifest_path.open("r") as f:
        header = f.readline().strip().split("\t")
        expected = ["utt_id", "rel_path", "speaker_id", "accent"]
        if header != expected:
            raise ValueError(f"Unexpected manifest header in {manifest_path}: {header}")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                raise ValueError(f"Malformed manifest row: {line!r}")
            rows.append(
                {
                    "utt_id": parts[0],
                    "rel_path": parts[1],
                    "speaker_id": parts[2],
                    "accent": parts[3],
                    "wav_path": raw_dir / parts[1],
                }
            )

    missing = [row for row in rows if not row["wav_path"].is_file()]
    if missing:
        print("[GLOBE] ERROR: canonical manifest wavs are missing from raw_data.")
        print(f"  missing_count: {len(missing)} / {len(rows)}")
        print("  sample missing paths:")
        for row in missing[:10]:
            print(f"    {row['wav_path']}")
        raise SystemExit(2)

    with (prep_dir / "wav.scp").open("w") as f_wav, \
        (prep_dir / "wav_all.scp").open("w") as f_wav_all, \
        (prep_dir / "utt2spk").open("w") as f_u2s, \
        (prep_dir / "utt2accent").open("w") as f_u2a:
        for row in sorted(rows, key=lambda item: item["utt_id"]):
            line = f"{row['utt_id']} {row['wav_path']}\n"
            f_wav.write(line)
            f_wav_all.write(line)
            f_u2s.write(f"{row['utt_id']} {row['speaker_id']}\n")
            f_u2a.write(f"{row['utt_id']} {row['accent']}\n")

    spk2utts: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        spk2utts[row["speaker_id"]].append(row["utt_id"])
    with (prep_dir / "spk2utt").open("w") as f:
        for spk in sorted(spk2utts):
            f.write(f"{spk} {' '.join(sorted(spk2utts[spk]))}\n")

    shutil.copyfile(trials_path, prep_dir / "trials_accent.txt")
    print(
        "[GLOBE] Applied frozen OpenSVBench GLOBE manifest: "
        f"{len(rows)} utterances, {sum(1 for _ in trials_path.open())} trials"
    )


if __name__ == "__main__":
    main()
