#!/usr/bin/env python3
"""Prepare AliMeeting test set for speaker verification.

Pipeline (when --max-utts > 0):
  1. Scan TextGrids and list candidate segments (no audio cut yet).
  2. Filter by --min-dur (default 3.0s).
  3. Balanced random sample: ~half near / half far (total <= max_utts).
  4. Use Sox to extract only the selected segments -> prepared/segments/*.wav
  5. Write wav.scp, utt2spk, utt2channel, spk2utt

This avoids cutting ~18k wavs when only 10k are needed.
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True, help="Root of extracted AliMeeting test data")
    p.add_argument("--prep-dir", required=True, help="Output directory for Kaldi files")
    p.add_argument(
        "--max-utts",
        type=int,
        default=0,
        help="Max utterances to keep (0 = keep all). When >0, sample before cutting audio.",
    )
    p.add_argument(
        "--min-dur",
        type=float,
        default=3.0,
        help="Minimum segment duration in seconds (default 3.0)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=777,
        help="RNG seed for subsampling (default 777)",
    )
    return p.parse_args()


def find_dir(raw_dir: Path, *candidates: str) -> Optional[Path]:
    for c in candidates:
        p = raw_dir / c
        if p.is_dir():
            return p
    return None


def parse_textgrid(tg_path: Path) -> list[tuple[str, float, float, str]]:
    try:
        import textgrid as tg_lib

        tg = tg_lib.TextGrid.fromFile(str(tg_path))
        segments = []
        for tier in tg.tiers:
            for interval in tier.intervals:
                if interval.mark and interval.mark.strip():
                    segments.append(
                        (
                            tier.name,
                            float(interval.minTime),
                            float(interval.maxTime),
                            interval.mark.strip(),
                        )
                    )
        return segments
    except ImportError:
        pass

    segments = []
    with open(tg_path, encoding="utf-8", errors="ignore") as f:
        content = f.read()
    tier_blocks = re.split(r"item\s*\[\d+\]", content)
    for block in tier_blocks[1:]:
        name_m = re.search(r'name\s*=\s*"([^"]*)"', block)
        tier_name = name_m.group(1) if name_m else "unknown"
        intervals = re.findall(
            r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"',
            block,
        )
        for xmin, xmax, text in intervals:
            if text.strip():
                segments.append((tier_name, float(xmin), float(xmax), text.strip()))
    return segments


def get_spk_from_tier_far(tier_name: str) -> str:
    parts = tier_name.split("_")
    for part in reversed(parts):
        if part.upper().startswith("SPK") and len(part) > 3:
            return part[3:]
    m = re.search(r"(\d{4})$", tier_name)
    return m.group(1) if m else tier_name


def sox_extract(src_wav: Path, out_wav: Path, start: float, duration: float, channel: int = 1) -> bool:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sox",
        str(src_wav),
        "-c",
        "1",
        str(out_wav),
        "remix",
        str(channel),
        "trim",
        str(start),
        str(duration),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def wav_duration_sox(src_wav: Path) -> float:
    r = subprocess.run(
        ["sox", "--i", "-D", str(src_wav)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        return 0.0
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def collect_near_candidates(
    near_audio: Path, near_tg: Path, min_dur: float
) -> list[dict[str, Any]]:
    cands: list[dict[str, Any]] = []
    tg_files = sorted(near_tg.rglob("*.TextGrid"))
    print(f"[AliMeeting] Scanning near-field TextGrids: {len(tg_files)} files ...")
    for tg_path in tg_files:
        session_id = tg_path.stem
        parts = session_id.split("_")
        spk_id = None
        for part in reversed(parts):
            if part.upper().startswith("SPK"):
                spk_id = part[3:]
                break
        if spk_id is None:
            m = re.search(r"(\d{4})$", session_id)
            spk_id = m.group(1) if m else session_id

        wavs = list(near_audio.rglob(f"{session_id}*.wav"))
        if not wavs:
            wavs = list(near_audio.rglob(f"*{session_id}*.wav"))
        if not wavs:
            print(f"  [WARN] No wav for {session_id}")
            continue
        src_wav = wavs[0]

        segments = parse_textgrid(tg_path)
        if not segments:
            total_dur = wav_duration_sox(src_wav)
            if total_dur >= min_dur:
                cands.append(
                    {
                        "ch": "near",
                        "spk_id": spk_id,
                        "session_id": session_id,
                        "seg_idx": 0,
                        "src_wav": src_wav,
                        "start": 0.0,
                        "dur": total_dur,
                    }
                )
            continue

        seg_idx = 0
        for _tier_name, start, end, _text in segments:
            dur = end - start
            if dur < min_dur:
                continue
            cands.append(
                {
                    "ch": "near",
                    "spk_id": spk_id,
                    "session_id": session_id,
                    "seg_idx": seg_idx,
                    "src_wav": src_wav,
                    "start": float(start),
                    "dur": float(dur),
                }
            )
            seg_idx += 1
    print(f"  Near-field candidates (>= {min_dur}s): {len(cands)}")
    return cands


def collect_far_candidates(
    far_audio: Path, far_tg: Path, min_dur: float
) -> list[dict[str, Any]]:
    cands: list[dict[str, Any]] = []
    tg_files = sorted(far_tg.rglob("*.TextGrid"))
    print(f"[AliMeeting] Scanning far-field TextGrids: {len(tg_files)} files ...")
    for tg_path in tg_files:
        session_id = tg_path.stem
        wavs = list(far_audio.rglob(f"{session_id}*.wav"))
        if not wavs:
            wavs = list(far_audio.rglob(f"*{session_id}*.wav"))
        if not wavs:
            print(f"  [WARN] No wav for {session_id}")
            continue
        src_wav = wavs[0]

        segments = parse_textgrid(tg_path)
        tier_segs: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for tier_name, start, end, text in segments:
            tier_segs[tier_name].append((start, end))

        for tier_name, tier_intervals in tier_segs.items():
            spk_id = get_spk_from_tier_far(tier_name)
            seg_idx = 0
            for start, end in tier_intervals:
                dur = end - start
                if dur < min_dur:
                    continue
                cands.append(
                    {
                        "ch": "far",
                        "spk_id": spk_id,
                        "session_id": session_id,
                        "seg_idx": seg_idx,
                        "src_wav": src_wav,
                        "start": float(start),
                        "dur": float(dur),
                    }
                )
                seg_idx += 1
    print(f"  Far-field candidates (>= {min_dur}s): {len(cands)}")
    return cands


def balanced_sample(
    near_cands: list[dict[str, Any]],
    far_cands: list[dict[str, Any]],
    max_utts: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Take up to max_utts total, prioritizing ~50%% near + 50%% far."""
    if max_utts <= 0:
        return near_cands + far_cands
    rng.shuffle(near_cands)
    rng.shuffle(far_cands)
    half = max_utts // 2
    n_near = min(half, len(near_cands))
    n_far = min(max_utts - n_near, len(far_cands))
    # Fill from the other side when one condition has too few candidates.
    if n_near + n_far < max_utts:
        need = max_utts - n_near - n_far
        extra_near = min(need, len(near_cands) - n_near)
        n_near += extra_near
        need -= extra_near
        n_far += min(need, len(far_cands) - n_far)

    selected = near_cands[:n_near] + far_cands[:n_far]
    print(
        f"[AliMeeting] Sampled {len(selected)} / requested max {max_utts} "
        f"(near={n_near}, far={n_far})"
    )
    return selected


def clean_segment_dir(seg_dir: Path) -> None:
    if seg_dir.is_dir():
        shutil.rmtree(seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = get_args()
    rng = random.Random(args.seed)
    raw_dir = Path(args.raw_dir)
    prep_dir = Path(args.prep_dir)
    prep_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = prep_dir / "segments"

    test_ali = find_dir(raw_dir, "Test_Ali", ".")
    if test_ali is None:
        test_ali = raw_dir

    near_audio = find_dir(
        test_ali,
        "Test_Ali_near/audio_dir",
        "Test_Ali_near",
        "near/audio_dir",
        "near",
    )
    far_audio = find_dir(
        test_ali,
        "Test_Ali_far/audio_dir",
        "Test_Ali_far",
        "far/audio_dir",
        "far",
    )
    near_tg = find_dir(
        test_ali,
        "Test_Ali_near/textgrid_dir",
        "Test_Ali_near/TextGrid",
    )
    far_tg = find_dir(
        test_ali,
        "Test_Ali_far/textgrid_dir",
        "Test_Ali_far/TextGrid",
    )

    if near_audio is None and far_audio is None:
        print(
            f"[AliMeeting] ERROR: Cannot find near/far audio dirs under {test_ali}",
            file=sys.stderr,
        )
        sys.exit(1)

    near_cands: list[dict[str, Any]] = []
    far_cands: list[dict[str, Any]] = []
    if near_audio and near_tg:
        near_cands = collect_near_candidates(near_audio, near_tg, args.min_dur)
    else:
        print("[AliMeeting] WARNING: near-field audio or TextGrid not found.")
    if far_audio and far_tg:
        far_cands = collect_far_candidates(far_audio, far_tg, args.min_dur)
    else:
        print("[AliMeeting] WARNING: far-field audio or TextGrid not found.")

    if not near_cands and not far_cands:
        print("[AliMeeting] ERROR: No candidate segments.", file=sys.stderr)
        sys.exit(1)

    selected = balanced_sample(near_cands, far_cands, args.max_utts, rng)

    print(f"[AliMeeting] Cutting {len(selected)} wav segments with sox ...")
    clean_segment_dir(seg_dir)

    records: list[tuple[str, str, str, Path]] = []
    errors = 0
    for c in selected:
        ch = c["ch"]
        spk_id = c["spk_id"]
        session_id = c["session_id"]
        seg_idx = c["seg_idx"]
        if ch == "near":
            utt_id = f"alimeeting_near_{spk_id}_{session_id}_{seg_idx:04d}"
        else:
            utt_id = f"alimeeting_far_{spk_id}_{session_id}_{seg_idx:04d}"
        out_wav = seg_dir / f"{utt_id}.wav"
        ok = sox_extract(
            c["src_wav"], out_wav, c["start"], c["dur"], channel=1
        )
        if not ok:
            errors += 1
            continue
        records.append((utt_id, spk_id, ch, out_wav))

    if not records:
        print("[AliMeeting] ERROR: No segments extracted after sox.", file=sys.stderr)
        sys.exit(1)

    print(f"[AliMeeting] Sox done: {len(records)} ok, {errors} errors")

    with open(prep_dir / "wav.scp", "w") as wav_scp, open(
        prep_dir / "utt2spk", "w"
    ) as utt2spk, open(prep_dir / "utt2channel", "w") as utt2ch:
        for utt_id, spk_id, ch, wav_path in sorted(records):
            wav_scp.write(f"{utt_id} {wav_path}\n")
            utt2spk.write(f"{utt_id} {spk_id}\n")
            utt2ch.write(f"{utt_id} {ch}\n")

    spk2utts: dict[str, list[str]] = defaultdict(list)
    for utt_id, spk_id, _ch, _p in records:
        spk2utts[spk_id].append(utt_id)
    with open(prep_dir / "spk2utt", "w") as f:
        for spk_id in sorted(spk2utts):
            f.write(f"{spk_id} {' '.join(sorted(spk2utts[spk_id]))}\n")

    near_cnt = sum(1 for r in records if r[2] == "near")
    far_cnt = sum(1 for r in records if r[2] == "far")
    print(
        f"[AliMeeting] Written: {len(records)} utts (near={near_cnt}, far={far_cnt}), "
        f"{len(spk2utts)} speakers -> {prep_dir}"
    )


if __name__ == "__main__":
    main()
