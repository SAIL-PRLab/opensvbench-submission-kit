#!/usr/bin/env python3
"""Speaker-balanced subsampling of a wav.scp to at most N utterances.

Sampling strategy:
  1. Group utterances by speaker (via --utt2spk).
  2. Shuffle speakers randomly.
  3. Assign each speaker a quota = max(min_utts_per_spk, max_utts // n_spks).
  4. Take min(quota, available) utterances per speaker until budget is filled.
  5. Overwrite wav.scp in-place (or write to --out-wav-scp).
  6. Optionally filter any number of additional metadata files in-place.
  7. Optionally regenerate spk2utt from the filtered utt2spk.

Usage example:
  python3 tools/subsample_wavscp.py \\
      --wav-scp  data/hi-mia/prepared/wav.scp \\
      --utt2spk  data/hi-mia/prepared/utt2spk \\
      --max-utts 10000 \\
      --filter-files data/hi-mia/prepared/utt2spk \\
      --regen-spk2utt data/hi-mia/prepared/spk2utt \\
      --seed 777
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser(description="Speaker-balanced wav.scp subsampling")
    p.add_argument("--wav-scp", required=True, help="Input (and default output) wav.scp")
    p.add_argument("--max-utts", type=int, required=True,
                   help="Maximum number of utterances to keep")
    p.add_argument("--utt2spk", default=None,
                   help="utt2spk file; enables speaker-balanced sampling")
    p.add_argument("--filter-files", nargs="*", default=[],
                   help="Additional metadata files to filter in-place "
                        "(e.g. utt2spk utt2lang utt2text utt2genre utt2session utt2accent)")
    p.add_argument("--regen-spk2utt", default=None,
                   help="If given, regenerate spk2utt at this path from filtered utt2spk")
    p.add_argument("--out-wav-scp", default=None,
                   help="Output wav.scp path (default: overwrite input)")
    p.add_argument("--min-utts-per-spk", type=int, default=2,
                   help="Minimum utterances to keep per speaker (default: 2)")
    p.add_argument("--seed", type=int, default=777)
    return p.parse_args()


def _regen_spk2utt(utt2spk_path: str, spk2utt_path: str):
    spk2utts = defaultdict(list)
    with open(utt2spk_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                spk2utts[parts[1]].append(parts[0])
    with open(spk2utt_path, "w") as f:
        for spk in sorted(spk2utts):
            f.write(f"{spk} {' '.join(sorted(spk2utts[spk]))}\n")
    print(f"[Subsample] Regenerated spk2utt: {len(spk2utts)} speakers -> {spk2utt_path}")


def main():
    args = get_args()
    random.seed(args.seed)

    # --- Read wav.scp ---
    utt2path = {}
    with open(args.wav_scp) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt2path[parts[0]] = parts[1]
    n_orig = len(utt2path)

    if n_orig <= args.max_utts:
        print(f"[Subsample] {n_orig} utterances <= max_utts={args.max_utts}; skipping.")
        return

    # --- Read utt2spk (optional) ---
    utt2spk = {}
    if args.utt2spk and Path(args.utt2spk).exists():
        with open(args.utt2spk) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utt2spk[parts[0]] = parts[1]

    # --- Speaker-balanced sampling ---
    if utt2spk:
        spk2utts = defaultdict(list)
        for utt in utt2path:
            spk2utts[utt2spk.get(utt, utt)].append(utt)

        # Shuffle within each speaker first
        for utts in spk2utts.values():
            random.shuffle(utts)

        # Shuffle speaker order for random speaker selection
        speakers = list(spk2utts.keys())
        random.shuffle(speakers)

        n_spks = len(speakers)
        min_u = args.min_utts_per_spk
        quota = max(min_u, args.max_utts // n_spks)

        selected = []
        remainder = []
        for spk in speakers:
            utts = spk2utts[spk]
            take = min(quota, len(utts))
            selected.extend(utts[:take])
            remainder.extend(utts[take:])
            if len(selected) >= args.max_utts:
                break

        # Fill remaining budget from leftovers if needed
        need = args.max_utts - len(selected)
        if need > 0 and remainder:
            random.shuffle(remainder)
            selected.extend(remainder[:need])

        selected = selected[:args.max_utts]
    else:
        all_utts = list(utt2path.keys())
        random.shuffle(all_utts)
        selected = all_utts[:args.max_utts]

    selected_set = set(selected)
    n_spks_after = len({utt2spk.get(u, u) for u in selected_set}) if utt2spk else "N/A"
    print(f"[Subsample] {n_orig} -> {len(selected_set)} utterances "
          f"({n_spks_after} speakers)")

    # --- Write filtered wav.scp ---
    out_wav_scp = args.out_wav_scp or args.wav_scp
    with open(out_wav_scp, "w") as f:
        for utt in sorted(selected_set):
            f.write(f"{utt} {utt2path[utt]}\n")

    # --- Filter additional metadata files in-place ---
    for fpath in (args.filter_files or []):
        p = Path(fpath)
        if not p.exists():
            continue
        lines_out = []
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split()
                if parts and parts[0] in selected_set:
                    lines_out.append(line)
        with open(fpath, "w") as f:
            f.writelines(lines_out)
        print(f"[Subsample] Filtered {fpath}: {len(lines_out)} entries")

    # --- Regenerate spk2utt ---
    if args.regen_spk2utt and args.utt2spk:
        _regen_spk2utt(args.utt2spk, args.regen_spk2utt)


if __name__ == "__main__":
    main()
