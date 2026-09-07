#!/usr/bin/env python3
"""
Generate speaker verification trials for short-utterance duration evaluation.

Input:
  --wav-scp          : wav.scp whose utt_ids follow the pattern {orig_utt_id}_dur_{Xs}
                       produced by cut_short_utterances.py
  --utt2spk          : utt2spk file for the ORIGINAL (full-length) utterances

Output trial format (4 columns, space-separated):
  utt1  utt2  target|nontarget  condition

where condition is one of: dur_0.5s, dur_1s, dur_2s, dur_3s

Trial generation strategy:
  - Target pairs  : enumerate ALL possible same-speaker pairs C(n, 2) per speaker.
                    If total > --max-target-trials, randomly subsample.
  - Nontarget pairs: random cross-speaker pairs, count = n_target * --nontarget-ratio.

This ensures:
  - No artificial duplication of positive samples.
  - When utterances per speaker are few (e.g. 5), all C(5,2)=10 pairs are kept.
  - Nontarget count scales with the actual positive count, not a fixed number.
"""

import argparse
import os
import random
from collections import defaultdict
from itertools import combinations


def dur_str_sort_key(dur_str: str) -> float:
    """Parse '0.5s', '1s', '2s', '3s' -> 0.5, 1.0, 2.0, 3.0 for sorting."""
    return float(dur_str.rstrip("s"))


def generate_target_pairs(spk2utts: dict, max_target: int,
                          rng: random.Random) -> list:
    """Enumerate all same-speaker pairs; subsample if exceeding max_target.

    Args:
        spk2utts:   dict mapping speaker_id -> list of utt_ids.
        max_target: maximum number of target pairs to keep (0 = keep all).
        rng:        random.Random instance.

    Returns:
        List of (utt1, utt2) target pairs.
    """
    all_pairs = []
    for utts in spk2utts.values():
        utts_sorted = sorted(utts)  # deterministic enumeration order
        all_pairs.extend(combinations(utts_sorted, 2))

    if max_target > 0 and len(all_pairs) > max_target:
        all_pairs = rng.sample(all_pairs, max_target)

    return list(all_pairs)


def generate_nontarget_pairs(spk2utts: dict, n_nontarget: int,
                             rng: random.Random) -> list:
    """Randomly sample cross-speaker pairs.

    Args:
        spk2utts:     dict mapping speaker_id -> list of utt_ids.
        n_nontarget:  desired number of nontarget pairs.
        rng:          random.Random instance.

    Returns:
        List of (utt1, utt2) nontarget pairs (may contain duplicates when
        the speaker pool is small relative to n_nontarget).
    """
    speakers = sorted(spk2utts.keys())
    if len(speakers) < 2:
        return []

    pairs = []
    max_attempts = n_nontarget * 10
    attempts = 0
    while len(pairs) < n_nontarget and attempts < max_attempts:
        spk1, spk2 = rng.sample(speakers, 2)
        u1 = rng.choice(spk2utts[spk1])
        u2 = rng.choice(spk2utts[spk2])
        pairs.append((u1, u2))
        attempts += 1

    return pairs


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate short-utterance duration-bucketed SV trials"
    )
    parser.add_argument("--wav-scp", required=True,
                        help="wav.scp with duration-cut utterances "
                             "(utt_ids end with _dur_{Xs})")
    parser.add_argument("--utt2spk", required=True,
                        help="utt2spk for the original (full-length) utterances")
    parser.add_argument("--output", required=True,
                        help="Output trial file path")
    parser.add_argument("--max-target-trials", type=int, default=20000,
                        help="Maximum number of target (positive) trials per "
                             "duration bucket. All C(n,2) pairs are enumerated "
                             "first; if the count exceeds this limit a random "
                             "subset is taken. 0 = keep all pairs. (default: 20000)")
    parser.add_argument("--nontarget-ratio", type=int, default=10,
                        help="Number of nontarget trials = "
                             "actual_target_count * nontarget_ratio (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


def main():
    args = get_args()
    rng = random.Random(args.seed)

    # Load original utt2spk
    utt2spk: dict[str, str] = {}
    with open(args.utt2spk) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt2spk[parts[0]] = parts[1]

    # Group duration-cut utterances by (dur_bucket, speaker)
    # utt_id format: {orig_utt_id}_dur_{Xs}
    sep = "_dur_"
    dur_spk2utts: dict[str, dict[str, list]] = defaultdict(
        lambda: defaultdict(list)
    )

    with open(args.wav_scp) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if not parts:
                continue
            utt_id = parts[0]

            idx = utt_id.rfind(sep)
            if idx < 0:
                continue

            orig_utt = utt_id[:idx]
            dur_str = utt_id[idx + len(sep):]   # e.g. "0.5s", "1s"

            spk = utt2spk.get(orig_utt)
            if spk is None:
                continue

            dur_spk2utts[dur_str][spk].append(utt_id)

    if not dur_spk2utts:
        print(
            "Warning: no duration-bucketed utterances found. "
            "Check that utt_ids in wav_scp end with _dur_{Xs}."
        )
        return

    all_trials: list[str] = []

    for dur_str in sorted(dur_spk2utts.keys(), key=dur_str_sort_key):
        condition = f"dur_{dur_str}"
        spk2utts = {s: list(u) for s, u in dur_spk2utts[dur_str].items()}

        n_spks = len(spk2utts)
        n_utts = sum(len(v) for v in spk2utts.values())
        spks_with_pairs = sum(1 for v in spk2utts.values() if len(v) >= 2)

        # --- Target pairs: enumerate all C(n,2), subsample if needed ---
        target_pairs = generate_target_pairs(spk2utts, args.max_target_trials, rng)
        n_target = len(target_pairs)

        # --- Nontarget pairs: n_target * ratio ---
        n_nontarget = n_target * args.nontarget_ratio
        nontarget_pairs = generate_nontarget_pairs(spk2utts, n_nontarget, rng)

        print(
            f"  {condition}: {n_spks} speakers ({spks_with_pairs} with >=2 utts), "
            f"{n_utts} utts -> "
            f"{n_target} target + {len(nontarget_pairs)} nontarget trials"
        )

        for u1, u2 in target_pairs:
            all_trials.append(f"{u1} {u2} target {condition}")
        for u1, u2 in nontarget_pairs:
            all_trials.append(f"{u1} {u2} nontarget {condition}")

    rng.shuffle(all_trials)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w") as f:
        for line in all_trials:
            f.write(line + "\n")

    print(f"Total {len(all_trials)} trials written to {args.output}")


if __name__ == "__main__":
    main()
