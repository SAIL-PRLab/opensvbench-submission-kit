#!/usr/bin/env python3
"""Generate CN-Celeb trial list with genre-only conditions: same_genre / diff_genre.

Unified rule (per condition):
  - target trials are enumerated exhaustively then capped at
    --max-target-trials-per-condition.
  - nontarget trials = target_count * --nontarget-ratio for the SAME condition.

Output format (one line per trial):
    <utt1> <utt2> <target|nontarget> <condition>

The scorer computes Overall first and then reports same_genre and diff_genre
by condition, so this script does not write a separate overall condition.
"""

import os
import random
import argparse
from collections import defaultdict
from itertools import combinations


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate CN-Celeb trials (same_genre / diff_genre only)"
    )
    parser.add_argument("--utt2spk",   required=True, help="Path to utt2spk file")
    parser.add_argument("--utt2genre", required=True, help="Path to utt2genre file")
    parser.add_argument("--output",    required=True, help="Path to output trials file")
    parser.add_argument("--max-target-trials-per-condition", type=int, default=20000,
                        help="Cap on target trials PER condition (same_genre / diff_genre). "
                             "All valid pairs are enumerated first; if the count "
                             "exceeds this limit a random subset is taken. "
                             "(default: 20000)")
    parser.add_argument("--nontarget-ratio", type=int, default=10,
                        help="For each condition, multiply the target count by this ratio (default: 10)")
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def _load_mapping(path):
    mapping = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    return mapping


def _sorted_pair(a, b):
    return (a, b) if a <= b else (b, a)


def _sample_pairs(pool_a, pool_b, n, cross=False, seen=None):
    if seen is None:
        seen = set()
    pairs = []
    max_attempts = n * 30
    attempts = 0
    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        if cross:
            u1 = random.choice(pool_a)
            u2 = random.choice(pool_b)
        else:
            if len(pool_a) < 2:
                break
            u1, u2 = random.sample(pool_a, 2)
        key = _sorted_pair(u1, u2)
        if u1 == u2 or key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def main():
    args = get_args()
    random.seed(args.seed)
    utt2spk   = _load_mapping(args.utt2spk)
    utt2genre = _load_mapping(args.utt2genre)
    all_utts = list(utt2spk.keys())
    print(f"Loaded {len(all_utts)} utterances.")

    spk2genre_utts = defaultdict(lambda: defaultdict(list))
    genre2utts = defaultdict(list)
    for utt in all_utts:
        spk, gen = utt2spk.get(utt), utt2genre.get(utt, "unknown")
        if spk:
            spk2genre_utts[spk][gen].append(utt)
        genre2utts[gen].append(utt)
    n_spks = len(spk2genre_utts)
    genres = list(genre2utts.keys())
    print(f"  Speakers : {n_spks}, Genres : {sorted(genres)}")

    seen_pairs = set()
    trials = []

    def emit(u1, u2, label, cond):
        key = _sorted_pair(u1, u2)
        if key not in seen_pairs:
            seen_pairs.add(key)
            trials.append((key[0], key[1], label, cond))
            return True
        return False

    cap = args.max_target_trials_per_condition

    # ================================================================
    # same_genre: target (same spk, same genre) capped per condition
    # ================================================================
    print("Enumerating same_genre target pairs ...")
    same_genre_targets = []
    for spk, gen_dict in spk2genre_utts.items():
        for gen, utts in gen_dict.items():
            if len(utts) < 2:
                continue
            for u1, u2 in combinations(sorted(utts), 2):
                same_genre_targets.append(_sorted_pair(u1, u2))
    # Deduplicate
    same_genre_targets = list(dict.fromkeys(same_genre_targets))
    if cap > 0 and len(same_genre_targets) > cap:
        same_genre_targets = random.sample(same_genre_targets, cap)
    print(f"  same_genre targets: {len(same_genre_targets)} (cap={cap})")
    for u1, u2 in same_genre_targets:
        emit(u1, u2, "target", "same_genre")

    # same_genre non-target count = target count multiplied by the ratio
    n_same_nt = len(same_genre_targets) * args.nontarget_ratio
    print(f"Generating same_genre nontarget pairs: {n_same_nt} ...")
    cnt, attempts = 0, 0
    while cnt < n_same_nt and attempts < n_same_nt * 20:
        attempts += 1
        gen = random.choice(genres)
        utts = genre2utts[gen]
        if len(utts) < 2:
            continue
        u1, u2 = random.sample(utts, 2)
        if utt2spk.get(u1) == utt2spk.get(u2):
            continue
        if emit(u1, u2, "nontarget", "same_genre"):
            cnt += 1
    print(f"  same_genre nontargets: {cnt}")

    # ================================================================
    # diff_genre: target (same spk, different genre) capped per condition
    # ================================================================
    print("Enumerating diff_genre target pairs ...")
    diff_genre_targets = []
    for spk, gen_dict in spk2genre_utts.items():
        gs = [g for g in gen_dict.keys() if gen_dict[g]]
        if len(gs) < 2:
            continue
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                g1, g2 = gs[i], gs[j]
                for u1 in gen_dict[g1]:
                    for u2 in gen_dict[g2]:
                        diff_genre_targets.append(_sorted_pair(u1, u2))
    diff_genre_targets = list(dict.fromkeys(diff_genre_targets))
    if cap > 0 and len(diff_genre_targets) > cap:
        diff_genre_targets = random.sample(diff_genre_targets, cap)
    print(f"  diff_genre targets: {len(diff_genre_targets)} (cap={cap})")
    for u1, u2 in diff_genre_targets:
        emit(u1, u2, "target", "diff_genre")

    # diff_genre non-target count = target count multiplied by the ratio
    n_diff_nt = len(diff_genre_targets) * args.nontarget_ratio
    print(f"Generating diff_genre nontarget pairs: {n_diff_nt} ...")
    cnt, attempts = 0, 0
    while cnt < n_diff_nt and attempts < n_diff_nt * 20:
        attempts += 1
        if len(genres) < 2:
            break
        g1, g2 = random.sample(genres, 2)
        u1 = random.choice(genre2utts[g1])
        u2 = random.choice(genre2utts[g2])
        if utt2spk.get(u1) == utt2spk.get(u2):
            continue
        if emit(u1, u2, "nontarget", "diff_genre"):
            cnt += 1
    print(f"  diff_genre nontargets: {cnt}")

    trials.sort(key=lambda x: x[0])
    from collections import Counter
    stats = Counter(t[3] for t in trials)
    print("\nFinal condition stats:", dict(stats))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for u1, u2, label, cond in trials:
            f.write(f"{u1} {u2} {label} {cond}\n")
    print(f"Saved {len(trials)} trials to {args.output}")


if __name__ == "__main__":
    main()
