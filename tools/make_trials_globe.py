#!/usr/bin/env python3
"""Generate GLOBE accent-aware speaker verification trials.

Target pairs are enumerated exhaustively (all C(n,2) per speaker) and capped
at --max-target-trials.

Nontarget total = actual_target * --nontarget-ratio, split evenly between:
  intra_accent - different speaker, same accent
  inter_accent - different speaker, different accent
"""

import os
import random
import argparse
from collections import defaultdict
from itertools import combinations


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate GLOBE trials (target / intra-accent / inter-accent)"
    )
    parser.add_argument("--utt2spk", required=True)
    parser.add_argument("--utt2accent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-target-trials", type=int, default=20000,
                        help="Cap on total target trials. All C(n,2) pairs are "
                             "enumerated; subsampled if count exceeds this limit. "
                             "(default: 20000)")
    parser.add_argument("--nontarget-ratio", type=int, default=10,
                        help="Nontarget count = actual target count multiplied by this ratio, "
                             "split evenly between intra- and inter-accent. "
                             "(default: 10)")
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def main():
    args = get_args()
    random.seed(args.seed)

    # Load metadata
    utt_info = {}
    spk2utts = defaultdict(list)
    acc2spks = defaultdict(set)

    with open(args.utt2spk) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, s = parts[0], parts[1]
                utt_info[u] = {"spk": s}

    with open(args.utt2accent) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, a = parts[0], parts[1]
                if u in utt_info:
                    utt_info[u]["accent"] = a

    for u, info in utt_info.items():
        if "spk" in info and "accent" in info:
            s, a = info["spk"], info["accent"]
            spk2utts[s].append(u)
            acc2spks[a].add(s)

    for a in acc2spks:
        acc2spks[a] = list(acc2spks[a])

    print(f"Loaded {len(utt_info)} utts, {len(spk2utts)} speakers, "
          f"{len(acc2spks)} accents.")

    seen = set()

    # ================================================================
    # 1. Target: enumerate all C(n,2) per speaker
    # ================================================================
    print("Enumerating all target pairs ...")
    all_target_pairs = []
    for spk, utts in spk2utts.items():
        for u1, u2 in combinations(sorted(utts), 2):
            p = tuple(sorted((u1, u2)))
            if p not in seen:
                seen.add(p)
                all_target_pairs.append(p)

    total_possible = len(all_target_pairs)
    if args.max_target_trials > 0 and total_possible > args.max_target_trials:
        print(f"  {total_possible} pairs found. Sampling {args.max_target_trials} ...")
        all_target_pairs = random.sample(all_target_pairs, args.max_target_trials)
    else:
        print(f"  {total_possible} pairs found. Keeping all.")

    actual_target = len(all_target_pairs)

    # ================================================================
    # 2. Nontarget: actual target count multiplied by the ratio, 50% intra / 50% inter
    # ================================================================
    n_nontarget = actual_target * args.nontarget_ratio
    n_intra = n_nontarget // 2
    n_inter = n_nontarget - n_intra

    accs = list(acc2spks.keys())
    nontarget_trials = []

    # --- Intra-accent nontarget (diff speaker, same accent) ---
    print(f"Generating {n_intra} intra-accent nontarget pairs ...")
    cnt, attempts = 0, 0
    while cnt < n_intra and attempts < n_intra * 20:
        attempts += 1
        a = random.choice(accs)
        spks_in_acc = acc2spks[a]
        if len(spks_in_acc) < 2:
            continue
        s1, s2 = random.sample(spks_in_acc, 2)
        u1 = random.choice(spk2utts[s1])
        u2 = random.choice(spk2utts[s2])
        p = tuple(sorted((u1, u2)))
        if p not in seen:
            seen.add(p)
            nontarget_trials.append((p[0], p[1], "nontarget", "intra_accent"))
            cnt += 1
    print(f"  Intra-accent: {cnt} pairs generated.")

    # --- Inter-accent nontarget (diff speaker, diff accent) ---
    print(f"Generating {n_inter} inter-accent nontarget pairs ...")
    cnt, attempts = 0, 0
    while cnt < n_inter and attempts < n_inter * 20:
        attempts += 1
        if len(accs) < 2:
            break
        a1, a2 = random.sample(accs, 2)
        s1 = random.choice(acc2spks[a1])
        s2 = random.choice(acc2spks[a2])
        u1 = random.choice(spk2utts[s1])
        u2 = random.choice(spk2utts[s2])
        p = tuple(sorted((u1, u2)))
        if p not in seen:
            seen.add(p)
            nontarget_trials.append((p[0], p[1], "nontarget", "inter_accent"))
            cnt += 1
    print(f"  Inter-accent: {cnt} pairs generated.")

    # ================================================================
    # 3. Write output
    # ================================================================
    all_trials = (
        [(p[0], p[1], "target", "same") for p in all_target_pairs] +
        nontarget_trials
    )
    all_trials.sort()

    stats = defaultdict(int)
    for t in all_trials:
        stats[(t[2], t[3])] += 1
    print("Final stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for t in all_trials:
            f.write(f"{t[0]} {t[1]} {t[2]} {t[3]}\n")
    print(f"Saved {len(all_trials)} trials to {args.output}")


if __name__ == "__main__":
    main()
