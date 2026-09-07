#!/usr/bin/env python3
"""Generate speaker verification trial lists for the HI-MIA dataset.

When --wav-scp is provided, utterances are classified as near-field or far-field
according to HI-MIA filename convention (readme): the second field (pointID)
  - 7: HI-FI Mic (near-field)
  - 2, 5, 6: circular microphone arrays (far-field)
Three conditions are then generated:

  near_near  - both utterances are near-field
  near_far   - one near-field, one far-field (same speaker for targets)
  far_far    - both utterances are far-field

When --wav-scp is not provided, or when no pointID can be parsed from utt_id,
the script falls back to the original behaviour (single "default" condition).

Output format (one trial per line):
  <utt1> <utt2> <target|nontarget> <condition>
"""

import os
import random
import argparse
from collections import defaultdict


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate HI-MIA trials, optionally with near/far-field conditions"
    )
    parser.add_argument("--utt2spk", required=True, help="Path to utt2spk file")
    parser.add_argument(
        "--wav-scp", default=None,
        help="Path to wav.scp; enables near/far-field condition generation"
    )
    parser.add_argument("--output", required=True, help="Path to output trials file")
    parser.add_argument(
        "--num-neg-target-ratio", type=int, default=10,
        help="Ratio of non-target to target pairs (default: 10)"
    )
    parser.add_argument(
        "--max-target-trials-per-condition", type=int, default=20000,
        help="Max target pairs per condition (global cap across speakers) "
             "(default: 20000)"
    )
    parser.add_argument("--seed", type=int, default=777, help="Random seed")
    return parser.parse_args()


# HI-MIA filename: <speakerID>_<pointID>_<micID>_<speed><uttID>.wav
# pointID 7 = HI-FI Mic (near-field); 2, 5, 6 = circular arrays (far-field)
NEAR_POINT_IDS = {"7"}
FAR_POINT_IDS = {"2", "5", "6"}


def infer_field_from_utt_id(utt_id: str):
    """Return 'near', 'far', or None from HI-MIA utt_id (pointID is 2nd field)."""
    parts = utt_id.strip().split("_")
    if len(parts) < 2:
        return None
    point_id = parts[1]
    if point_id in NEAR_POINT_IDS:
        return "near"
    if point_id in FAR_POINT_IDS:
        return "far"
    return None


def _sorted_pair(a, b):
    return (a, b) if a <= b else (b, a)


def generate_target_pairs(utts_a, utts_b, max_pairs, cross_field=False):
    """Return a list of (u1, u2) target pairs.

    For same-field (cross_field=False): utts_a and utts_b are the same list;
    we enumerate all unordered pairs without repetition.
    For cross-field (cross_field=True), use all combinations of utts_a and utts_b.
    """
    if cross_field:
        possible = [(a, b) for a in utts_a for b in utts_b]
    else:
        possible = [
            _sorted_pair(utts_a[i], utts_a[j])
            for i in range(len(utts_a))
            for j in range(i + 1, len(utts_a))
        ]

    if len(possible) > max_pairs:
        possible = random.sample(possible, max_pairs)
    return possible


def generate_nontarget_pairs(spk_list_a, spk2utt_a, spk_list_b, spk2utt_b, n):
    """Generate n non-target (different-speaker) pairs.

    spk_list_a / spk2utt_a: source pool for utterance 1
    spk_list_b / spk2utt_b: source pool for utterance 2
    When both pools are the same object, we enforce s1 != s2.
    """
    same_pool = spk_list_a is spk_list_b
    pairs = []
    max_attempts = n * 20
    attempts = 0
    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        if same_pool:
            if len(spk_list_a) < 2:
                break
            s1, s2 = random.sample(spk_list_a, 2)
        else:
            s1 = random.choice(spk_list_a)
            s2 = random.choice(spk_list_b)
            if s1 == s2:
                continue
        u1_list = spk2utt_a.get(s1, [])
        u2_list = spk2utt_b.get(s2, [])
        if not u1_list or not u2_list:
            continue
        pairs.append((random.choice(u1_list), random.choice(u2_list)))
    return pairs


def generate_field_trials(spk2utt, utt2field, max_target_per_cond, neg_ratio):
    """Generate near_near / near_far / far_far trials.

    Target trials are enumerated across ALL speakers per condition, then capped
    by max_target_per_cond for each condition independently.
    """
    # Split each speaker's utterances by field
    spk2near = {spk: [u for u in utts if utt2field.get(u) == "near"]
                for spk, utts in spk2utt.items()}
    spk2far  = {spk: [u for u in utts if utt2field.get(u) == "far"]
                for spk, utts in spk2utt.items()}

    # Collect target pairs per condition
    cond2pairs = defaultdict(list)  # cond -> list[(u1,u2)]

    for spk in spk2utt:
        near = spk2near[spk]
        far  = spk2far[spk]

        # near_near
        if len(near) >= 2:
            for u1, u2 in generate_target_pairs(near, near, max_pairs=len(near) * (len(near) - 1) // 2, cross_field=False):
                cond2pairs["near_near"].append((u1, u2))

        # far_far
        if len(far) >= 2:
            for u1, u2 in generate_target_pairs(far, far, max_pairs=len(far) * (len(far) - 1) // 2, cross_field=False):
                cond2pairs["far_far"].append((u1, u2))

        # near_far (cross-field, same speaker)
        if near and far:
            for u1, u2 in generate_target_pairs(near, far, max_pairs=len(near) * len(far), cross_field=True):
                cond2pairs["near_far"].append((u1, u2))

    # Cap targets per condition
    target_trials = []
    for cond in ["near_near", "near_far", "far_far"]:
        pairs = cond2pairs.get(cond, [])
        if max_target_per_cond > 0 and len(pairs) > max_target_per_cond:
            pairs = random.sample(pairs, max_target_per_cond)
        for u1, u2 in pairs:
            target_trials.append((u1, u2, "target", cond))

    print(f"  Target trials: {len(target_trials)} "
          f"(near_near + far_far + near_far; cap_per_cond={max_target_per_cond})")

    # Speaker lists with at least one utterance per field
    spk_near = [s for s, u in spk2near.items() if u]
    spk_far  = [s for s, u in spk2far.items()  if u]

    nontarget_trials = []
    # Generate nontarget per condition: n_neg = n_target * neg_ratio
    cond_counts = defaultdict(int)
    for _, _, _, cond in target_trials:
        cond_counts[cond] += 1

    for cond, n_tar in cond_counts.items():
        n_neg = n_tar * neg_ratio
        if cond == "near_near":
            pairs = generate_nontarget_pairs(spk_near, spk2near, spk_near, spk2near, n_neg)
        elif cond == "far_far":
            pairs = generate_nontarget_pairs(spk_far, spk2far, spk_far, spk2far, n_neg)
        else:  # near_far
            pairs = generate_nontarget_pairs(spk_near, spk2near, spk_far, spk2far, n_neg)
        for u1, u2 in pairs:
            nontarget_trials.append((u1, u2, "nontarget", cond))

    print(f"  Non-target trials: {len(nontarget_trials)}")
    return target_trials + nontarget_trials


def generate_default_trials(spk2utt, max_target_per_cond, neg_ratio):
    """Generate trials without field distinction (single 'default' condition).

    Target pairs are enumerated across all speakers, capped globally to
    max_target_per_cond. Non-target count is target count multiplied by ratio.
    """
    speakers = list(spk2utt.keys())

    print(f"Generating target pairs (cap per condition: {max_target_per_cond})...")

    seen = set()
    all_pairs = []
    for spk, utts in spk2utt.items():
        if len(utts) < 2:
            continue
        utts_sorted = sorted(utts)
        for i in range(len(utts_sorted)):
            for j in range(i + 1, len(utts_sorted)):
                p = _sorted_pair(utts_sorted[i], utts_sorted[j])
                if p not in seen:
                    seen.add(p)
                    all_pairs.append(p)

    if max_target_per_cond > 0 and len(all_pairs) > max_target_per_cond:
        all_pairs = random.sample(all_pairs, max_target_per_cond)

    target_trials = [(p[0], p[1], "target", "default") for p in all_pairs]

    num_neg = len(target_trials) * neg_ratio
    print(f"Generated {len(target_trials)} target trials. Generating {num_neg} non-target pairs...")

    nontarget_trials = []
    attempts = 0
    while len(nontarget_trials) < num_neg and attempts < num_neg * 20:
        attempts += 1
        if len(speakers) < 2:
            break
        s1, s2 = random.sample(speakers, 2)
        u1 = random.choice(spk2utt[s1])
        u2 = random.choice(spk2utt[s2])
        nontarget_trials.append((u1, u2, "nontarget", "default"))

    return target_trials + nontarget_trials


def main():
    args = get_args()
    random.seed(args.seed)

    # Load utt2spk
    spk2utt = defaultdict(list)
    with open(args.utt2spk, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                spk2utt[parts[1]].append(parts[0])

    # Optionally load wav.scp to get utt list and infer near/far from HI-MIA pointID (2nd field of utt_id)
    utt2field = {}
    if args.wav_scp:
        with open(args.wav_scp, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utt_id, _ = parts[0], parts[1]
                    field = infer_field_from_utt_id(utt_id)
                    if field:
                        utt2field[utt_id] = field

    has_field_info = len(utt2field) > 0
    if has_field_info:
        near_count = sum(1 for v in utt2field.values() if v == "near")
        far_count  = sum(1 for v in utt2field.values() if v == "far")
        print(f"Field info loaded from wav.scp: near={near_count}, far={far_count}")
        all_trials = generate_field_trials(
            spk2utt, utt2field,
            max_target_per_cond=args.max_target_trials_per_condition,
            neg_ratio=args.num_neg_target_ratio,
        )
    else:
        if args.wav_scp:
            print("Warning: wav.scp provided but no near/far pointID in utt_ids (7=near, 2/5/6=far). "
                  "Falling back to default (no-condition) mode.")
        else:
            print("No wav.scp provided. Generating standard trials (no field conditions).")
        all_trials = generate_default_trials(
            spk2utt,
            max_target_per_cond=args.max_target_trials_per_condition,
            neg_ratio=args.num_neg_target_ratio,
        )

    # Sort by utt1
    all_trials.sort(key=lambda x: x[0])

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        for u1, u2, label, cond in all_trials:
            f.write(f"{u1} {u2} {label} {cond}\n")

    print(f"Saved {len(all_trials)} trials to {args.output}")


if __name__ == "__main__":
    main()
