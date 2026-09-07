#!/usr/bin/env python3
"""Field-aware subsampling for HI-MIA.

Keep as many near-field utterances as possible under the clean-source budget,
then fill the remaining budget with speaker-balanced far-field utterances.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path


NEAR_POINT_IDS = {"7"}
FAR_POINT_IDS = {"2", "5", "6"}


def get_args():
    parser = argparse.ArgumentParser(description="Field-aware HI-MIA wav.scp subsampling")
    parser.add_argument("--wav-scp", required=True)
    parser.add_argument("--utt2spk", required=True)
    parser.add_argument("--max-utts", type=int, required=True)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--regen-spk2utt", default=None)
    return parser.parse_args()


def infer_field(utt_id: str) -> str | None:
    parts = utt_id.split("_")
    if len(parts) < 2:
        return None
    point_id = parts[1]
    if point_id in NEAR_POINT_IDS:
        return "near"
    if point_id in FAR_POINT_IDS:
        return "far"
    return None


def balanced_take(spk2utts: dict[str, list[str]], budget: int, rng: random.Random) -> list[str]:
    if budget <= 0:
        return []
    work = {spk: list(utts) for spk, utts in spk2utts.items() if utts}
    for utts in work.values():
        rng.shuffle(utts)
    speakers = list(work)
    rng.shuffle(speakers)

    selected: list[str] = []
    quota = max(1, budget // max(len(speakers), 1))
    remainder: list[str] = []
    for spk in speakers:
        utts = work[spk]
        take = min(len(utts), quota)
        selected.extend(utts[:take])
        remainder.extend(utts[take:])
        if len(selected) >= budget:
            return selected[:budget]

    if len(selected) < budget and remainder:
        rng.shuffle(remainder)
        selected.extend(remainder[: budget - len(selected)])
    return selected[:budget]


def regen_spk2utt(utt2spk_path: Path, spk2utt_path: Path) -> None:
    spk2utts: dict[str, list[str]] = defaultdict(list)
    with utt2spk_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                spk2utts[parts[1]].append(parts[0])
    with spk2utt_path.open("w") as f:
        for spk in sorted(spk2utts):
            f.write(f"{spk} {' '.join(sorted(spk2utts[spk]))}\n")


def main():
    args = get_args()
    rng = random.Random(args.seed)

    utt2path: dict[str, str] = {}
    with open(args.wav_scp, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt2path[parts[0]] = parts[1]

    utt2spk: dict[str, str] = {}
    with open(args.utt2spk, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                utt2spk[parts[0]] = parts[1]

    if len(utt2path) <= args.max_utts:
        print(f"[HI-MIA subsample] {len(utt2path)} utterances <= max_utts={args.max_utts}; skipping.")
        return

    near_by_spk: dict[str, list[str]] = defaultdict(list)
    far_by_spk: dict[str, list[str]] = defaultdict(list)
    unknown = 0
    for utt_id in utt2path:
        spk = utt2spk.get(utt_id)
        if spk is None:
            continue
        field = infer_field(utt_id)
        if field == "near":
            near_by_spk[spk].append(utt_id)
        elif field == "far":
            far_by_spk[spk].append(utt_id)
        else:
            unknown += 1

    near_all = [utt for utts in near_by_spk.values() for utt in utts]
    far_all = [utt for utts in far_by_spk.values() for utt in utts]
    print(
        f"[HI-MIA subsample] original={len(utt2path)} near={len(near_all)} "
        f"far={len(far_all)} unknown={unknown}"
    )

    if len(near_all) >= args.max_utts:
        selected = balanced_take(near_by_spk, args.max_utts, rng)
        print("[HI-MIA subsample] Near-field alone exceeds budget; taking balanced near-only subset.")
    else:
        far_budget = args.max_utts - len(near_all)
        selected = list(near_all)
        selected.extend(balanced_take(far_by_spk, far_budget, rng))
        print(
            f"[HI-MIA subsample] Keeping all {len(near_all)} near-field utterances and "
            f"{len(selected) - len(near_all)} far-field utterances."
        )

    selected_set = set(selected)
    near_selected = sum(1 for utt in selected_set if infer_field(utt) == "near")
    far_selected = sum(1 for utt in selected_set if infer_field(utt) == "far")
    speaker_count = len({utt2spk[utt] for utt in selected_set})
    print(
        f"[HI-MIA subsample] selected={len(selected_set)} speakers={speaker_count} "
        f"near={near_selected} far={far_selected}"
    )

    wav_scp_path = Path(args.wav_scp)
    utt2spk_path = Path(args.utt2spk)
    with wav_scp_path.open("w") as f:
        for utt in sorted(selected_set):
            f.write(f"{utt} {utt2path[utt]}\n")
    with utt2spk_path.open("w") as f:
        for utt in sorted(selected_set):
            f.write(f"{utt} {utt2spk[utt]}\n")

    if args.regen_spk2utt:
        regen_spk2utt(utt2spk_path, Path(args.regen_spk2utt))

    # Helpful diagnostics for near/far target-pair ceilings after subsampling.
    near_counts = Counter(utt2spk[utt] for utt in selected_set if infer_field(utt) == "near")
    far_counts = Counter(utt2spk[utt] for utt in selected_set if infer_field(utt) == "far")
    near_near_targets = sum(n * (n - 1) // 2 for n in near_counts.values())
    near_far_targets = sum(near_counts[spk] * far_counts[spk] for spk in set(near_counts) | set(far_counts))
    far_far_targets = sum(n * (n - 1) // 2 for n in far_counts.values())
    print(
        "[HI-MIA subsample] target upper bounds:",
        {
            "near_near": near_near_targets,
            "near_far": near_far_targets,
            "far_far": far_far_targets,
        },
    )


if __name__ == "__main__":
    main()
