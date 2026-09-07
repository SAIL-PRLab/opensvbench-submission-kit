#!/usr/bin/env python3
"""
Cut short utterances from a wav.scp file using simple energy-based VAD.

For each utterance:
  1. Load audio
  2. Divide into 20ms frames and compute RMS energy per frame
  3. Find the first frame whose energy exceeds the vad_percentile-th percentile
     (i.e. skip leading silence to locate the speech start)
  4. Take a CONTINUOUS target_dur-second segment starting from that position
  5. Save to output directory as a .wav file

No frames are concatenated or reordered; the output is always a contiguous
slice of the original waveform.

Speaker-balanced sampling (when --utt2spk is provided):
  Per-speaker quota = max(2, max_utts // n_speakers).
  A first pass gives each speaker up to quota entries; remaining budget is
  filled by speakers that have extras, ensuring every speaker with >=2 utts
  contributes at least 2, and the overall distribution is as balanced as
  possible.

Output utt_id format:  {orig_utt_id}_dur_{X}s
  e.g. spk001-video001-00001 -> spk001-video001-00001_dur_1s
       spk001-video001-00001 -> spk001-video001-00001_dur_0.5s

Usage:
  python3 tools/cut_short_utterances.py \
      --wav-scp    path/to/wav.scp \
      --utt2spk    path/to/utt2spk \
      --output-dir path/to/short_utts/dur_1s \
      --target-dur 1.0 \
      --output-scp path/to/wav_short_dur_1s.scp \
      --max-utts   5000 \
      --nj 8
"""

import argparse
import os
import random
from collections import defaultdict

import numpy as np
import soundfile as sf
from multiprocessing import Pool
from functools import partial


def dur_to_str(dur: float) -> str:
    """Convert float duration to a filename-safe string.
    Examples: 0.5 -> '0.5s', 1.0 -> '1s', 2.0 -> '2s'
    """
    if dur == int(dur):
        return f"{int(dur)}s"
    return f"{dur}s"


def balanced_subsample(spk2entries: dict, max_utts: int,
                       rng: random.Random) -> list:
    """Speaker-balanced subsampling of utterance entries.

    Strategy (same as subsample_wavscp.py):
      quota = max(2, max_utts // n_speakers)
      First pass: each speaker contributes min(len(utts), quota) entries.
      Second pass: fill remaining budget with leftover entries from speakers
                   that have more than quota.

    Returns a list of (utt_id, wav_path) tuples of length <= max_utts.
    """
    if not spk2entries or max_utts <= 0:
        return [e for es in spk2entries.values() for e in es]

    n_spks = len(spk2entries)
    quota = max(2, max_utts // n_spks)

    selected = []
    leftover = []

    for spk, ents in spk2entries.items():
        shuffled = list(ents)
        rng.shuffle(shuffled)
        selected.extend(shuffled[:quota])
        if len(shuffled) > quota:
            leftover.extend(shuffled[quota:])

    # Fill remaining budget from leftovers
    remaining = max_utts - len(selected)
    if remaining > 0 and leftover:
        rng.shuffle(leftover)
        selected.extend(leftover[:remaining])

    rng.shuffle(selected)
    return selected[:max_utts]


def find_speech_start(signal: np.ndarray, sr: int,
                      vad_percentile: float = 20.0) -> int:
    """Find the sample index where the first active speech frame begins.

    Divides the signal into non-overlapping 20ms frames, computes RMS energy
    per frame, and returns the start sample of the first frame whose energy
    exceeds the vad_percentile-th percentile.

    Returns 0 if the entire signal is below the threshold or too short to frame.
    """
    frame_size = max(1, int(0.020 * sr))  # 20ms
    n_frames = len(signal) // frame_size
    if n_frames == 0:
        return 0

    frames = signal[: n_frames * frame_size].reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    if rms.max() == 0:
        return 0

    threshold = np.percentile(rms, vad_percentile)

    for i, r in enumerate(rms):
        if r > threshold:
            return i * frame_size

    return 0


def cut_continuous(signal: np.ndarray, sr: int, target_dur: float,
                   vad_percentile: float = 20.0):
    """Return a continuous target_dur-second slice starting from speech onset.

    Steps:
      1. Locate the first speech-active frame via energy VAD.
      2. Take signal[speech_start : speech_start + target_samples].
      3. If there is not enough signal after the speech start, fall back to
         starting from sample 0.
      4. Return None if the total signal is shorter than target_dur.
    """
    target_samples = int(target_dur * sr)

    if len(signal) < target_samples:
        return None

    start = find_speech_start(signal, sr, vad_percentile)

    if start + target_samples > len(signal):
        start = 0

    return signal[start: start + target_samples].astype(np.float32)


def process_entry(entry, output_dir: str, target_dur: float,
                  vad_percentile: float):
    """Process one (utt_id, wav_path) entry.

    Returns:
        (new_utt_id, out_path) on success, or None on failure.
    """
    utt_id, wav_path = entry
    try:
        signal, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if signal.ndim > 1:
            signal = signal[:, 0]

        cut = cut_continuous(signal, sr, target_dur, vad_percentile)
        if cut is None:
            return None

        safe_name = utt_id.replace("/", "_")
        out_path = os.path.join(output_dir, safe_name + ".wav")
        sf.write(out_path, cut, sr)

        new_utt_id = f"{utt_id}_dur_{dur_to_str(target_dur)}"
        return (new_utt_id, out_path)

    except Exception:
        return None


def get_args():
    parser = argparse.ArgumentParser(
        description="Cut short utterances from wav.scp using energy-based VAD"
    )
    parser.add_argument("--wav-scp", required=True,
                        help="Input wav.scp file")
    parser.add_argument("--utt2spk",
                        help="utt2spk file for speaker-balanced sampling "
                             "(strongly recommended; without it a flat shuffle "
                             "is used which may leave some speakers with only "
                             "1 utterance)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save cut .wav files")
    parser.add_argument("--target-dur", type=float, required=True,
                        help="Target duration in seconds (e.g. 0.5, 1, 2, 3)")
    parser.add_argument("--output-scp", required=True,
                        help="Output wav.scp path for the cut utterances")
    parser.add_argument("--max-utts", type=int, default=5000,
                        help="Maximum number of utterances to output "
                             "(0 = no limit, default: 5000). "
                             "Speaker-balanced when --utt2spk is provided.")
    parser.add_argument("--nj", type=int, default=4,
                        help="Number of parallel worker processes")
    parser.add_argument("--vad-percentile", type=float, default=20.0,
                        help="Percentile of frame energies used as VAD threshold. "
                             "Bottom vad_percentile%% of frames are treated as "
                             "silence (default: 20).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


def main():
    args = get_args()
    rng = random.Random(args.seed)

    # --- Load wav.scp ---
    entries = []
    with open(args.wav_scp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                entries.append((parts[0], parts[1]))

    # --- Select entries to process ---
    if args.utt2spk and args.max_utts > 0:
        # Speaker-balanced sampling: quota = max(2, max_utts // n_spks)
        utt2spk: dict[str, str] = {}
        with open(args.utt2spk) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    utt2spk[parts[0]] = parts[1]

        spk2entries: dict[str, list] = defaultdict(list)
        for e in entries:
            spk = utt2spk.get(e[0])
            if spk:
                spk2entries[spk].append(e)
            # entries with unknown speaker are silently skipped

        n_spks = len(spk2entries)
        quota = max(2, args.max_utts // n_spks) if n_spks else 0
        print(
            f"[cut_short_utterances] dur={dur_to_str(args.target_dur)}: "
            f"{n_spks} speakers found, quota={quota} utts/spk, "
            f"target total={args.max_utts}"
        )

        # Oversample each speaker to allow for VAD failures.
        oversample_spk2entries = {
            spk: ents for spk, ents in spk2entries.items()
        }
        # Use a larger candidate pool so VAD failures do not leave too few files.
        candidates = balanced_subsample(oversample_spk2entries,
                                        args.max_utts * 2, rng)
    else:
        # Fallback: flat shuffle, take first max_utts * 3 as candidate pool
        rng.shuffle(entries)
        candidates = entries[: args.max_utts * 3] if args.max_utts > 0 else entries
        if not args.utt2spk:
            print(
                f"[cut_short_utterances] WARNING: --utt2spk not provided; "
                f"using flat shuffle. Some speakers may end up with only 1 utt."
            )

    # --- Cut audio in parallel ---
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = os.path.dirname(os.path.abspath(args.output_scp))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    worker = partial(
        process_entry,
        output_dir=args.output_dir,
        target_dur=args.target_dur,
        vad_percentile=args.vad_percentile,
    )

    if args.nj > 1:
        with Pool(args.nj) as pool:
            results = pool.map(worker, candidates)
    else:
        results = [worker(e) for e in candidates]

    valid = [r for r in results if r is not None]

    # Trim to max_utts (preserving the original balanced order)
    if args.max_utts > 0 and len(valid) > args.max_utts:
        valid = valid[: args.max_utts]

    dur_str = dur_to_str(args.target_dur)
    print(
        f"[cut_short_utterances] dur={dur_str}: "
        f"{len(valid)} utterances kept "
        f"(candidates={len(candidates)}, "
        f"max_utts={args.max_utts if args.max_utts > 0 else 'unlimited'})"
    )

    with open(args.output_scp, "w") as f:
        for utt_id, path in valid:
            f.write(f"{utt_id} {path}\n")


if __name__ == "__main__":
    main()
