#!/usr/bin/env python3
"""Prepare the VoxPopuli English aging benchmark subset for OpenSVBench."""

from __future__ import annotations

import argparse
import io
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from datasets import Audio, load_dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.scoring_utils import canonical_pair_key


GAP_BUCKETS = ("gap_1_3y", "gap_3_6y", "gap_6y_plus")
DATE_RE = re.compile(r"(\d{8})")


def get_args():
    parser = argparse.ArgumentParser(description="Prepare VoxPopuli English aging benchmark assets")
    parser.add_argument("--dataset-id", default="facebook/voxpopuli")
    parser.add_argument("--config", default="en")
    parser.add_argument("--split", default="train")
    parser.add_argument("--download-dir", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--prep-dir", required=True)
    parser.add_argument("--max-clean-utts", type=int, default=10000)
    parser.add_argument("--min-gap-years", type=float, default=1.0)
    parser.add_argument("--max-target-trials-per-bucket", type=int, default=5000)
    parser.add_argument("--nontarget-ratio", type=int, default=10)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--voiced-max-seconds", type=float, default=5.0)
    parser.add_argument("--vad-frame-ms", type=float, default=30.0)
    parser.add_argument("--vad-hop-ms", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=777)
    return parser.parse_args()


def parse_date(audio_id: str) -> datetime | None:
    match = DATE_RE.search(str(audio_id or ""))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d")
    except ValueError:
        return None


def gap_years(left: datetime, right: datetime) -> float:
    return abs((right - left).days) / 365.2425


def gap_bucket(gap: float, min_gap_years: float) -> str | None:
    if gap < min_gap_years:
        return None
    if gap < 3.0:
        return "gap_1_3y"
    if gap < 6.0:
        return "gap_3_6y"
    return "gap_6y_plus"


def sanitize_token(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    return clean.strip("-") or "unknown"


def build_spk2utt(utt2spk_path: Path, out_path: Path) -> None:
    spk2utts: dict[str, list[str]] = defaultdict(list)
    with utt2spk_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                spk2utts[parts[1]].append(parts[0])
    with out_path.open("w") as f:
        for spk in sorted(spk2utts):
            f.write(f"{spk} {' '.join(sorted(spk2utts[spk]))}\n")


def load_voxpopuli_stream(args, cache_dir: Path, decode_audio: bool):
    ds = load_dataset(
        args.dataset_id,
        args.config,
        split=args.split,
        streaming=True,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    )
    try:
        if "audio" in ds.features:
            if decode_audio:
                # Keep the raw Audio feature payload and decode with soundfile
                # below. Newer datasets versions may require torchcodec for
                # Audio(decode=True), which is unnecessary for this workflow.
                ds = ds.cast_column("audio", Audio(decode=False))
            else:
                ds = ds.remove_columns(["audio"])
    except TypeError:
        pass
    return ds


def local_parquet_files(download_dir: Path, config: str, split: str) -> list[Path]:
    parquet_dir = download_dir / "mirror_repo" / config
    return sorted(parquet_dir.glob(f"{split}-*.parquet"))


def normalize_arrow_value(value):
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def parquet_schema_names(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def read_cached_scan(scan_path: Path, require_parquet_file: bool = False) -> list[dict] | None:
    if not scan_path.is_file():
        return None
    rows = [json.loads(line) for line in scan_path.read_text().splitlines() if line.strip()]
    if not rows:
        return None
    if require_parquet_file and not all(Path(row.get("parquet_file", "")).is_file() for row in rows[:50]):
        return None
    print(f"[VoxPopuli] Reusing cached metadata scan: {scan_path} ({len(rows):,} rows)")
    return rows


def scan_local_parquet_metadata(args, parquet_files: list[Path], scan_path: Path) -> list[dict]:
    cached = read_cached_scan(scan_path, require_parquet_file=True)
    if cached is not None:
        return cached

    rows = []
    global_index = 0
    for parquet_file in parquet_files:
        schema_names = parquet_schema_names(parquet_file)
        required = {"audio_id", "speaker_id", "gender", "audio"}
        missing = sorted(required - schema_names)
        if missing:
            raise RuntimeError(f"Missing required VoxPopuli columns in {parquet_file}: {missing}")
        columns = ["audio_id", "speaker_id", "gender"]
        if "language" in schema_names:
            columns.append("language")

        pf = pq.ParquetFile(parquet_file)
        local_row = 0
        for batch in pf.iter_batches(batch_size=4096, columns=columns):
            table = batch.to_pydict()
            n_rows = batch.num_rows
            for row_offset in range(n_rows):
                audio_id = str(table["audio_id"][row_offset] or "")
                date = parse_date(audio_id)
                speaker = str(table["speaker_id"][row_offset] or "").strip()
                gender = str(table["gender"][row_offset] or "").strip().lower()
                language = str((table.get("language") or [args.config] * n_rows)[row_offset] or args.config).strip() or args.config
                if speaker and date is not None and gender:
                    rows.append(
                        {
                            "index": global_index,
                            "parquet_file": str(parquet_file.resolve()),
                            "local_row": local_row,
                            "audio_id": audio_id,
                            "speaker": speaker,
                            "gender": gender,
                            "language": language,
                            "date": date.strftime("%Y-%m-%d"),
                            "year": date.year,
                            "ordinal": date.toordinal(),
                        }
                    )
                local_row += 1
                global_index += 1
        print(f"[VoxPopuli] scanned local shard {parquet_file.name}; valid_rows={len(rows):,}")

    scan_path.parent.mkdir(parents=True, exist_ok=True)
    with scan_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"[VoxPopuli] Wrote local metadata scan cache: {scan_path} ({len(rows):,} rows)")
    return rows


def scan_metadata(args, cache_dir: Path, scan_path: Path) -> list[dict]:
    cached = read_cached_scan(scan_path)
    if cached is not None:
        return cached

    ds = load_voxpopuli_stream(args, cache_dir, decode_audio=False)
    rows = []
    for idx, row in enumerate(ds):
        audio_id = str(row.get("audio_id") or "")
        date = parse_date(audio_id)
        speaker = str(row.get("speaker_id") or "").strip()
        gender = str(row.get("gender") or "").strip().lower()
        language = str(row.get("language") or args.config).strip() or args.config
        if not speaker or date is None or not gender:
            continue
        rows.append(
            {
                "index": idx,
                "audio_id": audio_id,
                "speaker": speaker,
                "gender": gender,
                "language": language,
                "date": date.strftime("%Y-%m-%d"),
                "year": date.year,
                "ordinal": date.toordinal(),
            }
        )
        if idx and idx % 10_000 == 0:
            print(f"[VoxPopuli] scanned {idx:,} metadata rows; valid={len(rows):,}")

    scan_path.parent.mkdir(parents=True, exist_ok=True)
    with scan_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"[VoxPopuli] Wrote metadata scan cache: {scan_path}")
    return rows


def evenly_spaced_by_date(rows: list[dict], limit: int) -> list[dict]:
    ordered = sorted(rows, key=lambda item: (item["ordinal"], item["index"]))
    if len(ordered) <= limit:
        return ordered
    if limit <= 1:
        return [ordered[0]]
    chosen = []
    used = set()
    last = len(ordered) - 1
    for idx in range(limit):
        pos = round(idx * last / (limit - 1))
        if pos in used:
            continue
        used.add(pos)
        chosen.append(ordered[pos])
    if len(chosen) < limit:
        for pos, item in enumerate(ordered):
            if pos not in used:
                chosen.append(item)
                if len(chosen) >= limit:
                    break
    return sorted(chosen[:limit], key=lambda item: item["index"])


def select_candidate_records(rows: list[dict], max_clean_utts: int, min_gap_years: float) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["speaker"]].append(row)

    eligible = {}
    for speaker, speaker_rows in grouped.items():
        if len(speaker_rows) < 4:
            continue
        ordered = sorted(speaker_rows, key=lambda item: (item["ordinal"], item["index"]))
        span_years = (ordered[-1]["ordinal"] - ordered[0]["ordinal"]) / 365.2425
        if span_years >= min_gap_years:
            eligible[speaker] = ordered

    if not eligible:
        raise RuntimeError("No VoxPopuli speakers satisfy the minimum aging requirement.")

    candidate_limit = max_clean_utts * 2
    quota = max(4, max_clean_utts // max(1, len(eligible)))
    chosen_by_speaker = {
        speaker: evenly_spaced_by_date(rows, min(len(rows), quota))
        for speaker, rows in sorted(eligible.items())
    }

    candidates = []
    max_len = max(len(rows) for rows in chosen_by_speaker.values())
    for offset in range(max_len):
        for speaker in sorted(chosen_by_speaker):
            rows_for_speaker = chosen_by_speaker[speaker]
            if offset >= len(rows_for_speaker):
                continue
            candidates.append(rows_for_speaker[offset])
            if len(candidates) >= candidate_limit:
                break
        if len(candidates) >= candidate_limit:
            break

    candidates = sorted(candidates, key=lambda item: item["index"])
    print(
        f"[VoxPopuli] Selected {len(candidates):,} candidate utterances from "
        f"{len(eligible):,} eligible speakers (materialized cap={max_clean_utts:,})"
    )
    return candidates


def frame_rms(audio: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    if audio.size <= frame_len:
        return np.array([float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12))], dtype=np.float32)
    n_frames = 1 + (audio.size - frame_len) // hop
    framed = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, frame_len),
        strides=(audio.strides[0] * hop, audio.strides[0]),
        writeable=False,
    )
    energy = np.einsum("ij,ij->i", framed, framed, dtype=np.float64)
    return np.sqrt(np.maximum(energy / frame_len, 1e-12)).astype(np.float32)


def voiced_frame_mask(rms: np.ndarray) -> np.ndarray:
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    peak = float(np.percentile(rms, 95))
    floor = float(np.percentile(rms, 20))
    if peak <= 1e-6:
        return np.zeros(rms.shape, dtype=bool)
    threshold = max(floor * 2.0, peak * 0.10, 1e-4)
    mask = rms >= threshold
    if not mask.any():
        threshold = max(peak * 0.05, 5e-5)
        mask = rms >= threshold
    return mask


def select_voiced_excerpt(audio: np.ndarray, sr: int, target_seconds: float, frame_ms: float, hop_ms: float):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    source_duration_sec = float(audio.shape[0] / sr) if sr > 0 else 0.0
    target_samples = int(round(target_seconds * sr)) if sr > 0 and target_seconds > 0 else audio.shape[0]
    if audio.size == 0 or target_samples <= 0:
        return audio, {
            "clip_source_duration_sec": source_duration_sec,
            "clip_selected_duration_sec": source_duration_sec,
            "clip_start_sec": 0.0,
            "clip_end_sec": source_duration_sec,
        }

    frame_len = max(1, int(round(sr * frame_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    rms = frame_rms(audio, frame_len, hop)
    mask = voiced_frame_mask(rms)
    trim_start = 0
    trim_end = audio.shape[0]
    if mask.any():
        voiced_idx = np.flatnonzero(mask)
        trim_start = int(voiced_idx[0] * hop)
        trim_end = min(audio.shape[0], int(voiced_idx[-1] * hop + frame_len))
    trimmed = audio[trim_start:trim_end]
    if trimmed.size == 0:
        trimmed = audio
        trim_start = 0
        trim_end = audio.shape[0]

    if trimmed.shape[0] <= target_samples:
        selected = trimmed
        start_sample = trim_start
        end_sample = trim_start + trimmed.shape[0]
    else:
        rms = frame_rms(trimmed, frame_len, hop)
        mask = voiced_frame_mask(rms)
        window_frames = max(1, int(np.ceil(max(target_samples - frame_len, 0) / hop)) + 1)
        if rms.size <= window_frames:
            start_local = 0
        else:
            peak = float(max(rms.max(), 1e-6))
            weights = mask.astype(np.float32) + 0.05 * (rms / peak)
            prefix = np.concatenate(([0.0], np.cumsum(weights, dtype=np.float64)))
            scores = prefix[window_frames:] - prefix[:-window_frames]
            best_frame = int(np.argmax(scores))
            start_local = best_frame * hop
        end_local = min(trimmed.shape[0], start_local + target_samples)
        start_local = max(0, end_local - target_samples)
        selected = trimmed[start_local:end_local]
        start_sample = trim_start + start_local
        end_sample = trim_start + end_local

    return selected, {
        "clip_source_duration_sec": source_duration_sec,
        "clip_selected_duration_sec": float(selected.shape[0] / sr) if sr > 0 else 0.0,
        "clip_start_sec": float(start_sample / sr) if sr > 0 else 0.0,
        "clip_end_sec": float(end_sample / sr) if sr > 0 else 0.0,
    }


def decode_audio_value(audio_value) -> tuple[np.ndarray, int]:
    if isinstance(audio_value, dict):
        if "array" in audio_value and "sampling_rate" in audio_value:
            return np.asarray(audio_value["array"], dtype=np.float32), int(audio_value["sampling_rate"])
        audio_bytes = audio_value.get("bytes")
        audio_path = audio_value.get("path")
    else:
        audio_bytes = None
        audio_path = None
    if isinstance(audio_bytes, memoryview):
        audio_bytes = audio_bytes.tobytes()
    if audio_bytes:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    elif audio_path and Path(audio_path).is_file():
        data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    else:
        raise FileNotFoundError(f"Unable to decode VoxPopuli audio field: {audio_value}")
    if getattr(data, "ndim", 1) == 2:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), int(sr)


def duration_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(float(item) for item in values)

    def percentile(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
        return ordered[idx]

    return {
        "min": round(ordered[0], 6),
        "median": round(percentile(0.5), 6),
        "mean": round(sum(ordered) / len(ordered), 6),
        "p95": round(percentile(0.95), 6),
        "max": round(ordered[-1], 6),
    }


def reusable_manifest(manifest_path: Path, crop_cfg: dict, max_clean_utts: int) -> list[dict] | None:
    if not manifest_path.is_file():
        return None
    existing = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if (
        existing
        and len(existing) <= max_clean_utts
        and all(Path(item["audio_path"]).is_file() for item in existing)
        and all(all(item.get(key) == value for key, value in crop_cfg.items()) for item in existing)
    ):
        print(f"[VoxPopuli] Reusing materialized subset: {manifest_path} ({len(existing):,} utterances)")
        return existing
    return None


def build_crop_cfg(args) -> dict:
    return {
        "clip_strategy": "max_voiced_window",
        "clip_target_seconds": float(args.voiced_max_seconds),
        "clip_frame_ms": float(args.vad_frame_ms),
        "clip_hop_ms": float(args.vad_hop_ms),
        "min_duration": float(args.min_duration),
        "max_duration": float(args.max_duration),
    }


def write_materialized_item(args, raw_dir: Path, meta: dict, audio_value, crop_cfg: dict, source_index: int) -> tuple[dict | None, str | None]:
    try:
        audio, sr = decode_audio_value(audio_value)
    except Exception as exc:
        return None, f"decode_error:{exc}"

    source_duration = float(audio.shape[0] / sr) if sr > 0 else 0.0
    if source_duration < args.min_duration:
        return None, "too_short"
    if source_duration > args.max_duration:
        return None, "too_long"

    clipped_audio, clip_meta = select_voiced_excerpt(
        audio,
        sr,
        target_seconds=float(args.voiced_max_seconds),
        frame_ms=float(args.vad_frame_ms),
        hop_ms=float(args.vad_hop_ms),
    )
    if clip_meta["clip_selected_duration_sec"] < args.min_duration:
        return None, "selected_too_short"

    speaker_token = sanitize_token(meta["speaker"])
    date_token = meta["date"].replace("-", "")
    utt_id = f"voxpopuli-{speaker_token}-{date_token}-{source_index:06d}"
    out_dir = raw_dir / "audio" / speaker_token
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{utt_id}.wav"
    sf.write(out_path, clipped_audio, sr, subtype="PCM_16")

    item = {
        "source_index": source_index,
        "source_audio_id": meta["audio_id"],
        "utt_id": utt_id,
        "speaker": meta["speaker"],
        "gender": meta["gender"],
        "language": meta["language"],
        "date": meta["date"],
        "year": meta["year"],
        "ordinal": meta["ordinal"],
        "audio_path": str(out_path.resolve()),
    }
    item.update(crop_cfg)
    item.update(clip_meta)
    return item, None


def materialize_selection_from_parquet(args, candidates: list[dict], raw_dir: Path, manifest_path: Path) -> list[dict]:
    crop_cfg = build_crop_cfg(args)
    existing = reusable_manifest(manifest_path, crop_cfg, args.max_clean_utts)
    if existing is not None:
        return existing

    grouped: dict[str, dict[int, dict]] = defaultdict(dict)
    for item in candidates:
        parquet_file = item.get("parquet_file")
        local_row = item.get("local_row")
        if parquet_file is None or local_row is None:
            continue
        grouped[str(parquet_file)][int(local_row)] = item
    if not grouped:
        raise RuntimeError("No local Parquet row references found in selected VoxPopuli candidates.")

    written_rows = []
    skipped = Counter()

    with manifest_path.open("w") as f_manifest:
        for parquet_file_str in sorted(grouped):
            parquet_file = Path(parquet_file_str)
            wanted = grouped[parquet_file_str]
            schema_names = parquet_schema_names(parquet_file)
            columns = ["audio_id", "speaker_id", "gender", "audio"]
            if "language" in schema_names:
                columns.append("language")

            local_row = 0
            pf = pq.ParquetFile(parquet_file)
            for batch in pf.iter_batches(batch_size=64, columns=columns):
                table = batch.to_pydict()
                n_rows = batch.num_rows
                for row_offset in range(n_rows):
                    meta = wanted.get(local_row)
                    if meta is not None:
                        item, reason = write_materialized_item(
                            args,
                            raw_dir,
                            meta,
                            table["audio"][row_offset],
                            crop_cfg,
                            int(meta["index"]),
                        )
                        if item is None:
                            skipped[reason or "unknown"] += 1
                        else:
                            item["parquet_file"] = str(parquet_file.resolve())
                            item["local_row"] = int(local_row)
                            f_manifest.write(json.dumps(item, ensure_ascii=True) + "\n")
                            written_rows.append(item)
                            if len(written_rows) % 250 == 0:
                                print(f"[VoxPopuli] materialized {len(written_rows):,} / {args.max_clean_utts:,} utterances")
                            if len(written_rows) >= args.max_clean_utts:
                                break
                    local_row += 1
                if len(written_rows) >= args.max_clean_utts:
                    break
            print(f"[VoxPopuli] materialized from {parquet_file.name}; total={len(written_rows):,}")
            if len(written_rows) >= args.max_clean_utts:
                break

    print(f"[VoxPopuli] materialized={len(written_rows):,}; skipped={dict(sorted(skipped.items()))}")
    if len(written_rows) < 100:
        raise RuntimeError("Too few VoxPopuli utterances were materialized for a stable benchmark.")
    return written_rows


def materialize_selection(args, cache_dir: Path, candidates: list[dict], raw_dir: Path, manifest_path: Path) -> list[dict]:
    crop_cfg = build_crop_cfg(args)
    existing = reusable_manifest(manifest_path, crop_cfg, args.max_clean_utts)
    if existing is not None:
        return existing

    candidate_by_index = {item["index"]: item for item in candidates}
    selected_indices = set(candidate_by_index)
    max_index = max(selected_indices)
    audio_root = raw_dir / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    ds_audio = load_voxpopuli_stream(args, cache_dir, decode_audio=True)
    written_rows = []
    skipped = Counter()

    with manifest_path.open("w") as f_manifest:
        for idx, row in enumerate(ds_audio):
            if idx not in selected_indices:
                if idx > max_index:
                    break
                continue

            meta = candidate_by_index[idx]
            item, reason = write_materialized_item(args, raw_dir, meta, row["audio"], crop_cfg, idx)
            if item is None:
                skipped[reason or "unknown"] += 1
                continue
            f_manifest.write(json.dumps(item, ensure_ascii=True) + "\n")
            written_rows.append(item)
            if len(written_rows) % 250 == 0:
                print(f"[VoxPopuli] materialized {len(written_rows):,} / {args.max_clean_utts:,} utterances")
            if len(written_rows) >= args.max_clean_utts:
                break

    print(f"[VoxPopuli] materialized={len(written_rows):,}; skipped={dict(sorted(skipped.items()))}")
    if len(written_rows) < 100:
        raise RuntimeError("Too few VoxPopuli utterances were materialized for a stable benchmark.")
    return written_rows


def make_target_pairs(by_speaker: dict[str, list[dict]], max_per_bucket: int, min_gap_years: float, rng) -> dict[str, list[dict]]:
    target_pairs: dict[str, list[dict]] = {bucket: [] for bucket in GAP_BUCKETS}
    for speaker, rows in sorted(by_speaker.items()):
        for left, right in combinations(rows, 2):
            left_date = datetime.strptime(left["date"], "%Y-%m-%d")
            right_date = datetime.strptime(right["date"], "%Y-%m-%d")
            gap = gap_years(left_date, right_date)
            bucket = gap_bucket(gap, min_gap_years)
            if bucket is None:
                continue
            target_pairs[bucket].append(
                {
                    "utt1": left["utt_id"],
                    "utt2": right["utt_id"],
                    "label": "target",
                    "condition": bucket,
                    "source": "self_constructed",
                    "speaker1": left["speaker"],
                    "speaker2": right["speaker"],
                    "gender1": left["gender"],
                    "gender2": right["gender"],
                    "date1": left["date"],
                    "date2": right["date"],
                    "year1": left["year"],
                    "year2": right["year"],
                    "time_gap_years": gap,
                }
            )
    for bucket in GAP_BUCKETS:
        rng.shuffle(target_pairs[bucket])
        if len(target_pairs[bucket]) > max_per_bucket:
            target_pairs[bucket] = target_pairs[bucket][:max_per_bucket]
    return target_pairs


def make_nontarget_pairs(enriched: list[dict], target_pairs: dict[str, list[dict]], nontarget_ratio: int, min_gap_years: float, rng):
    seen_pairs = set()
    all_records = []
    for bucket in GAP_BUCKETS:
        for row in target_pairs[bucket]:
            key = canonical_pair_key(row["utt1"], row["utt2"])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            all_records.append(row)

    gender_groups: dict[str, list[dict]] = defaultdict(list)
    for item in enriched:
        gender_groups[item["gender"]].append(item)
    viable_genders = [gender for gender, rows in gender_groups.items() if len({row["speaker"] for row in rows}) >= 2]
    all_pool = list(enriched)

    nontarget_counts = Counter()
    nontarget_goal = {bucket: len(target_pairs[bucket]) * nontarget_ratio for bucket in GAP_BUCKETS}
    attempts = Counter()

    def sample_negative(bucket: str) -> dict | None:
        pool = gender_groups[rng.choice(viable_genders)] if viable_genders else all_pool
        if len(pool) < 2:
            return None
        for _ in range(100):
            left, right = rng.sample(pool, 2)
            if left["speaker"] == right["speaker"]:
                continue
            left_date = datetime.strptime(left["date"], "%Y-%m-%d")
            right_date = datetime.strptime(right["date"], "%Y-%m-%d")
            gap = gap_years(left_date, right_date)
            if gap_bucket(gap, min_gap_years) != bucket:
                continue
            key = canonical_pair_key(left["utt_id"], right["utt_id"])
            if key in seen_pairs:
                continue
            return {
                "utt1": left["utt_id"],
                "utt2": right["utt_id"],
                "label": "nontarget",
                "condition": bucket,
                "source": "matched_gender_time_gap",
                "speaker1": left["speaker"],
                "speaker2": right["speaker"],
                "gender1": left["gender"],
                "gender2": right["gender"],
                "date1": left["date"],
                "date2": right["date"],
                "year1": left["year"],
                "year2": right["year"],
                "time_gap_years": gap,
            }
        return None

    while any(nontarget_counts[bucket] < nontarget_goal[bucket] for bucket in GAP_BUCKETS):
        made_progress = False
        for bucket in GAP_BUCKETS:
            if nontarget_counts[bucket] >= nontarget_goal[bucket]:
                continue
            attempts[bucket] += 1
            if attempts[bucket] > nontarget_goal[bucket] * 100 + 2000:
                continue
            row = sample_negative(bucket)
            if row is None:
                continue
            key = canonical_pair_key(row["utt1"], row["utt2"])
            seen_pairs.add(key)
            all_records.append(row)
            nontarget_counts[bucket] += 1
            made_progress = True
        if not made_progress:
            break

    return all_records


def write_outputs(args, prep_dir: Path, enriched: list[dict], trial_rows: list[dict]) -> None:
    utt_rows = sorted(enriched, key=lambda item: item["utt_id"])
    trial_rows = sorted(trial_rows, key=lambda item: (item["utt1"], item["utt2"], item["label"], item["condition"]))

    wav_scp = prep_dir / "wav.scp"
    wav_all_scp = prep_dir / "wav_all.scp"
    utt2spk = prep_dir / "utt2spk"
    utt2gender = prep_dir / "utt2gender"
    utt2lang = prep_dir / "utt2lang"
    utt2date = prep_dir / "utt2date"
    utt2year = prep_dir / "utt2year"
    utt2duration = prep_dir / "utt2duration"
    spk2utt = prep_dir / "spk2utt"
    trials_path = prep_dir / "trials_voxpopuli_cross_time.txt"
    pair_meta = prep_dir / "pair_metadata_voxpopuli.csv"
    stats_source_path = prep_dir / "stats_voxpopuli_source.json"
    stats_trial_path = prep_dir / "stats_voxpopuli_cross_time.json"

    with (
        wav_scp.open("w") as f_wav,
        utt2spk.open("w") as f_spk,
        utt2gender.open("w") as f_gender,
        utt2lang.open("w") as f_lang,
        utt2date.open("w") as f_date,
        utt2year.open("w") as f_year,
        utt2duration.open("w") as f_duration,
    ):
        for row in utt_rows:
            f_wav.write(f"{row['utt_id']} {row['audio_path']}\n")
            f_spk.write(f"{row['utt_id']} {row['speaker']}\n")
            f_gender.write(f"{row['utt_id']} {row['gender']}\n")
            f_lang.write(f"{row['utt_id']} {row['language']}\n")
            f_date.write(f"{row['utt_id']} {row['date']}\n")
            f_year.write(f"{row['utt_id']} {row['year']}\n")
            f_duration.write(f"{row['utt_id']} {row['clip_selected_duration_sec']:.6f}\n")
    wav_all_scp.write_text(wav_scp.read_text())
    build_spk2utt(utt2spk, spk2utt)

    with trials_path.open("w") as f_trial:
        for row in trial_rows:
            f_trial.write(f"{row['utt1']} {row['utt2']} {row['label']} {row['condition']}\n")

    with pair_meta.open("w") as f_meta:
        f_meta.write(
            "utt1,utt2,label,condition,source,speaker1,speaker2,gender1,gender2,"
            "date1,date2,year1,year2,time_gap_years\n"
        )
        for row in trial_rows:
            f_meta.write(
                f"{row['utt1']},{row['utt2']},{row['label']},{row['condition']},{row['source']},"
                f"{row['speaker1']},{row['speaker2']},{row['gender1']},{row['gender2']},"
                f"{row['date1']},{row['date2']},{row['year1']},{row['year2']},{row['time_gap_years']:.6f}\n"
            )

    pair_distribution = Counter((row["label"], row["condition"]) for row in trial_rows)
    speaker_distribution = Counter(row["speaker"] for row in utt_rows)
    gender_distribution = Counter(row["gender"] for row in utt_rows)
    year_distribution = Counter(str(row["year"]) for row in utt_rows)

    source_stats = {
        "dataset": "voxpopuli",
        "dataset_id": args.dataset_id,
        "config": args.config,
        "split": args.split,
        "seed": args.seed,
        "clean_source_utterances": len(utt_rows),
        "speakers": len(speaker_distribution),
        "language_distribution": dict(sorted(Counter(row["language"] for row in utt_rows).items())),
        "gender_distribution": dict(sorted(gender_distribution.items())),
        "year_distribution": dict(sorted(year_distribution.items())),
        "source_limit": args.max_clean_utts,
        "source_duration_seconds": duration_summary([row["clip_source_duration_sec"] for row in utt_rows]),
        "selected_duration_seconds": duration_summary([row["clip_selected_duration_sec"] for row in utt_rows]),
        "audio_clip": {
            "clip_strategy": "max_voiced_window",
            "clip_target_seconds": args.voiced_max_seconds,
            "clip_frame_ms": args.vad_frame_ms,
            "clip_hop_ms": args.vad_hop_ms,
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
        },
    }
    trial_stats = {
        "dataset": "voxpopuli",
        "scenario": "voxpopuli_cross_time",
        "seed": args.seed,
        "total_trials": len(trial_rows),
        "target_trials": sum(1 for row in trial_rows if row["label"] == "target"),
        "nontarget_trials": sum(1 for row in trial_rows if row["label"] == "nontarget"),
        "condition_counts": dict(sorted(Counter(row["condition"] for row in trial_rows).items())),
        "pair_distribution": {f"{label}|{bucket}": count for (label, bucket), count in sorted(pair_distribution.items())},
        "time_gap_years": duration_summary([row["time_gap_years"] for row in trial_rows]),
    }
    stats_source_path.write_text(json.dumps(source_stats, indent=2, ensure_ascii=True) + "\n")
    stats_trial_path.write_text(json.dumps(trial_stats, indent=2, ensure_ascii=True) + "\n")

    print("[VoxPopuli] Statistics")
    print(f"  clean_source_utterances: {len(utt_rows)}")
    print(f"  speakers               : {len(speaker_distribution)}")
    for (label, bucket), count in sorted(pair_distribution.items()):
        print(f"  {label:10s} {bucket:12s}: {count}")
    print(f"[VoxPopuli] Wrote: {wav_scp}, {wav_all_scp}, {trials_path}, {stats_source_path}, {stats_trial_path}")


def validate_trials(enriched: list[dict], trial_rows: list[dict], min_gap_years: float) -> None:
    by_utt = {row["utt_id"]: row for row in enriched}
    seen = set()
    valid_conditions = set(GAP_BUCKETS)
    for row in trial_rows:
        if row["condition"] not in valid_conditions:
            raise RuntimeError(f"Invalid condition: {row['condition']}")
        if row["utt1"] not in by_utt or row["utt2"] not in by_utt:
            raise RuntimeError(f"Trial references unknown utterance: {row}")
        key = canonical_pair_key(row["utt1"], row["utt2"])
        if key in seen:
            raise RuntimeError(f"Duplicate trial pair: {key}")
        seen.add(key)
        same_speaker = by_utt[row["utt1"]]["speaker"] == by_utt[row["utt2"]]["speaker"]
        if row["label"] == "target" and not same_speaker:
            raise RuntimeError(f"Target pair has different speakers: {row}")
        if row["label"] == "nontarget" and same_speaker:
            raise RuntimeError(f"Nontarget pair has same speaker: {row}")
        left = datetime.strptime(by_utt[row["utt1"]]["date"], "%Y-%m-%d")
        right = datetime.strptime(by_utt[row["utt2"]]["date"], "%Y-%m-%d")
        gap = gap_years(left, right)
        if gap_bucket(gap, min_gap_years) != row["condition"]:
            raise RuntimeError(f"Gap bucket mismatch: {row} gap={gap}")


def main():
    args = get_args()
    rng = random.Random(args.seed)

    download_dir = Path(args.download_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    prep_dir = Path(args.prep_dir).resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    prep_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = download_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[VoxPopuli] Preparing {args.dataset_id}:{args.config}:{args.split}")
    parquet_files = local_parquet_files(download_dir, args.config, args.split)
    if parquet_files:
        print(f"[VoxPopuli] Using local Parquet shards: {len(parquet_files)} file(s)")
        for path in parquet_files:
            print(f"  {path}")
        metadata_rows = scan_local_parquet_metadata(args, parquet_files, raw_dir / "metadata_scan_parquet.jsonl")
        use_local_parquet = True
    else:
        print("[VoxPopuli] No local Parquet shards found; falling back to Hugging Face streaming.")
        metadata_rows = scan_metadata(args, cache_dir, raw_dir / "metadata_scan.jsonl")
        use_local_parquet = False

    candidates = select_candidate_records(metadata_rows, args.max_clean_utts, args.min_gap_years)
    if use_local_parquet:
        enriched = materialize_selection_from_parquet(args, candidates, raw_dir, raw_dir / "selected_manifest.jsonl")
    else:
        enriched = materialize_selection(args, cache_dir, candidates, raw_dir, raw_dir / "selected_manifest.jsonl")

    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in enriched:
        by_speaker[row["speaker"]].append(row)
    for rows in by_speaker.values():
        rows.sort(key=lambda item: (item["ordinal"], item["source_index"]))

    target_pairs = make_target_pairs(by_speaker, args.max_target_trials_per_bucket, args.min_gap_years, rng)
    trial_rows = make_nontarget_pairs(enriched, target_pairs, args.nontarget_ratio, args.min_gap_years, rng)
    validate_trials(enriched, trial_rows, args.min_gap_years)
    write_outputs(args, prep_dir, enriched, trial_rows)


if __name__ == "__main__":
    main()
