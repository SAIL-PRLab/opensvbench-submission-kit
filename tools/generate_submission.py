#!/usr/bin/env python3
"""Score prepared trials and create a submission archive."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark_catalog import (
    DEFAULT_DATA_ROOT,
    REPO_ROOT,
    dataset_by_id,
    iter_scenarios,
    scenario_by_id,
)
from tools.metric_aggregation import scenario_macro_metrics
from tools.scoring_utils import (
    load_trials,
    metrics_report_from_scored_df,
    score_trials_from_embeddings,
    write_score_file,
)


def get_args():
    parser = argparse.ArgumentParser(description="Generate and optionally upload an OpenSVBench submission")
    parser.add_argument("--data", default=str(DEFAULT_DATA_ROOT), help="Benchmark data root")
    parser.add_argument("--model-name", default=None, help="Model path or checkpoint identifier")
    parser.add_argument(
        "--backend",
        default=None,
        choices=["wespeaker", "huggingface", "modelscope", "speechbrain", "redimnet2"],
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--extract-batch-size",
        type=int,
        default=0,
        help="Optional override for extract_embeddings.py --batch-size.",
    )
    parser.add_argument(
        "--extract-waveform-cmvn",
        choices=["none", "utterance"],
        default="none",
        help="Pass through to extract_embeddings.py --waveform-cmvn.",
    )
    parser.add_argument(
        "--extract-max-chunk-seconds",
        type=float,
        default=20.0,
        help="Pass through to extract_embeddings.py --max-chunk-seconds.",
    )
    parser.add_argument(
        "--speechbrain-normalize",
        choices=["auto", "true", "false"],
        default="auto",
        help="Pass through to extract_embeddings.py --speechbrain-normalize.",
    )
    parser.add_argument("--output-dir", default="submissions")
    parser.add_argument("--submission-name", default=None)
    parser.add_argument("--p-target", type=float, default=0.05)
    parser.add_argument("--force-extract", action="store_true", help="Recompute embeddings even if cached")
    parser.add_argument(
        "--submission-mode",
        choices=["full-core", "partial"],
        default="full-core",
        help="Use partial mode when only a subset of scenarios is available.",
    )
    parser.add_argument(
        "--scenario-ids",
        nargs="+",
        default=None,
        help="Optional subset of core scenario ids. Using a subset requires --submission-mode partial.",
    )
    parser.add_argument("--list-scenarios", action="store_true", help="List current core scenario ids and exit")
    parser.add_argument("--precheck-only", action="store_true", help="Check local benchmark readiness and exit")
    args = parser.parse_args()
    if not args.list_scenarios and not args.precheck_only:
        if not args.model_name or not args.backend:
            parser.error("--model-name and --backend are required unless using --precheck-only or --list-scenarios")
    return args


def print_scenarios() -> None:
    print("Current core scenarios:")
    for spec in iter_scenarios(collection="core"):
        print(
            f"  {spec.scenario_id:<18} "
            f"dataset={spec.dataset_id:<12} "
            f"variant={spec.variant:<6} "
            f"score_file={spec.submission_filename}"
        )


def resolve_requested_scenarios(selected_ids: list[str] | None) -> list:
    core_specs = list(iter_scenarios(collection="core"))
    if not selected_ids:
        return core_specs
    selected_set = []
    seen = set()
    for scenario_id in selected_ids:
        try:
            spec = scenario_by_id(scenario_id)
        except KeyError as exc:
            raise ValueError(f"Unknown scenario id: {scenario_id}") from exc
        if spec.collection != "core":
            raise ValueError(f"Scenario {scenario_id} is not part of the current core collection.")
        if scenario_id not in seen:
            seen.add(scenario_id)
            selected_set.append(scenario_id)
    return [spec for spec in core_specs if spec.scenario_id in seen]


def resolve_wav_scp(prep_dir: Path) -> Path | None:
    wav_all = prep_dir / "wav_all.scp"
    wav_clean = prep_dir / "wav.scp"
    if wav_all.is_file():
        return wav_all
    if wav_clean.is_file():
        return wav_clean
    return None


def read_wav_scp(wav_scp_path: Path) -> dict[str, str]:
    mapping = {}
    with wav_scp_path.open("r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def load_embedding_cache(emb_file: Path) -> dict:
    with emb_file.open("rb") as f:
        embeddings = pickle.load(f)
    if not isinstance(embeddings, dict):
        raise ValueError(f"{emb_file} did not contain a dict embedding cache")
    return embeddings


def save_embedding_cache(emb_file: Path, embeddings: dict) -> None:
    tmp_path = emb_file.with_suffix(emb_file.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(embeddings, f)
    os.replace(tmp_path, emb_file)


def collect_required_utts(specs: list, data_root: Path) -> dict[Path, set[str]]:
    prep_to_required = defaultdict(set)
    for spec in specs:
        trials = load_trials(spec.trial_path(data_root))
        required = prep_to_required[spec.prep_dir(data_root)]
        for rec in trials:
            required.add(rec.utt1)
            required.add(rec.utt2)
    return prep_to_required


def write_filtered_wav_scp(
    wav_scp: Path,
    selected_utts: set[str],
    output_scp: Path,
) -> tuple[int, int]:
    selected = 0
    total = 0
    with wav_scp.open("r") as src, output_scp.open("w") as dst:
        for line in src:
            total += 1
            utt_id = line.strip().split(maxsplit=1)[0]
            if utt_id in selected_utts:
                dst.write(line)
                selected += 1
    return selected, total


def precheck_scenarios(specs: list, data_root: Path) -> tuple[dict, list, list]:
    ready_specs = []
    missing_entries = []
    checked_entries = []
    for spec in specs:
        dataset_spec = dataset_by_id(spec.dataset_id)
        prep_dir = spec.prep_dir(data_root)
        trial_path = spec.trial_path(data_root)
        wav_scp = resolve_wav_scp(prep_dir)
        issues = []
        if not prep_dir.is_dir():
            issues.append(f"missing prepared dir: {prep_dir}")
        if not trial_path.is_file():
            issues.append(f"missing trial file: {trial_path}")
        if wav_scp is None:
            issues.append(f"missing wav.scp or wav_all.scp under: {prep_dir}")

        entry = {
            "scenario_id": spec.scenario_id,
            "scenario_name": spec.display_name,
            "dataset_id": spec.dataset_id,
            "variant": spec.variant,
            "status": "ready" if not issues else "missing",
            "trial_path": str(trial_path),
            "prep_dir": str(prep_dir),
            "wav_scp": str(wav_scp) if wav_scp else None,
            "issues": issues,
            "download_hint": dataset_spec.download_method,
            "readme_path": str(dataset_spec.readme_path()),
        }
        checked_entries.append(entry)
        if issues:
            missing_entries.append(entry)
        else:
            ready_specs.append(spec)

    report = {
        "requested_scenario_count": len(specs),
        "ready_scenario_count": len(ready_specs),
        "missing_scenario_count": len(missing_entries),
        "ready_scenario_ids": [spec.scenario_id for spec in ready_specs],
        "missing_scenario_ids": [item["scenario_id"] for item in missing_entries],
        "scenarios": checked_entries,
    }
    return report, ready_specs, missing_entries


def print_precheck_report(report: dict) -> None:
    print(
        "[Submission] Precheck summary: "
        f"requested={report['requested_scenario_count']} "
        f"ready={report['ready_scenario_count']} "
        f"missing={report['missing_scenario_count']}"
    )
    if report["missing_scenario_count"] == 0:
        return
    print("[Submission] Missing or unavailable scenarios:")
    for item in report["scenarios"]:
        if item["status"] != "missing":
            continue
        print(f"  - {item['scenario_id']} ({item['dataset_id']}/{item['variant']})")
        for issue in item["issues"]:
            print(f"      issue: {issue}")
        print(f"      hint : {item['download_hint']}")
        print(f"      readme: {item['readme_path']}")


def ensure_embeddings(
    prep_dir: Path,
    model_name: str,
    backend: str,
    gpu: int,
    force_extract: bool,
    required_utts: set[str] | None = None,
    extract_batch_size: int = 0,
    extract_waveform_cmvn: str = "none",
    extract_max_chunk_seconds: float = 20.0,
    speechbrain_normalize: str = "auto",
) -> Path:
    model_id = os.path.basename(model_name.rstrip("/"))
    wav_scp = resolve_wav_scp(prep_dir)
    if wav_scp is None:
        raise FileNotFoundError(f"Missing wav.scp or wav_all.scp under {prep_dir}")
    emb_suffix = "embeddings_all" if wav_scp.name == "wav_all.scp" else "embeddings"
    emb_file = prep_dir / f"{emb_suffix}_{model_id}.pkl"
    invalid_cache = emb_file.is_file() and emb_file.stat().st_size <= 16
    extract_subset = required_utts

    if emb_file.is_file() and not force_extract and not invalid_cache and required_utts:
        try:
            cached_embeddings = load_embedding_cache(emb_file)
        except Exception:
            invalid_cache = True
        else:
            missing_required = required_utts - set(cached_embeddings)
            if not missing_required:
                print(f"[Submission] Reusing cached embeddings: {emb_file}")
                return emb_file
            extract_subset = missing_required
            print(
                f"[Submission] Cache missing {len(missing_required)} required utterances "
                f"for {prep_dir.name}; extracting only the remainder."
            )

    if force_extract or not emb_file.is_file() or invalid_cache or extract_subset is not None:
        if force_extract and emb_file.is_file():
            # Ensure a true recomputation instead of resuming stale embeddings.
            emb_file.unlink()
            invalid_cache = False
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "extract_embeddings.py"),
            "--output",
            str(emb_file),
            "--model-name",
            model_name,
            "--backend",
            backend,
            "--gpu",
            str(gpu),
            "--waveform-cmvn",
            extract_waveform_cmvn,
            "--max-chunk-seconds",
            str(extract_max_chunk_seconds),
            "--speechbrain-normalize",
            speechbrain_normalize,
        ]
        if extract_batch_size > 0:
            cmd.extend(["--batch-size", str(extract_batch_size)])

        extract_wav_scp = wav_scp
        tmpdir_ctx = tempfile.TemporaryDirectory(prefix="opensvbench_extract_", dir=str(prep_dir))
        try:
            if extract_subset is not None:
                tmp_wav_scp = Path(tmpdir_ctx.name) / "required.scp"
                selected, total = write_filtered_wav_scp(wav_scp, extract_subset, tmp_wav_scp)
                if selected == 0:
                    raise ValueError(f"No required utterances were found in {wav_scp}")
                extract_wav_scp = tmp_wav_scp
                print(
                    f"[Submission] Extracting only required utterances for {prep_dir.name}: "
                    f"{selected}/{total}"
                )

            cmd[2:2] = ["--wav-scp", str(extract_wav_scp)]
            if invalid_cache and not force_extract:
                print(f"[Submission] Cached embeddings look empty/corrupt, regenerating: {emb_file}")
            print("[Submission] Extracting embeddings:", " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
        finally:
            tmpdir_ctx.cleanup()
    else:
        print(f"[Submission] Reusing cached embeddings: {emb_file}")
    return emb_file


def backfill_missing_embeddings(
    prep_dir: Path,
    emb_file: Path,
    embeddings: dict,
    trials,
    model_name: str,
    backend: str,
    gpu: int,
    extract_waveform_cmvn: str = "none",
    extract_max_chunk_seconds: float = 20.0,
    speechbrain_normalize: str = "auto",
) -> tuple[dict, int]:
    wav_scp = resolve_wav_scp(prep_dir)
    if wav_scp is None:
        return embeddings, 0

    required_utts = {rec.utt1 for rec in trials} | {rec.utt2 for rec in trials}
    missing_utts = sorted(utt for utt in required_utts if utt not in embeddings)
    if not missing_utts:
        return embeddings, 0

    wav_map = read_wav_scp(wav_scp)
    available_missing = [utt for utt in missing_utts if utt in wav_map]
    unresolved = len(missing_utts) - len(available_missing)
    if not available_missing:
        return embeddings, len(missing_utts)

    with tempfile.TemporaryDirectory(prefix="opensvbench_missing_", dir=str(prep_dir)) as tmpdir:
        tmpdir_path = Path(tmpdir)
        tmp_wav_scp = tmpdir_path / "missing.scp"
        tmp_emb = tmpdir_path / "missing.pkl"
        with tmp_wav_scp.open("w") as f:
            for utt_id in available_missing:
                f.write(f"{utt_id} {wav_map[utt_id]}\n")

        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "extract_embeddings.py"),
            "--wav-scp",
            str(tmp_wav_scp),
            "--output",
            str(tmp_emb),
            "--model-name",
            model_name,
            "--backend",
            backend,
            "--gpu",
            str(gpu),
            "--waveform-cmvn",
            extract_waveform_cmvn,
            "--max-chunk-seconds",
            str(extract_max_chunk_seconds),
            "--speechbrain-normalize",
            speechbrain_normalize,
        ]
        print(
            f"[Submission] Backfilling {len(available_missing)} missing utterance embeddings "
            f"for {prep_dir.name}: {' '.join(cmd)}"
        )
        result = subprocess.run(cmd, check=False, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(
                f"[Submission] Warning: backfill extractor exited with {result.returncode}; "
                "continuing with any recovered embeddings."
            )
        recovered = load_embedding_cache(tmp_emb) if tmp_emb.is_file() else {}

    if recovered:
        embeddings.update(recovered)
        save_embedding_cache(emb_file, embeddings)

    remaining = unresolved + sum(1 for utt in available_missing if utt not in embeddings)
    return embeddings, remaining


def main():
    args = get_args()
    if args.list_scenarios:
        print_scenarios()
        return

    data_root = Path(args.data).resolve()
    out_root = Path(args.output_dir).resolve()
    requested_specs = resolve_requested_scenarios(args.scenario_ids)
    all_core_specs = list(iter_scenarios(collection="core"))
    full_core_mode = args.submission_mode == "full-core"
    submission_mode_label = "full-core" if full_core_mode else "partial"

    if full_core_mode and len(requested_specs) != len(all_core_specs):
        raise ValueError(
            "Full-core submissions must cover all current core scenarios. "
            "Use --submission-mode partial when submitting only a subset."
        )

    model_id = os.path.basename(args.model_name.rstrip("/")) if args.model_name else "precheck"
    submission_name = args.submission_name or f"{model_id}_{submission_mode_label}"
    submission_root = out_root / submission_name
    scores_dir = submission_root / "scores"
    metrics_dir = submission_root / "metrics"
    submission_root.mkdir(parents=True, exist_ok=True)
    if scores_dir.exists():
        shutil.rmtree(scores_dir)
    if metrics_dir.exists():
        shutil.rmtree(metrics_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    precheck_report, ready_specs, missing_entries = precheck_scenarios(requested_specs, data_root)
    precheck_path = submission_root / "precheck.json"
    precheck_path.write_text(json.dumps(precheck_report, indent=2, ensure_ascii=True) + "\n")
    print_precheck_report(precheck_report)
    print(f"[Submission] Precheck report written to {precheck_path}")

    if args.precheck_only:
        raise SystemExit(0 if precheck_report["missing_scenario_count"] == 0 else 2)

    if full_core_mode and missing_entries:
        missing_ids = ", ".join(precheck_report["missing_scenario_ids"])
        raise ValueError(
            "Full-core submission precheck failed; the following scenarios are unavailable: "
            f"{missing_ids}. See {precheck_path}"
        )

    active_specs = requested_specs if full_core_mode else ready_specs
    if not active_specs:
        raise ValueError("No ready scenarios are available to score.")

    created_at = datetime.now(timezone.utc).isoformat()
    prep_dirs = sorted({spec.prep_dir(data_root) for spec in active_specs})
    skipped_scenarios: list[dict[str, str]] = []
    emb_files = {}
    loaded_embeddings = {}
    prep_required_utts = collect_required_utts(active_specs, data_root)

    prep_to_specs = defaultdict(list)
    for spec in active_specs:
        prep_to_specs[spec.prep_dir(data_root)].append(spec)

    for prep_dir in prep_dirs:
        try:
            emb_files[prep_dir] = ensure_embeddings(
                prep_dir=prep_dir,
                model_name=args.model_name,
                backend=args.backend,
                gpu=args.gpu,
                force_extract=args.force_extract,
                required_utts=prep_required_utts.get(prep_dir),
                extract_batch_size=args.extract_batch_size,
                extract_waveform_cmvn=args.extract_waveform_cmvn,
                extract_max_chunk_seconds=args.extract_max_chunk_seconds,
                speechbrain_normalize=args.speechbrain_normalize,
            )
        except Exception as exc:
            if full_core_mode:
                raise
            for spec in prep_to_specs[prep_dir]:
                skipped_scenarios.append(
                    {
                        "scenario_id": spec.scenario_id,
                        "stage": "embedding",
                        "reason": str(exc),
                    }
                )
            print(f"[Submission] Skipping {prep_dir} because embedding extraction failed: {exc}")

    preview_rows = []
    completed_scenario_ids = []
    for spec in active_specs:
        prep_dir = spec.prep_dir(data_root)
        if prep_dir not in emb_files:
            continue
        try:
            if prep_dir not in loaded_embeddings:
                try:
                    loaded_embeddings[prep_dir] = load_embedding_cache(emb_files[prep_dir])
                except Exception:
                    emb_files[prep_dir] = ensure_embeddings(
                        prep_dir=prep_dir,
                        model_name=args.model_name,
                        backend=args.backend,
                        gpu=args.gpu,
                        force_extract=True,
                        required_utts=prep_required_utts.get(prep_dir),
                        extract_batch_size=args.extract_batch_size,
                        extract_waveform_cmvn=args.extract_waveform_cmvn,
                        extract_max_chunk_seconds=args.extract_max_chunk_seconds,
                        speechbrain_normalize=args.speechbrain_normalize,
                    )
                    loaded_embeddings[prep_dir] = load_embedding_cache(emb_files[prep_dir])
            trials = load_trials(spec.trial_path(data_root))
            loaded_embeddings[prep_dir], remaining_missing_utts = backfill_missing_embeddings(
                prep_dir=prep_dir,
                emb_file=emb_files[prep_dir],
                embeddings=loaded_embeddings[prep_dir],
                trials=trials,
                model_name=args.model_name,
                backend=args.backend,
                gpu=args.gpu,
                extract_waveform_cmvn=args.extract_waveform_cmvn,
                extract_max_chunk_seconds=args.extract_max_chunk_seconds,
                speechbrain_normalize=args.speechbrain_normalize,
            )
            if remaining_missing_utts > 0:
                raise ValueError(
                    f"{remaining_missing_utts} required utterance embeddings are still missing "
                    f"for {spec.scenario_id}; refusing to score an incomplete trial set"
                )
            scored_df, missing = score_trials_from_embeddings(loaded_embeddings[prep_dir], trials)
            if missing > 0:
                raise ValueError(f"{missing} trial pairs were missing embeddings")
            if len(scored_df) and not np.isfinite(scored_df["score"].to_numpy(dtype=float)).all():
                bad_rows = scored_df[~np.isfinite(scored_df["score"].to_numpy(dtype=float))][["utt1", "utt2"]].head(5)
                raise ValueError(
                    f"produced non-finite scores; sample pairs: {bad_rows.to_dict(orient='records')}"
                )

            score_path = scores_dir / spec.submission_filename
            write_score_file(scored_df, score_path)

            report_df = metrics_report_from_scored_df(scored_df, p_target=args.p_target)
            metrics_path = metrics_dir / f"{spec.scenario_id}_metrics.csv"
            report_df.to_csv(metrics_path, index=False)

            overall_row = (
                report_df.iloc[0].to_dict()
                if len(report_df) > 0
                else {"EER(%)": float("nan"), "minDCF": float("nan")}
            )
            preview_rows.append(
                {
                    "ScenarioId": spec.scenario_id,
                    "Scenario": spec.display_name,
                    "Variant": spec.variant,
                    "NumTrials": int(overall_row.get("NumTrials", 0)),
                    "EER(%)": overall_row.get("EER(%)"),
                    "minDCF": overall_row.get("minDCF"),
                    "ScoreFile": score_path.name,
                }
            )
            completed_scenario_ids.append(spec.scenario_id)
            print(f"[Submission] Wrote {score_path}")
        except Exception as exc:
            if full_core_mode:
                raise
            skipped_scenarios.append(
                {
                    "scenario_id": spec.scenario_id,
                    "stage": "scoring",
                    "reason": str(exc),
                }
            )
            print(f"[Submission] Skipping {spec.scenario_id} because scoring failed: {exc}")

    if not preview_rows:
        raise ValueError("No scenario scores were generated. Check precheck.json and skipped_scenarios.")

    preview_df = pd.DataFrame(preview_rows)
    preview_path = submission_root / "local_preview.csv"
    preview_df.to_csv(preview_path, index=False)

    aggregate = scenario_macro_metrics(
        preview_rows,
        trial_set_id_key="ScenarioId",
        eer_key="EER(%)",
        min_dcf_key="minDCF",
    )
    aggregate_eer = float(aggregate["aggregate_eer"]) if aggregate["aggregate_eer"] is not None else float("nan")
    aggregate_min_dcf = (
        float(aggregate["aggregate_min_dcf"]) if aggregate["aggregate_min_dcf"] is not None else float("nan")
    )
    trial_set_macro_eer = float(preview_df["EER(%)"].mean()) if len(preview_df) else float("nan")
    trial_set_macro_min_dcf = float(preview_df["minDCF"].mean()) if len(preview_df) else float("nan")

    missing_scenario_ids = sorted(
        set(precheck_report["missing_scenario_ids"]) | {item["scenario_id"] for item in skipped_scenarios}
    )
    ranking_status = "complete"
    if len(completed_scenario_ids) != len(all_core_specs) or set(completed_scenario_ids) != {
        spec.scenario_id for spec in all_core_specs
    }:
        ranking_status = "incomplete"

    warnings = []
    if missing_entries:
        warnings.append(
            "Precheck found unavailable scenarios: " + ", ".join(precheck_report["missing_scenario_ids"])
        )
    if skipped_scenarios:
        warnings.append(
            "Some scenarios were skipped during packaging: "
            + ", ".join(sorted({item["scenario_id"] for item in skipped_scenarios}))
        )
    if ranking_status != "complete":
        warnings.append("This submission does not cover the full current core scenario set and is not eligible for public ranking.")

    metadata = {
        "submission_name": submission_name,
        "model_name": args.model_name,
        "model_id": model_id,
        "backend": args.backend,
        "created_at_utc": created_at,
        "extraction_config": {
            "backend": args.backend,
            "force_extract": bool(args.force_extract),
            "batch_size": int(args.extract_batch_size),
            "waveform_cmvn": args.extract_waveform_cmvn,
            "max_chunk_seconds": float(args.extract_max_chunk_seconds),
            "speechbrain_normalize": args.speechbrain_normalize,
            "wespeaker_chunk_policy": (
                "deterministic_max_chunk"
                if args.backend == "wespeaker" and args.extract_max_chunk_seconds > 0
                else "single_utterance"
            ),
        },
        "submission_mode_requested": submission_mode_label,
        "requested_scenario_ids": [spec.scenario_id for spec in requested_specs],
        "submitted_scenario_ids": completed_scenario_ids,
        "missing_scenario_ids": missing_scenario_ids,
        "requested_scenario_count": len(requested_specs),
        "scenario_count": len(completed_scenario_ids),
        "required_scenario_count": len(all_core_specs),
        "missing_scenario_count": len(all_core_specs) - len(completed_scenario_ids),
        "ranking_status": ranking_status,
        "official_ranking_eligible": ranking_status == "complete",
        "public_ranking_eligible": ranking_status == "complete",
        "warnings": warnings,
        "precheck": precheck_report,
        "skipped_scenarios": skipped_scenarios,
        "aggregate_metric": "scenario_macro",
        "aggregate_definition": "source_balanced_by_dataset_id",
        "aggregate_macro_eer": aggregate_eer,
        "aggregate_macro_min_dcf": aggregate_min_dcf,
        "aggregate_scenario_macro_eer": aggregate_eer,
        "aggregate_scenario_macro_min_dcf": aggregate_min_dcf,
        "aggregate_trial_set_macro_eer": trial_set_macro_eer,
        "aggregate_trial_set_macro_min_dcf": trial_set_macro_min_dcf,
        "aggregate_scenario_components": aggregate["eer_components"],
        "score_format": {"columns": ["utt1", "utt2", "score"]},
    }
    metadata_path = submission_root / "submission.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n")

    zip_path = out_root / f"{submission_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(metadata_path, arcname="submission.json")
        for score_file in sorted(scores_dir.glob("*.csv")):
            zf.write(score_file, arcname=f"scores/{score_file.name}")

    report_path = submission_root / "generation_report.json"
    report_path.write_text(
        json.dumps(
            {
                "submission_name": submission_name,
                "ranking_status": ranking_status,
                "submitted_scenario_ids": completed_scenario_ids,
                "missing_scenario_ids": missing_scenario_ids,
                "skipped_scenarios": skipped_scenarios,
                "aggregate_metric": "scenario_macro",
                "aggregate_definition": "source_balanced_by_dataset_id",
                "aggregate_macro_eer": aggregate_eer,
                "aggregate_macro_min_dcf": aggregate_min_dcf,
                "aggregate_scenario_macro_eer": aggregate_eer,
                "aggregate_scenario_macro_min_dcf": aggregate_min_dcf,
                "aggregate_trial_set_macro_eer": trial_set_macro_eer,
                "aggregate_trial_set_macro_min_dcf": trial_set_macro_min_dcf,
                "precheck_path": str(precheck_path),
                "zip_path": str(zip_path),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n"
    )

    print(f"[Submission] Zip written to {zip_path}")
    print(f"[Submission] Local preview written to {preview_path}")
    print(f"[Submission] Generation report written to {report_path}")
    print(
        "[Submission] Coverage: "
        f"{len(completed_scenario_ids)}/{len(all_core_specs)} core scenarios "
        f"({ranking_status})"
    )

    print("[Submission] Upload the zip manually on the maintained OpenSVBench website at /submit/model.")


if __name__ == "__main__":
    main()
