#!/usr/bin/env python3
"""Shared scoring helpers for OpenSVBench."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PAIR_DELIMITER = "|||"
DEFAULT_CONDITION = "default"
REFERENCE_INDEX_VERSION = "pair_code_v1"


@dataclass(frozen=True)
class TrialRecord:
    utt1: str
    utt2: str
    label: str
    condition: str


@dataclass(frozen=True)
class ScenarioEvalIndex:
    scenario_id: str
    root: Path
    trial_count: int
    unique_pair_count: int
    target_count: int
    nontarget_count: int
    condition_to_id: dict[str, int]
    condition_names: tuple[str, ...]
    utt_to_id: dict[str, int]
    pairs: np.memmap
    labels: np.memmap
    conditions: np.memmap
    aggregation_version: str


@dataclass(frozen=True)
class IndexEvaluationResult:
    metrics_rows: list[dict[str, float | str | int]]
    overall_eer: float
    overall_min_dcf: float


def canonical_pair_key(utt1: str, utt2: str) -> tuple[str, str]:
    return (utt1, utt2) if utt1 <= utt2 else (utt2, utt1)


def pair_code_from_utt_ids(utt1_id: int, utt2_id: int) -> np.uint64:
    left_id = utt1_id if utt1_id <= utt2_id else utt2_id
    right_id = utt2_id if utt1_id <= utt2_id else utt1_id
    return (np.uint64(left_id) << np.uint64(32)) | np.uint64(right_id)


def cosine_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    n1 = np.linalg.norm(e1)
    n2 = np.linalg.norm(e2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(e1, e2) / (n1 * n2))


def _sorted_det_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must be 1D arrays of the same length.")

    target_total = int(np.count_nonzero(labels == 1))
    nontarget_total = int(np.count_nonzero(labels == 0))
    if target_total == 0 or nontarget_total == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    tp = np.cumsum(sorted_labels == 1, dtype=np.int64)
    fp = np.cumsum(sorted_labels == 0, dtype=np.int64)
    distinct = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]

    tp = tp[distinct]
    fp = fp[distinct]
    thresholds = sorted_scores[distinct]

    fpr = fp.astype(np.float64) / float(nontarget_total)
    fnr = (target_total - tp).astype(np.float64) / float(target_total)

    fpr = np.concatenate([np.array([0.0], dtype=np.float64), fpr])
    fnr = np.concatenate([np.array([1.0], dtype=np.float64), fnr])
    thresholds = np.concatenate([np.array([np.inf], dtype=np.float64), thresholds])
    return fpr, fnr, thresholds


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if len(labels) == 0 or len(scores) == 0:
        return float("nan"), float("nan")
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    fpr, fnr, thresholds = _sorted_det_points(scores, labels)
    if fpr.size == 0:
        return float("nan"), float("nan")

    diff = fpr - fnr
    zeros = np.flatnonzero(np.isclose(diff, 0.0, atol=1e-12))
    if zeros.size:
        idx = int(zeros[0])
        return float(fpr[idx] * 100.0), float(thresholds[idx])

    crossings = np.flatnonzero(diff[:-1] * diff[1:] < 0.0)
    if crossings.size == 0:
        idx = int(np.argmin(np.abs(diff)))
        return float(((fpr[idx] + fnr[idx]) * 0.5) * 100.0), float(thresholds[idx])

    idx = int(crossings[0])
    left_diff = float(diff[idx])
    right_diff = float(diff[idx + 1])
    weight = left_diff / (left_diff - right_diff)
    eer = float(fpr[idx] + weight * (fpr[idx + 1] - fpr[idx]))
    threshold = float(thresholds[idx] + weight * (thresholds[idx + 1] - thresholds[idx]))
    return eer * 100.0, threshold


def compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> tuple[float, float]:
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")

    fpr, fnr, thresholds = _sorted_det_points(scores, labels)
    if fpr.size == 0:
        return float("nan"), float("nan")

    dcf = c_miss * p_target * fnr + c_fa * (1.0 - p_target) * fpr
    normalization = min(c_miss * p_target, c_fa * (1.0 - p_target))
    if normalization > 0.0:
        dcf = dcf / normalization
    min_idx = int(np.argmin(dcf))
    return float(dcf[min_idx]), float(thresholds[min_idx])


def compute_cllr(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    target_scores = scores[labels == 1]
    nontarget_scores = scores[labels == 0]
    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        return float("nan")
    cllr_tar = np.mean(np.log2(1.0 + np.exp(-target_scores)))
    cllr_non = np.mean(np.log2(1.0 + np.exp(nontarget_scores)))
    return float(0.5 * (cllr_tar + cllr_non))


def compute_metrics(scores: np.ndarray, labels: np.ndarray, p_target: float = 0.05) -> dict[str, float]:
    eer, _ = compute_eer(scores, labels)
    min_dcf, _ = compute_min_dcf(scores, labels, p_target=p_target)
    cllr = compute_cllr(scores, labels)
    return {"EER(%)": eer, "minDCF": min_dcf, "Cllr": cllr}


def load_trials(trial_path: str | Path) -> list[TrialRecord]:
    records: list[TrialRecord] = []
    with open(trial_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            condition = parts[3] if len(parts) > 3 else DEFAULT_CONDITION
            records.append(TrialRecord(parts[0], parts[1], parts[2], condition))
    return records


def iter_trials(trial_path: str | Path) -> Iterable[TrialRecord]:
    with open(trial_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            condition = parts[3] if len(parts) > 3 else DEFAULT_CONDITION
            yield TrialRecord(parts[0], parts[1], parts[2], condition)


def trials_to_dataframe(records: Iterable[TrialRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        rows.append(
            {
                "utt1": rec.utt1,
                "utt2": rec.utt2,
                "label": rec.label,
                "label_int": 1 if rec.label == "target" else 0,
                "condition": rec.condition,
                "pair_key": PAIR_DELIMITER.join(canonical_pair_key(rec.utt1, rec.utt2)),
            }
        )
    return pd.DataFrame(rows)


def score_trials_from_embeddings(
    embeddings: dict[str, np.ndarray],
    trials: Iterable[TrialRecord],
) -> tuple[pd.DataFrame, int]:
    rows = []
    missing = 0
    for rec in trials:
        if rec.utt1 not in embeddings or rec.utt2 not in embeddings:
            missing += 1
            continue
        score = cosine_similarity(embeddings[rec.utt1], embeddings[rec.utt2])
        rows.append(
            {
                "utt1": rec.utt1,
                "utt2": rec.utt2,
                "pair_key": PAIR_DELIMITER.join(canonical_pair_key(rec.utt1, rec.utt2)),
                "score": score,
                "label": rec.label,
                "label_int": 1 if rec.label == "target" else 0,
                "condition": rec.condition,
            }
        )
    return pd.DataFrame(rows), missing


def metrics_report_from_arrays(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    conditions: np.ndarray | None = None,
    condition_names: Iterable[str] | None = None,
    p_target: float = 0.05,
) -> list[dict[str, float | str | int]]:
    metrics_rows: list[dict[str, float | str | int]] = []
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.size == 0 or labels.size == 0:
        return metrics_rows

    overall = compute_metrics(scores, labels, p_target=p_target)
    metrics_rows.append(
        {
            "Condition": "Overall",
            "NumTrials": int(labels.size),
            "EER(%)": overall["EER(%)"],
            "minDCF": overall["minDCF"],
            "Cllr": overall["Cllr"],
        }
    )

    if conditions is None or condition_names is None:
        return metrics_rows

    condition_name_list = list(condition_names)
    present_condition_ids = [cid for cid in range(len(condition_name_list)) if int(np.count_nonzero(conditions == cid)) > 0]
    if not present_condition_ids:
        return metrics_rows
    if len(present_condition_ids) == 1 and condition_name_list[present_condition_ids[0]] == DEFAULT_CONDITION:
        return metrics_rows

    all_target_scores = scores[labels == 1]
    all_target_labels = labels[labels == 1]
    for condition_id in present_condition_ids:
        condition_name = condition_name_list[condition_id]
        mask = conditions == condition_id
        sub_scores = scores[mask]
        sub_labels = labels[mask]
        borrowed = False
        if (
            sub_scores.size > 0
            and len(np.unique(sub_labels)) < 2
            and all_target_scores.size > 0
            and np.any(sub_labels == 0)
        ):
            sub_scores = np.concatenate([all_target_scores, sub_scores])
            sub_labels = np.concatenate([all_target_labels, sub_labels])
            borrowed = True
        if len(np.unique(sub_labels)) < 2:
            continue
        metrics = compute_metrics(sub_scores, sub_labels, p_target=p_target)
        label = condition_name + (" [borrowed targets]" if borrowed else "")
        metrics_rows.append(
            {
                "Condition": label,
                "NumTrials": int(sub_labels.size),
                "EER(%)": metrics["EER(%)"],
                "minDCF": metrics["minDCF"],
                "Cllr": metrics["Cllr"],
            }
        )
    return metrics_rows


def metrics_report_from_scored_df(df: pd.DataFrame, p_target: float = 0.05) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["Condition", "NumTrials", "EER(%)", "minDCF", "Cllr"])

    condition_names = list(dict.fromkeys(df["condition"].astype(str).tolist()))
    condition_to_id = {name: idx for idx, name in enumerate(condition_names)}
    conditions = df["condition"].map(condition_to_id).to_numpy(dtype=np.uint16, copy=False)
    rows = metrics_report_from_arrays(
        df["score"].to_numpy(),
        df["label_int"].to_numpy(),
        conditions=conditions,
        condition_names=condition_names,
        p_target=p_target,
    )
    return pd.DataFrame(rows, columns=["Condition", "NumTrials", "EER(%)", "minDCF", "Cllr"])


def _prepare_index_output_dir(output_dir: Path, *, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Index output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def build_scenario_eval_index_from_trial_path(
    *,
    scenario_id: str,
    trial_path: str | Path,
    output_dir: str | Path,
    force: bool = False,
) -> dict[str, object]:
    trial_path = Path(trial_path).resolve()
    output_dir = Path(output_dir).resolve()
    _prepare_index_output_dir(output_dir, force=force)

    utt_ids: set[str] = set()
    condition_to_id: dict[str, int] = {}
    target_count = 0
    nontarget_count = 0
    trial_count = 0

    for record in iter_trials(trial_path):
        trial_count += 1
        utt_ids.add(record.utt1)
        utt_ids.add(record.utt2)
        if record.condition not in condition_to_id:
            condition_to_id[record.condition] = len(condition_to_id)
        if record.label == "target":
            target_count += 1
        else:
            nontarget_count += 1

    utt_vocab = sorted(utt_ids)
    utt_to_id = {utt: idx for idx, utt in enumerate(utt_vocab)}
    condition_names = [None] * len(condition_to_id)
    for name, idx in condition_to_id.items():
        condition_names[idx] = name

    vocab_path = output_dir / "utt_vocab.txt"
    vocab_path.write_text("\n".join(utt_vocab) + ("\n" if utt_vocab else ""), encoding="utf-8")

    pairs_path = output_dir / "pairs.uint64.bin"
    labels_path = output_dir / "labels.uint8.bin"
    conditions_path = output_dir / "conditions.uint16.bin"

    pairs = np.memmap(pairs_path, dtype=np.uint64, mode="w+", shape=(trial_count,))
    labels = np.memmap(labels_path, dtype=np.uint8, mode="w+", shape=(trial_count,))
    conditions = np.memmap(conditions_path, dtype=np.uint16, mode="w+", shape=(trial_count,))

    cursor = 0
    for record in iter_trials(trial_path):
        pairs[cursor] = pair_code_from_utt_ids(utt_to_id[record.utt1], utt_to_id[record.utt2])
        labels[cursor] = 1 if record.label == "target" else 0
        conditions[cursor] = condition_to_id[record.condition]
        cursor += 1

    if trial_count:
        order = np.argsort(np.asarray(pairs), kind="mergesort")
        pairs[:] = pairs[order]
        labels[:] = labels[order]
        conditions[:] = conditions[order]

    pairs.flush()
    labels.flush()
    conditions.flush()

    unique_pair_count = 0
    if trial_count:
        unique_pair_count = 1
        for idx in range(1, trial_count):
            if int(pairs[idx]) != int(pairs[idx - 1]):
                unique_pair_count += 1

    meta = {
        "scenario_id": scenario_id,
        "trial_count": trial_count,
        "unique_pair_count": unique_pair_count,
        "target_count": target_count,
        "nontarget_count": nontarget_count,
        "condition_to_id": condition_to_id,
        "condition_names": condition_names,
        "aggregation_version": REFERENCE_INDEX_VERSION,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return meta


def load_scenario_eval_index(index_dir: str | Path) -> ScenarioEvalIndex:
    index_dir = Path(index_dir).resolve()
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    vocab = (index_dir / "utt_vocab.txt").read_text(encoding="utf-8").splitlines()
    utt_to_id = {utt: idx for idx, utt in enumerate(vocab)}
    trial_count = int(meta["trial_count"])
    pairs = np.memmap(index_dir / "pairs.uint64.bin", dtype=np.uint64, mode="r", shape=(trial_count,))
    labels = np.memmap(index_dir / "labels.uint8.bin", dtype=np.uint8, mode="r", shape=(trial_count,))
    conditions = np.memmap(index_dir / "conditions.uint16.bin", dtype=np.uint16, mode="r", shape=(trial_count,))
    return ScenarioEvalIndex(
        scenario_id=str(meta["scenario_id"]),
        root=index_dir,
        trial_count=trial_count,
        unique_pair_count=int(meta.get("unique_pair_count", trial_count)),
        target_count=int(meta["target_count"]),
        nontarget_count=int(meta["nontarget_count"]),
        condition_to_id={str(key): int(value) for key, value in meta["condition_to_id"].items()},
        condition_names=tuple(str(name) for name in meta["condition_names"]),
        utt_to_id=utt_to_id,
        pairs=pairs,
        labels=labels,
        conditions=conditions,
        aggregation_version=str(meta.get("aggregation_version", "")),
    )


def _read_score_header(score_path: str | Path) -> tuple[list[str], dict[str, int], int]:
    with open(score_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{score_path} is empty.") from exc
        columns = [column.strip() for column in header]
        required = {"utt1", "utt2", "score"}
        missing = required - set(columns)
        if missing:
            raise ValueError(f"{score_path} is missing required columns: {sorted(missing)}")
        column_index = {name: idx for idx, name in enumerate(columns)}
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            if len(row) < len(columns):
                raise ValueError(f"{score_path} has a malformed row at line {row_count + 1}.")
        return columns, column_index, row_count


def _stream_score_pairs_to_memmap(
    score_path: str | Path,
    *,
    utt_to_id: dict[str, int],
    work_dir: str | Path,
    file_stem: str,
) -> tuple[np.memmap, np.memmap]:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    columns, column_index, row_count = _read_score_header(score_path)
    if row_count == 0:
        raise ValueError(f"{score_path} does not contain any scores.")
    pairs = np.memmap(work_dir / f"{file_stem}.pairs.uint64.mmap", dtype=np.uint64, mode="w+", shape=(row_count,))
    scores = np.memmap(work_dir / f"{file_stem}.scores.float32.mmap", dtype=np.float32, mode="w+", shape=(row_count,))

    with open(score_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        parsed_columns = [column.strip() for column in header]
        if parsed_columns != columns:
            raise ValueError(f"{score_path} changed between validation passes.")
        for idx, row in enumerate(reader):
            if len(row) < len(columns):
                raise ValueError(f"{score_path} has a malformed row at line {idx + 2}.")
            utt1 = row[column_index["utt1"]].strip()
            utt2 = row[column_index["utt2"]].strip()
            score_text = row[column_index["score"]].strip()
            if utt1 not in utt_to_id:
                raise ValueError(f"{score_path} contains unknown utterance id at line {idx + 2}: {utt1}")
            if utt2 not in utt_to_id:
                raise ValueError(f"{score_path} contains unknown utterance id at line {idx + 2}: {utt2}")
            try:
                score_value = float(score_text)
            except ValueError as exc:
                raise ValueError(f"{score_path} has an invalid score at line {idx + 2}: {score_text}") from exc
            pairs[idx] = pair_code_from_utt_ids(utt_to_id[utt1], utt_to_id[utt2])
            scores[idx] = np.float32(score_value)

    pairs.flush()
    scores.flush()
    return pairs, scores


def _dedupe_sorted_pairs(
    sorted_pairs: np.ndarray,
    sorted_scores: np.ndarray,
    *,
    score_path: str | Path,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    if sorted_pairs.size == 0:
        return sorted_pairs, sorted_scores

    write_idx = 1
    duplicate_rows = 0
    for idx in range(1, sorted_pairs.size):
        if int(sorted_pairs[idx]) != int(sorted_pairs[write_idx - 1]):
            sorted_pairs[write_idx] = sorted_pairs[idx]
            sorted_scores[write_idx] = sorted_scores[idx]
            write_idx += 1
            continue
        duplicate_rows += 1
        if abs(float(sorted_scores[idx]) - float(sorted_scores[write_idx - 1])) > tolerance:
            raise ValueError(
                "Submission contains duplicate pair rows with inconsistent scores "
                f"in {score_path}."
            )
    if duplicate_rows:
        print(f"[Scoring] Collapsed {duplicate_rows} duplicate score row(s) by canonical pair.")
    return sorted_pairs[:write_idx], sorted_scores[:write_idx]


def _align_scores_to_reference_pairs(
    submitted_pairs: np.ndarray,
    submitted_scores: np.ndarray,
    reference_pairs: np.ndarray,
    *,
    score_path: str | Path,
    work_dir: str | Path,
    file_stem: str,
) -> np.ndarray:
    if submitted_pairs.shape == reference_pairs.shape and np.array_equal(submitted_pairs, reference_pairs):
        return submitted_scores

    aligned = np.memmap(
        Path(work_dir) / f"{file_stem}.aligned_scores.float32.mmap",
        dtype=np.float32,
        mode="w+",
        shape=(reference_pairs.size,),
    )

    missing = 0
    extra = 0
    i = 0
    j = 0
    while i < submitted_pairs.size and j < reference_pairs.size:
        submitted = int(submitted_pairs[i])
        expected = int(reference_pairs[j])
        if submitted == expected:
            score = np.float32(submitted_scores[i])
            while j < reference_pairs.size and int(reference_pairs[j]) == submitted:
                aligned[j] = score
                j += 1
            i += 1
        elif submitted < expected:
            extra += 1
            i += 1
        else:
            missing_pair = expected
            while j < reference_pairs.size and int(reference_pairs[j]) == missing_pair:
                missing += 1
                j += 1

    extra += int(submitted_pairs.size - i)
    while j < reference_pairs.size:
        missing_pair = int(reference_pairs[j])
        while j < reference_pairs.size and int(reference_pairs[j]) == missing_pair:
            missing += 1
            j += 1

    messages = []
    if missing:
        messages.append(f"missing {missing} required trial score(s)")
    if extra:
        messages.append(f"contains {extra} unexpected trial pair(s)")
    if messages:
        raise ValueError(f"{score_path} " + " and ".join(messages) + ".")
    aligned.flush()
    return aligned


def evaluate_score_file_with_index(
    score_path: str | Path,
    index: ScenarioEvalIndex,
    *,
    work_dir: str | Path,
    p_target: float = 0.05,
) -> IndexEvaluationResult:
    pair_memmap, score_memmap = _stream_score_pairs_to_memmap(
        score_path,
        utt_to_id=index.utt_to_id,
        work_dir=work_dir,
        file_stem=index.scenario_id,
    )
    order = np.argsort(np.asarray(pair_memmap), kind="mergesort")
    sorted_pairs = np.asarray(pair_memmap)[order]
    sorted_scores = np.asarray(score_memmap)[order]
    sorted_pairs, sorted_scores = _dedupe_sorted_pairs(sorted_pairs, sorted_scores, score_path=score_path)
    aligned_scores = _align_scores_to_reference_pairs(
        sorted_pairs,
        sorted_scores,
        np.asarray(index.pairs),
        score_path=score_path,
        work_dir=work_dir,
        file_stem=index.scenario_id,
    )

    metrics_rows = metrics_report_from_arrays(
        aligned_scores,
        np.asarray(index.labels),
        conditions=np.asarray(index.conditions),
        condition_names=index.condition_names,
        p_target=p_target,
    )
    if not metrics_rows:
        raise ValueError(f"No valid metrics could be computed for {index.scenario_id}")
    return IndexEvaluationResult(
        metrics_rows=metrics_rows,
        overall_eer=float(metrics_rows[0]["EER(%)"]),
        overall_min_dcf=float(metrics_rows[0]["minDCF"]),
    )


def load_score_file(score_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(score_path)
    required = {"utt1", "utt2", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{score_path} is missing required columns: {sorted(missing)}")
    df = df.copy()
    utt1 = df["utt1"].astype(str)
    utt2 = df["utt2"].astype(str)
    left = utt1.where(utt1 <= utt2, utt2)
    right = utt2.where(utt1 <= utt2, utt1)
    df["pair_key"] = left + PAIR_DELIMITER + right
    return df


def validate_and_merge_submission(expected_trials: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    if score_df["pair_key"].duplicated().any():
        dup_count = int(score_df["pair_key"].duplicated().sum())
        conflict_count = 0
        for _, group in score_df.groupby("pair_key", sort=False):
            if len(group) <= 1:
                continue
            scores = group["score"].astype(float).to_numpy()
            if float(scores.max() - scores.min()) > 1e-6:
                conflict_count += 1
                if conflict_count >= 5:
                    break
        if conflict_count:
            raise ValueError(
                "Submission contains duplicate pair rows with inconsistent scores "
                f"for at least {conflict_count} pair(s)."
            )
        score_df = score_df.drop_duplicates(subset=["pair_key"], keep="first").copy()
        print(f"[Scoring] Collapsed {dup_count} duplicate score row(s) by canonical pair.")

    expected = expected_trials[["pair_key", "utt1", "utt2", "label", "label_int", "condition"]].copy()
    scored = score_df[["pair_key", "score"]].copy()
    merged = expected.merge(scored, on="pair_key", how="left", validate="many_to_one")

    missing_scores = int(merged["score"].isna().sum())
    if missing_scores > 0:
        raise ValueError(f"Submission is missing {missing_scores} required trial scores.")

    extra = set(score_df["pair_key"]) - set(expected["pair_key"])
    if extra:
        raise ValueError(f"Submission contains {len(extra)} unexpected trial pairs.")
    return merged


def write_score_file(scored_df: pd.DataFrame, output_path: str | Path) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dedup_df = (
        scored_df[["pair_key", "utt1", "utt2", "score"]]
        .sort_values(["pair_key", "utt1", "utt2"])
        .drop_duplicates(subset=["pair_key"], keep="first")
    )
    dedup_df[["utt1", "utt2", "score"]].to_csv(out_path, index=False)
