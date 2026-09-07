#!/usr/bin/env python3
"""Shared aggregate metric helpers for OpenSVBench.

The public leaderboard calls the high-level groups "scenarios". In the catalog
they are represented by CapabilitySpec objects, while ScenarioSpec objects are
the concrete ranked trial sets.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from tools.benchmark_catalog import capability_by_id, dataset_by_id, iter_capabilities, scenario_by_id


def _as_metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _metric_by_trial_set(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    trial_set_id_key: str,
) -> dict[str, float]:
    metric_by_trial_set: dict[str, float] = {}
    for row in rows:
        trial_set_id = row.get(trial_set_id_key)
        if trial_set_id is None:
            continue
        metric = _as_metric(row.get(metric_key))
        if metric is None:
            continue
        metric_by_trial_set[str(trial_set_id)] = metric
    return metric_by_trial_set


def capability_average(
    rows: Sequence[Mapping[str, Any]],
    *,
    capability_id: str,
    metric_key: str,
    trial_set_id_key: str = "scenario_id",
    collection: str = "core",
) -> dict[str, Any] | None:
    """Compute the source-balanced average for one scenario group."""

    metric_by_trial_set = _metric_by_trial_set(
        rows,
        metric_key=metric_key,
        trial_set_id_key=trial_set_id_key,
    )
    capability = capability_by_id(capability_id)

    expected_trial_sets_by_source: dict[str, list[str]] = defaultdict(list)
    available_values_by_source: dict[str, list[float]] = defaultdict(list)
    available_trial_sets_by_source: dict[str, list[str]] = defaultdict(list)

    for trial_set_id in capability.scenario_ids:
        scenario = scenario_by_id(trial_set_id)
        expected_trial_sets_by_source[scenario.dataset_id].append(trial_set_id)
        metric = metric_by_trial_set.get(trial_set_id)
        if metric is None:
            continue
        available_values_by_source[scenario.dataset_id].append(metric)
        available_trial_sets_by_source[scenario.dataset_id].append(trial_set_id)

    source_components: list[dict[str, Any]] = []
    source_means: list[float] = []
    for dataset_id, expected_trial_set_ids in expected_trial_sets_by_source.items():
        mean_value = _mean(available_values_by_source.get(dataset_id, []))
        if mean_value is None:
            continue
        source_means.append(mean_value)
        source_components.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": dataset_by_id(dataset_id).display_name,
                "value": mean_value,
                "trial_set_count": len(available_trial_sets_by_source[dataset_id]),
                "expected_trial_set_count": len(expected_trial_set_ids),
                "trial_set_ids": tuple(available_trial_sets_by_source[dataset_id]),
                "expected_trial_set_ids": tuple(expected_trial_set_ids),
            }
        )

    mean_value = _mean(source_means)
    if mean_value is None:
        return None

    return {
        "capability_id": capability.capability_id,
        "display_name": capability.display_name,
        "value": mean_value,
        "trial_set_count": sum(item["trial_set_count"] for item in source_components),
        "expected_trial_set_count": len(capability.scenario_ids),
        "source_count": len(source_components),
        "expected_source_count": len(expected_trial_sets_by_source),
        "source_components": source_components,
        "aggregation": "source_balanced_by_dataset_id",
    }


def scenario_macro_average(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    trial_set_id_key: str = "scenario_id",
    collection: str = "core",
) -> tuple[float | None, list[dict[str, Any]]]:
    """Average a metric by high-level scenario, then across scenarios.

    Each high-level scenario first averages its available trial-set metrics
    within each source dataset, then averages those source means. The returned
    aggregate is the unweighted mean of those high-level scenario means.
    """

    components: list[dict[str, Any]] = []
    for capability in iter_capabilities(collection=collection):
        component = capability_average(
            rows,
            capability_id=capability.capability_id,
            metric_key=metric_key,
            trial_set_id_key=trial_set_id_key,
            collection=collection,
        )
        if component is None:
            continue
        components.append(component)

    aggregate = _mean([component["value"] for component in components])
    return aggregate, components


def scenario_macro_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    eer_key: str,
    min_dcf_key: str,
    trial_set_id_key: str = "scenario_id",
    collection: str = "core",
) -> dict[str, Any]:
    aggregate_eer, eer_components = scenario_macro_average(
        rows,
        metric_key=eer_key,
        trial_set_id_key=trial_set_id_key,
        collection=collection,
    )
    aggregate_min_dcf, min_dcf_components = scenario_macro_average(
        rows,
        metric_key=min_dcf_key,
        trial_set_id_key=trial_set_id_key,
        collection=collection,
    )
    return {
        "aggregate_eer": aggregate_eer,
        "aggregate_min_dcf": aggregate_min_dcf,
        "eer_components": eer_components,
        "min_dcf_components": min_dcf_components,
    }
