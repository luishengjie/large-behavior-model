"""Convert model predictions into Twin-2K-500 response tables for evaluation."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .digital_twin_metrics import (
    AccuracyResult,
    calculate_task_weighted_normalized_accuracy,
)


def _answer_values(response: object) -> list[object]:
    if not isinstance(response, Mapping):
        return []
    if "SelectedByPosition" in response:
        values = response["SelectedByPosition"]
    elif "Values" in response:
        values = response["Values"]
    elif "Text" in response:
        values = response["Text"]
    else:
        return []
    return values if isinstance(values, list) else [values]


def _configured_columns(
    question_id: str,
    *,
    response_ranges: Mapping[str, float],
    task_mapping: Mapping[str, str],
) -> list[str]:
    question_id = question_id.upper()
    return [
        column
        for column in task_mapping
        if column in response_ranges
        and (column == question_id or column.startswith(f"{question_id}_"))
    ]


def build_model_response_tables(
    records: Sequence[Mapping[str, Any]],
    *,
    response_ranges: Mapping[str, float],
    task_mapping: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ground-truth and model-prediction tables from evaluation records."""

    references: dict[str, dict[str, object]] = {}
    comparisons: dict[str, dict[str, object]] = {}

    for record in records:
        pid = str(record.get("pid", ""))
        columns = _configured_columns(
            str(record.get("question_id", "")),
            response_ranges=response_ranges,
            task_mapping=task_mapping,
        )
        if not columns:
            continue

        target_values = _answer_values(record.get("target"))
        if len(target_values) != len(columns):
            raise ValueError(
                f"Ground-truth response count does not match configured columns "
                f"for pid={pid!r}, question_id={record.get('question_id')!r}: "
                f"answers={len(target_values)}, columns={len(columns)}."
            )

        prediction_values = (
            _answer_values(record.get("prediction"))
            if bool(record.get("valid_schema"))
            else []
        )
        prediction_complete = len(prediction_values) == len(columns)
        reference_row = references.setdefault(pid, {})
        comparison_row = comparisons.setdefault(pid, {})

        for index, (column, target_value) in enumerate(
            zip(columns, target_values, strict=True)
        ):
            if column in reference_row:
                raise ValueError(
                    f"Duplicate response column for pid={pid!r}: {column!r}."
                )
            reference_row[column] = target_value
            comparison_row[column] = (
                prediction_values[index] if prediction_complete else np.nan
            )

    if not references:
        raise ValueError("No scoreable model-evaluation records were found.")

    reference = pd.DataFrame.from_dict(references, orient="index")
    comparison = pd.DataFrame.from_dict(comparisons, orient="index")
    reference.index.name = "pid"
    comparison.index.name = "pid"
    return reference.sort_index(axis=1), comparison.sort_index(axis=1)


def calculate_model_accuracy(
    records: Sequence[Mapping[str, Any]],
    *,
    response_ranges: Mapping[str, float],
    task_mapping: Mapping[str, str],
    anchoring_groups: Sequence[Sequence[str]] = (),
    missing_comparison: Literal["exclude", "zero"] = "exclude",
) -> AccuracyResult:
    """Score model predictions against held-out human responses.

    Use missing_comparison="exclude" to match tianyipeng-lab/Digital-Twin-Simulation evaluation.
    Use missing_comparison="zero" to penalize invalid or missing model predictions.
    """

    reference, comparison = build_model_response_tables(
        records,
        response_ranges=response_ranges,
        task_mapping=task_mapping,
    )
    return calculate_task_weighted_normalized_accuracy(
        reference=reference,
        comparison=comparison,
        response_ranges=response_ranges,
        task_mapping=task_mapping,
        anchoring_groups=anchoring_groups,
        missing_comparison=missing_comparison,
    )
