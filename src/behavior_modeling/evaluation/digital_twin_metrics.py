"""Digital-Twin-Simulation accuracy metrics adapted for Twin-2K-500.

Source: https://github.com/tianyipeng-lab/Digital-Twin-Simulation

This module contains the scoring logic adapted from ``evaluation/mad_accuracy_evaluation.py``.

Licensed under the Apache License, Version 2.0. This implementation has been modified to
support raw Hugging Face dataset schema and reusable dataframe-based evaluation.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import sem, t


@dataclass(frozen=True)
class AccuracyResult:
    """Item-level and aggregate normalized-accuracy results."""

    task_weighted_normalized_accuracy: float
    normalized_accuracy: float
    exact_match_accuracy: float
    item_scores: pd.DataFrame
    participant_task_scores: pd.DataFrame
    task_summary: pd.DataFrame
    participant_summary: pd.DataFrame

    @property
    def overall_accuracy(self) -> float:
        """Backward-compatible name for task-weighted normalized accuracy."""

        return self.task_weighted_normalized_accuracy

    @property
    def n_participants(self) -> int:
        return len(self.participant_summary)

    @property
    def n_tasks(self) -> int:
        return len(self.task_summary)

    @property
    def n_responses(self) -> int:
        return len(self.item_scores)


def assign_decile(value: float, thresholds: Sequence[float]) -> float:
    """Map a numeric response to a decile using nine ordered thresholds."""

    if pd.isna(value):
        return np.nan
    if len(thresholds) != 9:
        raise ValueError("Decile conversion requires exactly nine thresholds.")
    return float(np.searchsorted(thresholds, value, side="left") + 1)


def summary_mad(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Return the mean, standard error, and 95% confidence interval.

    Adapted from ``summary_mad`` in the original evaluation module.
    """

    array = np.asarray(values)
    mean_mad = array.mean()
    standard_error = sem(array) if len(array) > 1 else 0
    confidence_low, confidence_high = (
        t.interval(
            0.95,
            len(array) - 1,
            loc=mean_mad,
            scale=standard_error,
        )
        if len(array) > 1
        else (np.nan, np.nan)
    )
    return (
        round(float(mean_mad), 3),
        round(float(standard_error), 3),
        round(float(confidence_low), 3),
        round(float(confidence_high), 3),
    )


def compute_column_mad(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    column_ranges: Mapping[str, float],
    random_baseline: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute the authors' complete-case MAD metrics by response column.

    Adapted from ``compute_column_mad`` in the original evaluation module.
    """

    common_columns = list(
        set(predictions.columns) & set(ground_truth.columns) & set(column_ranges)
    )
    if random_baseline is not None:
        common_columns = list(set(common_columns) & set(random_baseline.columns))

    rows = []
    for column in common_columns:
        valid = predictions[column].notna() & ground_truth[column].notna()
        if random_baseline is not None:
            valid &= random_baseline[column].notna()

        prediction_values = predictions.loc[valid, column]
        truth_values = ground_truth.loc[valid, column]
        if prediction_values.empty:
            continue

        normalized_difference = (
            prediction_values - truth_values
        ).abs() / column_ranges[column]
        mean, _, confidence_low, confidence_high = summary_mad(
            1 - normalized_difference
        )
        row = {
            "Column": column,
            "predictions vs. ground_truth": normalized_difference.mean(),
            "predictions vs. ground_truth Accuracy": mean,
            "predictions vs. ground_truth Accuracy 95% CI Lower": confidence_low,
            "predictions vs. ground_truth Accuracy 95% CI Higher": confidence_high,
            "number of respondents": int(valid.sum()),
        }

        if random_baseline is not None:
            random_values = random_baseline.loc[valid, column]
            random_difference = (random_values - truth_values).abs() / column_ranges[
                column
            ]
            random_mean, _, random_low, random_high = summary_mad(1 - random_difference)
            random_accuracy = 1 - random_difference.mean()
            row.update(
                {
                    "random_baseline vs ground_truth": random_difference.mean(),
                    "random_baseline vs ground_truth Accuracy": random_mean,
                    "random_baseline vs. ground_truth Accuracy 95% CI Lower": (
                        random_low
                    ),
                    "random_baseline vs. ground_truth Accuracy 95% CI Higher": (
                        random_high
                    ),
                    "predictions accuracy / random accuracy": (
                        (1 - normalized_difference.mean()) / random_accuracy
                        if random_accuracy != 0
                        else np.inf
                    ),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def compute_task_mad(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    column_ranges: Mapping[str, float],
    qid_to_task: Mapping[str, str],
    random_baseline: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute the authors' complete-case MAD metrics by behavioral task.

    Adapted from ``compute_task_mad`` in the original evaluation module.
    """

    common_columns = list(
        set(predictions.columns) & set(ground_truth.columns) & set(column_ranges)
    )
    rows = []

    for task in sorted(set(qid_to_task.values())):
        task_columns = [
            column
            for column, mapped_task in qid_to_task.items()
            if mapped_task == task and column in common_columns
        ]
        normalized_differences = []
        random_differences = []
        participant_ids = set()

        for column in task_columns:
            valid = predictions[column].notna() & ground_truth[column].notna()
            if random_baseline is not None:
                valid &= random_baseline[column].notna()
            if not valid.any():
                continue

            prediction_values = predictions.loc[valid, column]
            truth_values = ground_truth.loc[valid, column]
            normalized_differences.append(
                (prediction_values - truth_values).abs() / column_ranges[column]
            )
            participant_ids.update(prediction_values.index)

            if random_baseline is not None:
                random_values = random_baseline.loc[valid, column]
                random_differences.append(
                    (random_values - truth_values).abs() / column_ranges[column]
                )

        if not normalized_differences:
            continue

        task_difference = pd.concat(normalized_differences)
        mean, _, confidence_low, confidence_high = summary_mad(1 - task_difference)
        row = {
            "Task": task,
            "predictions vs. ground_truth": task_difference.mean(),
            "predictions vs. ground_truth Accuracy": mean,
            "predictions vs. ground_truth Accuracy 95% CI Lower": confidence_low,
            "predictions vs. ground_truth Accuracy 95% CI Higher": confidence_high,
            "number of respondents": len(participant_ids),
        }

        if random_baseline is not None and random_differences:
            random_difference = pd.concat(random_differences)
            random_mean, _, random_low, random_high = summary_mad(1 - random_difference)
            random_accuracy = 1 - random_difference.mean()
            row.update(
                {
                    "random_baseline vs ground_truth": random_difference.mean(),
                    "random_baseline vs ground_truth Accuracy": random_mean,
                    "random_baseline vs. ground_truth Accuracy 95% CI Lower": (
                        random_low
                    ),
                    "random_baseline vs. ground_truth Accuracy 95% CI Higher": (
                        random_high
                    ),
                    "predictions accuracy / random accuracy": (
                        (1 - task_difference.mean()) / random_accuracy
                        if random_accuracy != 0
                        else np.inf
                    ),
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def transform_anchoring_deciles(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    column_groups: Sequence[Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert unbounded responses using thresholds from the reference data."""

    reference_out = reference.copy()
    comparison_out = comparison.copy()

    for requested_columns in column_groups:
        columns = [
            column
            for column in requested_columns
            if column in reference_out.columns and column in comparison_out.columns
        ]
        if not columns:
            continue

        reference_values = pd.concat(
            [
                pd.to_numeric(reference_out[column], errors="coerce")
                for column in columns
            ]
        ).dropna()
        if reference_values.empty:
            raise ValueError(
                f"Cannot calculate anchoring deciles for columns {columns}: "
                "all reference responses are missing."
            )

        thresholds = np.percentile(reference_values, np.arange(10, 100, 10))
        for frame in (reference_out, comparison_out):
            for column in columns:
                numeric = pd.to_numeric(frame[column], errors="coerce")
                frame[column] = numeric.map(
                    lambda value: assign_decile(value, thresholds)
                )

    return reference_out, comparison_out


def calculate_task_weighted_normalized_accuracy(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    response_ranges: Mapping[str, float],
    task_mapping: Mapping[str, str],
    anchoring_groups: Sequence[Sequence[str]] = (),
    missing_comparison: Literal["exclude", "zero"] = "exclude",
) -> AccuracyResult:
    """Calculate normalized accuracy with equal task weights per participant.

    ``reference`` contains ground-truth responses and ``comparison`` contains
    either repeated human responses or model predictions. Missing comparison
    values can be excluded for human test-retest analysis or scored as zero for
    model evaluation. Missing reference values are always excluded.
    """

    if missing_comparison not in {"exclude", "zero"}:
        raise ValueError("missing_comparison must be 'exclude' or 'zero'.")
    if not reference.index.is_unique or not comparison.index.is_unique:
        raise ValueError("Participant indexes must be unique.")

    common_participants = reference.index.intersection(comparison.index)
    if common_participants.empty:
        raise ValueError("The response tables have no participants in common.")

    reference_numeric = reference.loc[common_participants].apply(
        pd.to_numeric, errors="coerce"
    )
    comparison_numeric = comparison.loc[common_participants].apply(
        pd.to_numeric, errors="coerce"
    )
    reference_numeric, comparison_numeric = transform_anchoring_deciles(
        reference_numeric,
        comparison_numeric,
        anchoring_groups,
    )

    configured_columns = set(response_ranges) & set(task_mapping)
    score_columns = sorted(
        configured_columns
        & set(reference_numeric.columns)
        & set(comparison_numeric.columns)
    )
    if not score_columns:
        raise ValueError("No configured response columns are shared by both tables.")

    invalid_ranges = {
        column: response_ranges[column]
        for column in score_columns
        if not np.isfinite(response_ranges[column]) or response_ranges[column] <= 0
    }
    if invalid_ranges:
        raise ValueError(f"Response ranges must be positive: {invalid_ranges}")

    records: list[dict[str, object]] = []
    for column in score_columns:
        reference_present = reference_numeric[column].notna()
        comparison_present = comparison_numeric[column].notna()
        included = (
            reference_present & comparison_present
            if missing_comparison == "exclude"
            else reference_present
        )

        for participant_id in reference_numeric.index[included]:
            reference_value = reference_numeric.at[participant_id, column]
            comparison_value = comparison_numeric.at[participant_id, column]
            valid_comparison = bool(pd.notna(comparison_value))
            exact_match = bool(valid_comparison and comparison_value == reference_value)
            accuracy = 0.0
            if valid_comparison:
                accuracy = (
                    1.0
                    - abs(comparison_value - reference_value) / response_ranges[column]
                )

            records.append(
                {
                    "pid": participant_id,
                    "task": task_mapping[column],
                    "response_column": column,
                    "reference_answer": reference_value,
                    "comparison_answer": comparison_value,
                    "valid_comparison": valid_comparison,
                    "exact_match": exact_match,
                    "accuracy": accuracy,
                }
            )

    item_scores = pd.DataFrame.from_records(records)
    if item_scores.empty:
        raise ValueError("All configured response pairs are missing.")

    participant_task_scores = (
        item_scores.groupby(["pid", "task"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            exact_match_accuracy=("exact_match", "mean"),
            valid_comparison_rate=("valid_comparison", "mean"),
            responses=("response_column", "size"),
        )
        .sort_values(["pid", "task"], ignore_index=True)
    )
    task_summary = (
        participant_task_scores.groupby("task", as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            exact_match_accuracy=("exact_match_accuracy", "mean"),
            valid_comparison_rate=("valid_comparison_rate", "mean"),
            participants=("pid", "nunique"),
            responses=("responses", "sum"),
        )
        .sort_values("task", ignore_index=True)
    )
    participant_summary = (
        participant_task_scores.groupby("pid", as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            exact_match_accuracy=("exact_match_accuracy", "mean"),
            valid_comparison_rate=("valid_comparison_rate", "mean"),
            tasks=("task", "nunique"),
        )
        .sort_values("pid", ignore_index=True)
    )

    return AccuracyResult(
        task_weighted_normalized_accuracy=float(participant_summary["accuracy"].mean()),
        normalized_accuracy=float(item_scores["accuracy"].mean()),
        exact_match_accuracy=float(item_scores["exact_match"].mean()),
        item_scores=item_scores,
        participant_task_scores=participant_task_scores,
        task_summary=task_summary,
        participant_summary=participant_summary,
    )
