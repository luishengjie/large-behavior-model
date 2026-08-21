"""Paper-style human test-retest accuracy.

This module implements the normalized-absolute-deviation metric described in
Toubia et al. (2025), *Twin-2K-500*. It expects the nested answer blocks to
have already been converted to one numeric column per atomic survey response.

The response ranges and response-to-task mapping are deliberately supplied by
the caller. In Twin-2K-500 these are hand-authored parts of the evaluation
protocol; inferring them from observed data would change the metric.

Reference:
https://arxiv.org/html/2505.17479#S3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TestRetestResult:
    """Tables and headline values produced by a test-retest evaluation."""

    overall_accuracy: float
    exact_match_accuracy: float
    item_scores: pd.DataFrame
    participant_task_scores: pd.DataFrame
    task_summary: pd.DataFrame
    participant_summary: pd.DataFrame

    @property
    def n_participants(self) -> int:
        """Number of participants with at least one scored task."""

        return len(self.participant_summary)

    @property
    def n_tasks(self) -> int:
        """Number of tasks with at least one valid comparison."""

        return len(self.task_summary)

    @property
    def n_responses(self) -> int:
        """Number of matched atomic responses used in the calculation."""

        return len(self.item_scores)


# Prevent pytest from treating this result dataclass as a test container.
TestRetestResult.__test__ = False


def assign_decile(value: float, thresholds: Sequence[float]) -> float:
    """Map a numeric response to a decile using nine ordered thresholds."""

    if pd.isna(value):
        return np.nan
    if len(thresholds) != 9:
        raise ValueError("Decile conversion requires exactly nine thresholds.")

    return float(np.searchsorted(thresholds, value, side="left") + 1)


def transform_anchoring_deciles(
    earlier: pd.DataFrame,
    later: pd.DataFrame,
    column_groups: Sequence[Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform unbounded anchoring responses using earlier-wave deciles.

    Each group shares thresholds calculated only from the earlier responses.
    The same thresholds are then applied to both measurement occasions.
    Returned dataframes are copies; inputs are not mutated.
    """

    earlier_out = earlier.copy()
    later_out = later.copy()

    for requested_columns in column_groups:
        columns = [
            column
            for column in requested_columns
            if column in earlier_out.columns and column in later_out.columns
        ]
        if not columns:
            continue

        reference = pd.concat(
            [pd.to_numeric(earlier_out[column], errors="coerce") for column in columns]
        ).dropna()
        if reference.empty:
            raise ValueError(
                f"Cannot calculate anchoring deciles for columns {columns}: "
                "all earlier responses are missing."
            )

        thresholds = np.percentile(reference, np.arange(10, 100, 10))
        for frame in (earlier_out, later_out):
            for column in columns:
                numeric = pd.to_numeric(frame[column], errors="coerce")
                frame[column] = numeric.map(
                    lambda value: assign_decile(value, thresholds)
                )

    return earlier_out, later_out


def calculate_test_retest(
    earlier: pd.DataFrame,
    later: pd.DataFrame,
    *,
    response_ranges: Mapping[str, float],
    task_mapping: Mapping[str, str],
    anchoring_groups: Sequence[Sequence[str]] = (),
) -> TestRetestResult:
    """Calculate paper-style human test-retest accuracy.

    Parameters
    ----------
    earlier, later:
        Wide numeric dataframes with participant IDs in the index and one
        column per atomic response. ``earlier`` contains waves 1-3 holdout
        answers and ``later`` contains the matched wave-4 answers.
    response_ranges:
        Mapping from response column to ``maximum - minimum``. These must be
        theoretical scale ranges, not ranges inferred from observed values.
    task_mapping:
        Mapping from response column to one of the behavioral task names.
    anchoring_groups:
        Groups of unbounded anchoring columns that share decile thresholds.
        After transformation their configured response range must be 9
        because the resulting scale is 1-10.

    Returns
    -------
    TestRetestResult
        Item-, participant-task-, task-, and participant-level results. The
        headline value is the mean of each participant's mean task accuracy,
        so tasks receive equal weight within a participant.
    """

    if not earlier.index.is_unique or not later.index.is_unique:
        raise ValueError("Participant indexes must be unique.")

    common_participants = earlier.index.intersection(later.index)
    if common_participants.empty:
        raise ValueError("The two dataframes have no participants in common.")

    earlier_numeric = earlier.loc[common_participants].apply(
        pd.to_numeric, errors="coerce"
    )
    later_numeric = later.loc[common_participants].apply(pd.to_numeric, errors="coerce")
    earlier_numeric, later_numeric = transform_anchoring_deciles(
        earlier_numeric,
        later_numeric,
        anchoring_groups,
    )

    configured_columns = set(response_ranges) & set(task_mapping)
    score_columns = sorted(
        configured_columns & set(earlier_numeric.columns) & set(later_numeric.columns)
    )
    if not score_columns:
        raise ValueError(
            "No scoreable columns are shared by the dataframes, response "
            "ranges, and task mapping."
        )

    invalid_ranges = {
        column: response_ranges[column]
        for column in score_columns
        if not np.isfinite(response_ranges[column]) or response_ranges[column] <= 0
    }
    if invalid_ranges:
        raise ValueError(f"Response ranges must be positive: {invalid_ranges}")

    records: list[dict[str, object]] = []
    for column in score_columns:
        valid = earlier_numeric[column].notna() & later_numeric[column].notna()
        if not valid.any():
            continue

        earlier_values = earlier_numeric.loc[valid, column]
        later_values = later_numeric.loc[valid, column]
        accuracy = 1.0 - (
            (later_values - earlier_values).abs() / response_ranges[column]
        )

        for participant_id in earlier_values.index:
            records.append(
                {
                    "pid": participant_id,
                    "task": task_mapping[column],
                    "response_column": column,
                    "earlier_answer": earlier_values.at[participant_id],
                    "later_answer": later_values.at[participant_id],
                    "exact_match": bool(
                        earlier_values.at[participant_id]
                        == later_values.at[participant_id]
                    ),
                    "accuracy": accuracy.at[participant_id],
                }
            )

    item_scores = pd.DataFrame.from_records(records)
    if item_scores.empty:
        raise ValueError("All scoreable response pairs are missing.")

    participant_task_scores = (
        item_scores.groupby(["pid", "task"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            exact_match_accuracy=("exact_match", "mean"),
            responses=("response_column", "size"),
        )
        .sort_values(["pid", "task"], ignore_index=True)
    )

    task_summary = (
        participant_task_scores.groupby("task", as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            exact_match_accuracy=("exact_match_accuracy", "mean"),
            participants=("pid", "nunique"),
        )
        .sort_values("task", ignore_index=True)
    )

    participant_summary = (
        participant_task_scores.groupby("pid", as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            exact_match_accuracy=("exact_match_accuracy", "mean"),
            tasks=("task", "nunique"),
        )
        .sort_values("pid", ignore_index=True)
    )

    return TestRetestResult(
        overall_accuracy=float(participant_summary["accuracy"].mean()),
        exact_match_accuracy=float(item_scores["exact_match"].mean()),
        item_scores=item_scores,
        participant_task_scores=participant_task_scores,
        task_summary=task_summary,
        participant_summary=participant_summary,
    )
