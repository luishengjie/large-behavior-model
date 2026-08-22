"""Human test-retest evaluation for Twin-2K-500."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from .digital_twin_metrics import (
    AccuracyResult,
    calculate_task_weighted_normalized_accuracy,
)

# Backward-compatible result name used by existing notebooks.
TestRetestResult = AccuracyResult
TestRetestResult.__test__ = False


def calculate_test_retest(
    earlier: pd.DataFrame,
    later: pd.DataFrame,
    *,
    response_ranges: Mapping[str, float],
    task_mapping: Mapping[str, str],
    anchoring_groups: Sequence[Sequence[str]] = (),
) -> AccuracyResult:
    """Compare later human responses against the earlier human responses."""

    return calculate_task_weighted_normalized_accuracy(
        reference=earlier,
        comparison=later,
        response_ranges=response_ranges,
        task_mapping=task_mapping,
        anchoring_groups=anchoring_groups,
        missing_comparison="exclude",
    )
