"""Participant-disjoint splitting for Twin-2K-500 datasets."""

from __future__ import annotations

from typing import Any

from datasets import DatasetDict
from sklearn.model_selection import train_test_split


def split_by_participant(
    dataset: Any,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 42,
    pid_field: str = "pid",
) -> DatasetDict:
    """Split participant rows without allowing any PID to cross splits."""

    test_ratio = 1.0 - train_ratio - validation_ratio
    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise ValueError(
            "Train, validation, and test ratios must all be positive."
        )
    if pid_field not in dataset.column_names:
        raise KeyError(f"Dataset has no {pid_field!r} column.")

    pids = sorted(str(pid) for pid in dataset[pid_field])
    if len(pids) != len(set(pids)):
        raise ValueError("The dataset contains duplicate participant IDs.")

    participant_count = len(pids)
    train_count = int(train_ratio * participant_count)
    remaining_count = participant_count - train_count
    validation_count = round(
        validation_ratio / (validation_ratio + test_ratio) * remaining_count
    )
    if min(train_count, validation_count, remaining_count - validation_count) <= 0:
        raise ValueError("The requested ratios produce an empty participant split.")

    train_pids, remaining_pids = train_test_split(
        pids,
        train_size=train_count,
        random_state=seed,
        shuffle=True,
    )
    validation_pids, test_pids = train_test_split(
        remaining_pids,
        train_size=validation_count,
        random_state=seed,
        shuffle=True,
    )
    pid_sets = {
        "train": set(train_pids),
        "validation": set(validation_pids),
        "test": set(test_pids),
    }
    return DatasetDict(
        {
            name: dataset.filter(
                lambda row, allowed=allowed: str(row[pid_field]) in allowed,
                desc=f"Selecting {name} participants",
            )
            for name, allowed in pid_sets.items()
        }
    )
