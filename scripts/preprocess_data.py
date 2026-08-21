#!/usr/bin/env python3
"""Preprocess Twin-2K-500 into participant-level dataset splits.
This script creates train, validation, and test splits, generates compact persona text, and saves the resulting dataset locally.
"""

from __future__ import annotations

import json
import shutil
import logging
import argparse

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from datasets import DatasetDict, Value, load_dataset

from behavior_modeling.data import add_compact_personas
from behavior_modeling.utils import configure_logging, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data" / "twin2k500.yaml"
LOGGER = logging.getLogger(__name__)


def preprocess_dataset(
    *,
    dataset_id: str,
    dataset_config: str,
    dataset_split: str,
    train_ratio: float,
    validation_ratio: float,
    seed: int = 42,
) -> DatasetDict:
    """Downloads data, split data into train, val and test, and generate compact personas."""

    test_ratio = 1.0 - train_ratio - validation_ratio
    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise ValueError("Train, validation, and test ratios must all be positive.")

    dataset = load_dataset(dataset_id, dataset_config, split=dataset_split)
    dataset = dataset.cast_column("pid", Value("string"))

    # Split dataset
    pids = np.array(sorted(dataset["pid"]))

    if len(pids) != len(set(pids)):
        raise ValueError("The dataset contains duplicate participant IDs.")

    rng = np.random.default_rng(seed)
    rng.shuffle(pids)

    participant_count = len(pids)

    train_count = int(train_ratio * participant_count)

    validation_count = int(validation_ratio * participant_count)

    train_pids = set(pids[:train_count])

    val_pids = set(pids[train_count : train_count + validation_count])

    test_pids = set(pids[train_count + validation_count :])

    assert train_pids.isdisjoint(val_pids)
    assert train_pids.isdisjoint(test_pids)
    assert val_pids.isdisjoint(test_pids)
    assert train_pids | val_pids | test_pids == set(pids)

    train_ds = dataset.filter(lambda row: row["pid"] in train_pids)
    val_ds = dataset.filter(lambda row: row["pid"] in val_pids)
    test_ds = dataset.filter(lambda row: row["pid"] in test_pids)

    # Add compact personas
    train_ds = add_compact_personas(train_ds)
    val_ds = add_compact_personas(val_ds)
    test_ds = add_compact_personas(test_ds)

    return DatasetDict(
        {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds,
        }
    )


def build_preprocess_metadata(
    dataset: DatasetDict,
    *,
    dataset_id: str,
    dataset_config: str,
    dataset_split: str,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> dict[str, Any]:
    """Returns the metadata for the preprocessed dataset."""
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_dataset": dataset_id,
        "source_config": dataset_config,
        "source_split": dataset_split,
        "seed": seed,
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "test_ratio": 1 - train_ratio - validation_ratio,
        "participant_counts": {name: len(split) for name, split in dataset.items()},
        "columns": dataset["train"].column_names,
    }


def save_preprocessed_dataset(
    dataset: DatasetDict,
    output_dir: str | Path,
    metadata: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Save preprocessed data and metadata."""

    out_path = Path(output_dir).expanduser().resolve()

    protected_paths = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
    }
    if out_path in protected_paths:
        raise ValueError(f"Refusing to use protected output directory: {out_path}.")

    if out_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {out_path}. "
                "Select another directory or delete existing directory."
            )
        shutil.rmtree(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(out_path)
    metadata_path = out_path / "preprocess_metadata.json"

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Twin-2K-500, split data, generate compact personas and save processed results."
        )
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=DEFAULT_DATA_CONFIG_PATH,
        help=f"Path to the data config file. Default: {DEFAULT_DATA_CONFIG_PATH}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=("Overwrite existing output directory"),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = configure_logging(
        "preprocess_data",
        args.log_level,
        log_dir=PROJECT_ROOT / "logs",
    )
    LOGGER.info("Writing logs to %s", log_path)
    LOGGER.info("Loading data configuration from %s", args.data_config)
    config = load_config(args.data_config)
    if not isinstance(config.get("data"), dict):
        raise KeyError("Data configuration must 'data'.")
    data_config = config["data"]
    seed = int(config.get("seed", 42))

    dataset_id = data_config.get("dataset_id", "LLM-Digital-Twin/Twin-2K-500")
    dataset_config = data_config.get("dataset_config", "wave_split")
    dataset_split = data_config.get("split", "data")
    train_ratio = float(data_config.get("train_ratio", 0.70))
    validation_ratio = float(data_config.get("validation_ratio", 0.10))

    configured_output_dir = Path(
        data_config.get("preprocessed_data_dir", "data/processed/twin2k500_compact")
    ).expanduser()
    output_path = (
        configured_output_dir
        if configured_output_dir.is_absolute()
        else PROJECT_ROOT / configured_output_dir
    ).resolve()

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_path}. "
            "Use --overwrite to replace it."
        )

    dataset = preprocess_dataset(
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    metadata = build_preprocess_metadata(
        dataset,
        dataset_id=dataset_id,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        seed=seed,
    )

    output_path = save_preprocessed_dataset(
        dataset,
        output_path,
        metadata,
        overwrite=args.overwrite,
    )

    LOGGER.info("Saved preprocessed dataset to %s", output_path)
    for split_name, split in dataset.items():
        LOGGER.info("%s: %s participants", split_name, f"{len(split):,}")


if __name__ == "__main__":
    main()
