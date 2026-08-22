#!/usr/bin/env python3
"""Prepare input prompts and responses from preprocessed data."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer

from behavior_modeling.data import (
    build_persona_lookup,
    build_prompt_dataset,
    build_question_dataset,
)
from behavior_modeling.utils import configure_logging, load_config

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"


def prepare_prompt_datasets(
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    *,
    persona_field: str | None,
    exclude_persona: bool = False,
    target_field: str,
) -> DatasetDict:
    """Converts preprocessed dataset into question-level examples for model training and evaluation."""

    data_settings = data_config["data"]
    model_settings = model_config["model"]
    persona_config = model_config.get("persona", {})

    if exclude_persona == bool(persona_field):
        raise ValueError(
            "Specify exactly one persona mode: persona_field or exclude_persona."
        )
    if not target_field:
        raise ValueError("target_field must specify a held-out answer column.")

    preprocessed_data_dir_value = data_settings.get("preprocessed_data_dir")
    if not preprocessed_data_dir_value:
        raise ValueError("Config field 'data.preprocessed_data_dir' must be provided.")

    preprocessed_data_dir = Path(preprocessed_data_dir_value)
    if not preprocessed_data_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {preprocessed_data_dir}. "
            "Run scripts/preprocess_data.py first."
        )
    if not preprocessed_data_dir.is_dir():
        raise NotADirectoryError(
            f"Config field 'data.preprocessed_data_dir' must point to a directory: "
            f"{preprocessed_data_dir}."
        )

    LOGGER.info("Loading preprocessed dataset from %s", preprocessed_data_dir)
    participant_splits = load_from_disk(preprocessed_data_dir)
    if not isinstance(participant_splits, DatasetDict):
        raise TypeError(f"Expected a DatasetDict at {preprocessed_data_dir}.")

    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits - set(participant_splits)
    if missing_splits:
        raise ValueError(
            f"Preprocessed dataset is missing splits: {sorted(missing_splits)}."
        )

    model_name = str(model_settings["name"])
    LOGGER.info("Loading tokenizer for %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    max_persona_tokens = persona_config.get("max_tokens")
    model_splits = {}

    for split_name, participant_dataset in participant_splits.items():
        LOGGER.info(
            "Preparing %s split from %s participants",
            split_name,
            len(participant_dataset),
        )
        questions = build_question_dataset(
            participant_dataset,
            field=target_field,
            progress_desc=f"Extracting {split_name} questions",
        )
        personas = (
            None
            if exclude_persona
            else build_persona_lookup(
                participant_dataset,
                persona_field=str(persona_field),
            )
        )
        model_splits[split_name] = build_prompt_dataset(
            questions,
            personas,
            tokenizer,
            max_persona_tokens=(
                int(max_persona_tokens) if max_persona_tokens is not None else None
            ),
            progress_desc=f"Formatting {split_name} prompts",
        )
        LOGGER.info(
            "Prepared %s model examples for the %s split",
            len(model_splits[split_name]),
            split_name,
        )

    return DatasetDict(model_splits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train, validation, and test prompt datasets."
    )
    parser.add_argument("--data-config", default="configs/data/twin2k500.yaml")
    parser.add_argument(
        "--model-config",
        default="configs/models/qwen25_05b_lora.yaml",
    )
    persona_group = parser.add_mutually_exclusive_group(required=True)
    persona_group.add_argument(
        "--persona-field",
        help=(
            "Participant persona column, such as "
            "'wave1_3_compact_persona_text' or 'wave1_3_persona_text'."
        ),
    )
    persona_group.add_argument(
        "--exclude-persona",
        action="store_true",
        help=(
            "Exclude participant persona text from generated prompts for the "
            "no-persona baseline."
        ),
    )
    parser.add_argument(
        "--target-field",
        required=True,
        help=(
            "Held-out answer column used as the reference target, normally "
            "'wave4_Q_wave1_3_A'."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = configure_logging(
        "prepare_prompt_data",
        args.log_level,
        log_dir=LOG_DIR,
    )
    LOGGER.info("Writing logs to %s", log_path)

    if args.output_dir.exists():
        raise FileExistsError(f"Model-data output already exists: {args.output_dir}.")

    LOGGER.info("Loading data configuration from %s", args.data_config)
    data_config = load_config(args.data_config)
    if not isinstance(data_config.get("data"), dict):
        raise KeyError("Data configuration must contain a 'data' mapping.")

    LOGGER.info("Loading model configuration from %s", args.model_config)
    model_config = load_config(args.model_config)
    if not isinstance(model_config.get("model"), dict):
        raise KeyError("Model configuration must contain a 'model' mapping.")
    if args.exclude_persona:
        LOGGER.info("Preparing no-persona baseline prompts")
    else:
        LOGGER.info("Persona field: %s", args.persona_field)
    LOGGER.info("Target field: %s", args.target_field)
    datasets = prepare_prompt_datasets(
        data_config,
        model_config,
        persona_field=args.persona_field,
        exclude_persona=args.exclude_persona,
        target_field=args.target_field,
    )

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving dataset to %s", args.output_dir)
    datasets.save_to_disk(args.output_dir)
    metadata = {
        "base_model": str(model_config["model"]["name"]),
        "data_config": str(args.data_config),
        "model_config": str(args.model_config),
        "persona_field": args.persona_field,
        "exclude_persona": args.exclude_persona,
        "target_field": args.target_field,
        "max_persona_tokens": (
            None
            if args.exclude_persona
            else model_config.get("persona", {}).get("max_tokens")
        ),
        "splits": {name: len(dataset) for name, dataset in datasets.items()},
    }
    (args.output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOGGER.info(
        "Completed: %s",
        ", ".join(f"{name}={count}" for name, count in metadata["splits"].items()),
    )


if __name__ == "__main__":
    main()
