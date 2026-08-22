#!/usr/bin/env python3
"""SFT training script"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

from behavior_modeling.training import SFTConfig, run_sft_training
from behavior_modeling.utils import configure_logging, load_config

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_prompt_datasets(prompt_data_dir: Path) -> DatasetDict:
    """Load and validate prepared prompt dataset splits."""

    if not prompt_data_dir.exists():
        raise FileNotFoundError(
            f"Prompt dataset not found at {prompt_data_dir}. "
            "Run scripts/prepare_prompt_data.py first."
        )
    if not prompt_data_dir.is_dir():
        raise NotADirectoryError(
            f"Prompt-data path is not a directory: {prompt_data_dir}."
        )

    datasets = load_from_disk(prompt_data_dir)
    if not isinstance(datasets, DatasetDict):
        raise TypeError(f"Expected a DatasetDict at {prompt_data_dir}.")

    missing_splits = {"train", "validation", "test"} - set(datasets)
    if missing_splits:
        raise ValueError(f"Model dataset is missing splits: {sorted(missing_splits)}.")
    return datasets


def sample_prompt_datasets(
    datasets: DatasetDict,
    *,
    n_samples: int,
    seed: int,
) -> DatasetDict:
    """Select a deterministic random sample from each dataset split."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    return DatasetDict(
        {
            split_name: dataset.shuffle(seed=seed).select(
                range(min(n_samples, len(dataset)))
            )
            for split_name, dataset in datasets.items()
        }
    )


def build_training_config(
    config: dict[str, Any],
    *,
    output_dir: str | Path,
    n_samples: int | None = None,
) -> SFTConfig:
    model = config["model"]
    training = config.get("training", {})
    lora = config.get("lora", {})

    return SFTConfig(
        base_model=model["name"],
        output_dir=str(output_dir),
        cache_dir=model.get("cache_dir"),
        max_sequence_length=int(model.get("max_sequence_length", 8192)),
        seed=int(config.get("seed", 42)),
        n_samples_per_split=n_samples,
        train_batch_size=int(training.get("train_batch_size", 1)),
        eval_batch_size=int(training.get("eval_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 8)),
        epochs=float(training.get("epochs", 1.0)),
        learning_rate=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        scheduler=str(training.get("scheduler", "cosine")),
        logging_steps=int(training.get("logging_steps", 10)),
        eval_steps=int(training.get("eval_steps", 200)),
        save_steps=int(training.get("save_steps", 200)),
        save_total_limit=int(training.get("save_total_limit", 2)),
        report_to=str(training.get("report_to", "none")),
        bf16=bool(training.get("bf16", True)),
        fp16=bool(training.get("fp16", False)),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        lora_rank=int(lora.get("rank", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        lora_target_modules=",".join(
            lora.get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        default="configs/models/qwen25_05b_lora.yaml",
    )
    parser.add_argument(
        "--prompt-data-dir",
        type=Path,
        required=True,
        help="Directory containing the prepared prompt DatasetDict.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to save the trained adapter and metrics.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        help=(
            "Train and evaluate on a deterministic random sample of at most N "
            "examples per split. Omit to use all examples."
        ),
    )
    args = parser.parse_args()

    log_path = configure_logging(
        "train_sft",
        args.log_level,
        log_dir=PROJECT_ROOT / "logs",
    )
    LOGGER.info("Writing logs to %s", log_path)

    if args.prompt_data_dir.resolve() == args.output_dir.resolve():
        parser.error(
            "--prompt-data-dir and --output-dir must be different directories."
        )
    if args.n_samples is not None and args.n_samples <= 0:
        parser.error("--n-samples must be positive.")

    LOGGER.info("Loading model configuration from %s", args.model_config)
    config = load_config(args.model_config)
    if not isinstance(config.get("model"), dict):
        raise KeyError("Model configuration must contain a 'model' mapping.")
    LOGGER.info("Loading prepared prompt data from %s", args.prompt_data_dir)
    prompt_datasets = load_prompt_datasets(args.prompt_data_dir)
    if args.n_samples is not None:
        seed = int(config.get("seed", 42))
        prompt_datasets = sample_prompt_datasets(
            prompt_datasets,
            n_samples=args.n_samples,
            seed=seed,
        )
        LOGGER.info(
            "Sampled datasets with seed %d: %s",
            seed,
            ", ".join(
                f"{split}={len(dataset):,}"
                for split, dataset in prompt_datasets.items()
            ),
        )
    training_config = build_training_config(
        config,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
    )
    LOGGER.info("Starting SFT; outputs will be saved to %s", args.output_dir)
    run_sft_training(prompt_datasets, training_config)
    LOGGER.info("SFT complete")


if __name__ == "__main__":
    main()
