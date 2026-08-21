#!/usr/bin/env python3
"""Model evaluation script."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk
from tqdm.auto import tqdm

from behavior_modeling.evaluation import (
    generate_response,
    load_base_model,
    load_lora_model,
)
from behavior_modeling.utils import configure_logging, load_config

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"


def select_device() -> Any:
    """Select device for torch."""

    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data_split(prompt_data_dir: Path, split: str) -> Any:
    """Load data split."""

    if not prompt_data_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {prompt_data_dir}.")
    if not prompt_data_dir.is_dir():
        raise NotADirectoryError(f"Data path not a directory: {prompt_data_dir}.")

    datasets = load_from_disk(prompt_data_dir)
    if not isinstance(datasets, DatasetDict):
        raise TypeError(f"Expected a DatasetDict at {prompt_data_dir}.")
    if split not in datasets:
        raise KeyError(f"Model dataset has no {split!r} split.")

    required_columns = {
        "pid",
        "question_id",
        "question_type",
        "prompt_text",
        "target_text",
    }
    missing_columns = required_columns - set(datasets[split].column_names)
    if missing_columns:
        raise ValueError(
            f"Model {split!r} split is missing columns: {sorted(missing_columns)}."
        )
    return datasets[split]


def evaluate_examples(
    dataset: Any,
    model: Any,
    tokenizer: Any,
    *,
    device: Any,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate predictions and calculate JSON, schema, and exact-match metrics."""

    records: list[dict[str, Any]] = []
    type_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "examples": 0,
            "valid_json": 0,
            "format_repaired": 0,
            "valid_schema": 0,
            "exact": 0,
        }
    )

    for example in tqdm(dataset, desc="Generating responses", unit="question"):
        question_type = str(example["question_type"])
        generated = generate_response(
            model,
            tokenizer,
            str(example["prompt_text"]),
            device=device,
            question_type=question_type,
            max_new_tokens=max_new_tokens,
        )
        target = json.loads(str(example["target_text"]))
        exact_match = bool(generated.valid_schema and generated.parsed == target)

        counts = type_counts[question_type]
        counts["examples"] += 1
        counts["valid_json"] += int(generated.valid_json)
        counts["format_repaired"] += int(generated.format_repaired)
        counts["valid_schema"] += int(generated.valid_schema)
        counts["exact"] += int(exact_match)

        records.append(
            {
                "pid": str(example["pid"]),
                "block_name": str(example.get("block_name", "")),
                "question_id": str(example["question_id"]),
                "question_type": question_type,
                "prediction_text": generated.text,
                "prediction": generated.parsed,
                "target": target,
                "valid_json": generated.valid_json,
                "format_repaired": generated.format_repaired,
                "valid_schema": generated.valid_schema,
                "exact_match": exact_match,
            }
        )

    total = len(records)
    summary = {
        "examples": total,
        "valid_json_rate": (
            sum(row["valid_json"] for row in records) / total if total else 0.0
        ),
        "format_repair_rate": (
            sum(row["format_repaired"] for row in records) / total if total else 0.0
        ),
        "valid_schema_rate": (
            sum(row["valid_schema"] for row in records) / total if total else 0.0
        ),
        "exact_match_accuracy": (
            sum(row["exact_match"] for row in records) / total if total else 0.0
        ),
        "by_question_type": {
            question_type: {
                "examples": counts["examples"],
                "valid_json_rate": counts["valid_json"] / counts["examples"],
                "format_repair_rate": counts["format_repaired"] / counts["examples"],
                "valid_schema_rate": counts["valid_schema"] / counts["examples"],
                "exact_match_accuracy": counts["exact"] / counts["examples"],
            }
            for question_type, counts in sorted(type_counts.items())
        },
    }
    return records, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a base model or LoRA adapter on prepared prompt data."
    )
    parser.add_argument(
        "--model-config",
        default="configs/models/qwen25_05b_lora.yaml",
    )
    parser.add_argument("--prompt-data-dir", type=Path, required=True)
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        help="LoRA adapter directory. Omit to evaluate the base model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--n-samples",
        "--max-examples",
        dest="n_samples",
        type=int,
        help=(
            "Evaluate a deterministic random sample of N examples. "
            "Omit to evaluate the complete split."
        ),
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
        "evaluate_model",
        args.log_level,
        log_dir=LOG_DIR,
    )
    LOGGER.info("Writing logs to %s", log_path)

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if args.n_samples is not None and args.n_samples <= 0:
        raise ValueError("--n-samples must be positive.")
    if args.output_dir.exists():
        raise FileExistsError(f"Evaluation output already exists: {args.output_dir}.")
    if args.adapter_dir is not None and not args.adapter_dir.is_dir():
        raise NotADirectoryError(f"Adapter directory not found: {args.adapter_dir}.")

    LOGGER.info("Loading model configuration from %s", args.model_config)
    config = load_config(args.model_config)
    if not isinstance(config.get("model"), dict):
        raise KeyError("Model configuration must contain a 'model' mapping.")
    base_model = str(config["model"]["name"])
    seed = int(config.get("seed", 42))
    LOGGER.info("Loading %s split from %s", args.split, args.prompt_data_dir)
    dataset = load_data_split(args.prompt_data_dir, args.split)
    if args.n_samples is not None:
        sample_size = min(args.n_samples, len(dataset))
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))
        LOGGER.info(
            "Selected a deterministic random sample of %s examples with seed=%s",
            sample_size,
            seed,
        )
    LOGGER.info("Evaluating %s examples", len(dataset))

    device = select_device()
    LOGGER.info("Using device: %s", device)
    if args.adapter_dir is None:
        LOGGER.info("Loading baseline model: %s", base_model)
        model, tokenizer = load_base_model(base_model, device=device)
        model_type = "base"
    else:
        LOGGER.info(
            "Loading base model %s with LoRA adapter %s",
            base_model,
            args.adapter_dir,
        )
        model, tokenizer = load_lora_model(
            base_model,
            str(args.adapter_dir),
            device=device,
        )
        model_type = "lora"

    LOGGER.info("Starting generation with max_new_tokens=%s", args.max_new_tokens)
    records, summary = evaluate_examples(
        dataset,
        model,
        tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    summary.update(
        {
            "model_type": model_type,
            "base_model": base_model,
            "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
            "prompt_data_dir": str(args.prompt_data_dir),
            "split": args.split,
            "device": str(device),
            "max_new_tokens": args.max_new_tokens,
            "n_samples": args.n_samples,
            "sample_seed": seed if args.n_samples is not None else None,
        }
    )

    LOGGER.info("Saving evaluation results to %s", args.output_dir)
    args.output_dir.mkdir(parents=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    LOGGER.info(
        "Evaluation complete: valid_json=%.2f%%, valid_schema=%.2f%%, "
        "exact_match=%.2f%%",
        100 * summary["valid_json_rate"],
        100 * summary["valid_schema_rate"],
        100 * summary["exact_match_accuracy"],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
