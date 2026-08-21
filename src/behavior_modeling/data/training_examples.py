"""Build leakage-safe question-level datasets from Twin-2K-500 wave splits."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from datasets import Dataset, Features, Value
from tqdm.auto import tqdm

from .question_formatting import format_question, format_target


TRAINING_FEATURES = Features(
    {
        "pid": Value("string"),
        "block_name": Value("string"),
        "question_id": Value("string"),
        "question_type": Value("string"),
        "question_prompt": Value("large_string"),
        "target": Value("large_string"),
    }
)


def _load_blocks(raw_blocks: object, *, pid: str, field: str) -> list[dict[str, Any]]:
    try:
        blocks = json.loads(raw_blocks) if isinstance(raw_blocks, str) else raw_blocks
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {field!r} for participant {pid!r}.") from error

    if not isinstance(blocks, list):
        raise TypeError(f"Expected a list of blocks in {field!r} for PID {pid!r}.")
    return blocks


def build_question_dataset(
    dataset: Iterable[Mapping[str, Any]],
    *,
    field: str = "wave4_Q_wave4_A",
    pid_field: str = "pid",
    progress_desc: str | None = None,
) -> Dataset:
    """Create one model-target example per answerable survey question.

    Personas are intentionally excluded to avoid storing the same long persona once
    per question. Join them by ``pid`` only while constructing/tokenizing messages.
    Descriptive-information (``DB``) records are skipped because they have no target.
    """

    records: list[dict[str, str]] = []

    participants = (
        tqdm(dataset, desc=progress_desc, unit="participant")
        if progress_desc
        else dataset
    )
    for row_number, participant in enumerate(participants):
        if pid_field not in participant:
            raise KeyError(f"Dataset row {row_number} has no {pid_field!r} field.")
        if field not in participant:
            raise KeyError(f"Dataset row {row_number} has no {field!r} field.")

        pid = str(participant[pid_field])
        blocks = _load_blocks(participant[field], pid=pid, field=field)

        for block in blocks:
            block_name = str(block.get("BlockName") or "")

            for question in block.get("Questions", []):
                question_type = str(question.get("QuestionType") or "")
                if question_type == "DB" or not (question.get("Answers") or {}):
                    continue

                records.append(
                    {
                        "pid": pid,
                        "block_name": block_name,
                        "question_id": str(question.get("QuestionID") or ""),
                        "question_type": question_type,
                        "question_prompt": format_question(question),
                        "target": format_target(question),
                    }
                )

    if not records:
        return Dataset.from_dict(
            {column: [] for column in TRAINING_FEATURES},
            features=TRAINING_FEATURES,
        )

    return Dataset.from_list(records, features=TRAINING_FEATURES)


def build_persona_lookup(
    dataset: Iterable[Mapping[str, Any]],
    *,
    persona_field: str = "wave1_3_persona_text",
    pid_field: str = "pid",
) -> dict[str, str]:
    """Return PID-to-persona text without duplicating personas by question."""

    personas: dict[str, str] = {}
    for row_number, participant in enumerate(dataset):
        if pid_field not in participant:
            raise KeyError(f"Dataset row {row_number} has no {pid_field!r} field.")
        if persona_field not in participant:
            raise KeyError(f"Dataset row {row_number} has no {persona_field!r} field.")

        pid = str(participant[pid_field])
        if pid in personas:
            raise ValueError(f"Duplicate participant ID: {pid!r}.")
        personas[pid] = str(participant[persona_field] or "")

    return personas
