"""Convert Twin-2K-500 JSON answer blocks to wide response tables."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd


def _normalize_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _catalog_key(block_name: object, question_id: object) -> tuple[str, str]:
    return _normalize_name(block_name), str(question_id or "")


def _build_catalog_lookup(
    question_catalog: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}

    for item in question_catalog:
        key = _catalog_key(item.get("BlockName"), item.get("QuestionID"))
        if key in lookup:
            raise ValueError(
                "Question catalog contains a duplicate (BlockName, QuestionID) "
                f"key: {key!r}."
            )
        lookup[key] = item

    return lookup


def _answer_values(question: Mapping[str, Any]) -> list[Any]:
    """Extract atomic response values in the catalog column order."""

    answers = question.get("Answers") or {}

    if "SelectedByPosition" in answers:
        values = answers["SelectedByPosition"]
    elif "Values" in answers:
        values = answers["Values"]
    elif "Text" in answers:
        values = [answers["Text"]]
    else:
        values = []

    if isinstance(values, list):
        return values
    return [values]


def answer_blocks_to_frame(
    dataset: Iterable[Mapping[str, Any]],
    *,
    field: str,
    question_catalog: Sequence[Mapping[str, Any]],
    pid_field: str = "pid",
    uppercase_columns: bool = True,
    numeric: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """Convert a nested answer-block field to a participant-by-response table.

    Parameters
    ----------
    dataset:
        An iterable of Hugging Face dataset rows or equivalent mappings.
    field:
        JSON field to convert, for example ``wave4_Q_wave1_3_A`` or
        ``wave4_Q_wave4_A``.
    question_catalog:
        Parsed ``question_catalog.json``. Its ``csv_columns`` entries define
        the atomic response-column names and ordering.
    pid_field:
        Name of the participant identifier field.
    uppercase_columns:
        Uppercase response columns to match the paper's evaluation mappings.
    numeric:
        Coerce response values to numbers, replacing invalid values with NaN.
    strict:
        Raise on missing catalog entries, answer/column count mismatches, and
        duplicate response columns. If false, malformed questions are skipped.

    Returns
    -------
    pandas.DataFrame
        One row per participant, indexed by ``pid``, and one column per atomic
        response. No intermediate JSON files are written.
    """

    catalog_lookup = _build_catalog_lookup(question_catalog)
    records: list[dict[str, Any]] = []
    participant_ids: set[str] = set()

    for row_number, row in enumerate(dataset):
        if pid_field not in row:
            raise KeyError(f"Dataset row {row_number} has no {pid_field!r} field.")
        if field not in row:
            raise KeyError(f"Dataset row {row_number} has no {field!r} field.")

        pid = str(row[pid_field])
        if pid in participant_ids:
            raise ValueError(f"Duplicate participant ID: {pid!r}.")
        participant_ids.add(pid)

        raw_blocks = row[field]
        try:
            blocks = (
                json.loads(raw_blocks) if isinstance(raw_blocks, str) else raw_blocks
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in {field!r} for participant {pid!r}."
            ) from error

        response_record: dict[str, Any] = {"pid": pid}
        for block in blocks:
            block_name = block.get("BlockName")

            for question in block.get("Questions", []):
                question_id = question.get("QuestionID")
                key = _catalog_key(block_name, question_id)
                catalog_item = catalog_lookup.get(key)

                if catalog_item is None:
                    if strict:
                        raise KeyError(
                            "Question is missing from the catalog: "
                            f"pid={pid!r}, block={block_name!r}, "
                            f"question_id={question_id!r}."
                        )
                    continue

                columns = list(catalog_item.get("csv_columns") or [])
                values = _answer_values(question)
                if len(columns) != len(values):
                    if strict:
                        raise ValueError(
                            "Answer count does not match catalog csv_columns: "
                            f"pid={pid!r}, block={block_name!r}, "
                            f"question_id={question_id!r}, answers={len(values)}, "
                            f"columns={len(columns)}."
                        )
                    continue

                for column, value in zip(columns, values, strict=True):
                    output_column = column.upper() if uppercase_columns else column
                    if output_column in response_record:
                        if strict:
                            raise ValueError(
                                f"Duplicate response column {output_column!r} for "
                                f"participant {pid!r}."
                            )
                        continue
                    response_record[output_column] = value

        records.append(response_record)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return pd.DataFrame(index=pd.Index([], name="pid"))

    frame = frame.set_index("pid")
    if numeric:
        frame = frame.apply(pd.to_numeric, errors="coerce")

    return frame.sort_index(axis=1)
