"""Convert Twin-2K-500 persona JSON into compact, deterministic text by formatting answered questions and removing redundant survey metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from datasets import Dataset

from .text import clean_text


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _selected_answers(question: Mapping[str, Any]) -> list[str]:
    """Return selected answers and resolve positions when text is unavailable."""

    answers = question.get("Answers") or {}
    selected_text = _as_list(answers.get("SelectedText"))
    if selected_text:
        return [clean_text(value) for value in selected_text]

    positions = _as_list(answers.get("SelectedByPosition"))
    options = question.get("Options") or question.get("Columns") or []
    resolved: list[str] = []

    for position in positions:
        try:
            option_index = int(float(position)) - 1
        except (TypeError, ValueError):
            resolved.append(clean_text(position))
            continue

        if 0 <= option_index < len(options):
            resolved.append(clean_text(options[option_index]))
        else:
            resolved.append(f"Selected option {clean_text(position)}")

    return resolved


def _compact_multiple_choice(question: Mapping[str, Any]) -> str:
    answers = _selected_answers(question)
    if not answers:
        return ""
    return f"{clean_text(question.get('QuestionText'))}\nAnswer: {'; '.join(answers)}"


def _compact_matrix(question: Mapping[str, Any]) -> str:
    rows = question.get("Rows") or []
    answers = _selected_answers(question)
    lines = [clean_text(question.get("QuestionText"))]

    for row, answer in zip(rows, answers):
        lines.append(f"- {clean_text(row)}: {answer}")

    return "\n".join(line for line in lines if line)


def _compact_text_entry(question: Mapping[str, Any]) -> str:
    question_text = clean_text(question.get("QuestionText"))
    rows = question.get("Rows") or []
    text_answer = (question.get("Answers") or {}).get("Text")

    if isinstance(text_answer, list):
        answer_lookup: dict[str, Any] = {}
        for item in text_answer:
            if isinstance(item, Mapping):
                answer_lookup.update(item)

        lines = [question_text]
        for row in rows:
            if row in answer_lookup:
                lines.append(f"- {clean_text(row)}: {clean_text(answer_lookup[row])}")
        return "\n".join(line for line in lines if line)

    if text_answer is None:
        return ""
    return f"{question_text}\nAnswer: {clean_text(text_answer)}"


def _compact_slider(question: Mapping[str, Any]) -> str:
    statements = question.get("Statements") or []
    values = _as_list((question.get("Answers") or {}).get("Values"))
    lines = [clean_text(question.get("QuestionText"))]

    if statements:
        for statement, value in zip(statements, values):
            lines.append(f"- {clean_text(statement)}: {clean_text(value)}")
    elif values:
        lines.append("Answer: " + "; ".join(clean_text(value) for value in values))

    return "\n".join(line for line in lines if line)


def _compact_constant_sum(question: Mapping[str, Any]) -> str:
    rows = question.get("Rows") or []
    values = _as_list((question.get("Answers") or {}).get("Values"))
    lines = [clean_text(question.get("QuestionText"))]
    for row, value in zip(rows, values):
        lines.append(f"- {clean_text(row)}: {clean_text(value)}")
    return "\n".join(line for line in lines if line)


def format_compact_question(question: Mapping[str, Any]) -> str:
    """Format a persona question as compact text."""

    question_type = question.get("QuestionType")
    if question_type == "DB":
        return ""
    if question_type == "MC":
        return _compact_multiple_choice(question)
    if question_type == "Matrix":
        return _compact_matrix(question)
    if question_type == "TE":
        return _compact_text_entry(question)
    if question_type == "Slider":
        return _compact_slider(question)
    if question_type == "CS":
        return _compact_constant_sum(question)
    raise ValueError(f"Unsupported persona question type: {question_type!r}.")


def compact_persona_from_json(
    raw_persona_json: str | list[dict[str, Any]],
    *,
    include_block_names: bool = True,
) -> str:
    """Convert participant persona JSON to compact text."""

    blocks = (
        json.loads(raw_persona_json)
        if isinstance(raw_persona_json, str)
        else raw_persona_json
    )
    if not isinstance(blocks, list):
        raise TypeError("Persona JSON must contain a list of blocks.")

    formatted_blocks: list[str] = []
    for block in blocks:
        questions = [
            formatted
            for question in block.get("Questions", [])
            if (formatted := format_compact_question(question))
        ]
        if not questions:
            continue

        block_text = "\n\n".join(questions)
        block_name = clean_text(block.get("BlockName"))
        if include_block_names and block_name:
            block_text = f"## {block_name}\n\n{block_text}"
        formatted_blocks.append(block_text)

    return "\n\n".join(formatted_blocks)


def add_compact_personas(
    dataset: Dataset,
    *,
    source_field: str = "wave1_3_persona_json",
    output_field: str = "wave1_3_compact_persona_text",
) -> Dataset:
    """Add a compact persona column to a Hugging Face dataset."""

    return dataset.map(
        lambda row: {output_field: compact_persona_from_json(row[source_field])},
        desc="Building compact personas",
    )
