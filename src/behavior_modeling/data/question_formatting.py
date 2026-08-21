"""Format Twin-2K-500 survey questions for language-model training."""

from __future__ import annotations


import json
from collections.abc import Sequence
from typing import Any
from .text import clean_text


def _format_numbered_items(values: Sequence[object], *, separator: str) -> str:
    return "\n".join(
        f"  {position} {separator} {clean_text(value)}"
        for position, value in enumerate(values, start=1)
    )


def _format_matrix(question: dict[str, Any]) -> list[str]:
    rows = question.get("Rows") or []
    columns = question.get("Columns") or []

    sections = ["Question type: Matrix"]
    if columns:
        sections.append(
            "Response options:\n" + _format_numbered_items(columns, separator="=")
        )

    if rows:
        sections.append(
            "\n".join(
                f"{position}. {clean_text(row)}\nAnswer: [Masked]"
                for position, row in enumerate(rows, start=1)
            )
        )

    return sections


def _format_multiple_choice(question: dict[str, Any]) -> list[str]:
    options = question.get("Options") or []
    selector = (question.get("Settings") or {}).get("Selector")
    display_type = (
        "Multiple choice" if selector in {"MAVR", "MAHR"} else "Single choice"
    )

    sections = [f"Question type: {display_type}"]
    if options:
        sections.append(
            "Response options:\n" + _format_numbered_items(options, separator="-")
        )
    sections.append("Answer: [Masked]")
    return sections


def _format_text_entry(question: dict[str, Any]) -> list[str]:
    rows = question.get("Rows") or []
    selector = (question.get("Settings") or {}).get("Selector")

    if selector == "FORM" or rows:
        return [
            "Question type: Text-entry form",
            *(f"{clean_text(row)}: [Masked]" for row in rows),
        ]

    return ["Question type: Text entry", "Answer: [Masked]"]


def _format_slider(question: dict[str, Any]) -> list[str]:
    statements = question.get("Statements") or []
    sections = ["Question type: Slider"]

    if statements:
        sections.append(
            "\n".join(
                f"{position}. {clean_text(statement)}\nAnswer: [Masked]"
                for position, statement in enumerate(statements, start=1)
            )
        )
    else:
        sections.append("Answer: [Masked]")

    return sections


def format_question(question: dict[str, Any]) -> str:
    """Render a complete held-out question with its answers masked.

    Matrix rows and response columns are deliberately retained. A matrix remains
    one survey-question example rather than being split into independent rows.
    """

    question_text = clean_text(question.get("QuestionText"))
    question_type = question.get("QuestionType")

    if question_type == "Matrix":
        details = _format_matrix(question)
    elif question_type == "MC":
        details = _format_multiple_choice(question)
    elif question_type == "TE":
        details = _format_text_entry(question)
    elif question_type == "Slider":
        details = _format_slider(question)
    else:
        raise ValueError(f"Unsupported question type: {question_type!r}.")

    return "\n\n".join(section for section in [question_text, *details] if section)


def format_target(question: dict[str, Any]) -> str:
    """Serialize a question's complete human response as deterministic JSON."""

    answers = question.get("Answers") or {}
    if not answers:
        raise ValueError("Question does not contain an answer.")

    return json.dumps(answers, ensure_ascii=False, sort_keys=True)
