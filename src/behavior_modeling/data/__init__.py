"""Dataset loading, formatting, and answer-block conversion utilities."""

from .answer_extraction import answer_blocks_to_frame
from .persona_formatting import (
    add_compact_personas,
    compact_persona_from_json,
    format_compact_question,
)
from .question_formatting import format_question, format_target
from .text import clean_text

from .sft_formatting import (
    SYSTEM_PROMPT,
    build_sft_dataset,
    build_sft_messages,
    response_format_instruction,
    truncate_persona,
)
from .splitting import split_by_participant
from .training_examples import build_persona_lookup, build_question_dataset

__all__ = [
    "answer_blocks_to_frame",
    "add_compact_personas",
    "build_persona_lookup",
    "build_question_dataset",
    "build_sft_dataset",
    "build_sft_messages",
    "clean_text",
    "compact_persona_from_json",
    "format_compact_question",
    "format_question",
    "format_target",
    "response_format_instruction",
    "split_by_participant",
    "SYSTEM_PROMPT",
    "truncate_persona",
]
