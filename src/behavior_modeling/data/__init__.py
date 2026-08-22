"""Dataset loading, formatting, and answer-block conversion utilities."""

from .answer_extraction import answer_blocks_to_frame
from .persona_formatting import (
    add_compact_personas,
    compact_persona_from_json,
    format_compact_question,
)
from .question_formatting import format_question, format_target
from .text import clean_text

from .prompt_formatting import (
    NO_PERSONA_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_prompt_dataset,
    build_prompt_messages,
    response_format_instruction,
    truncate_persona,
)
from .question_examples import build_persona_lookup, build_question_dataset

__all__ = [
    "answer_blocks_to_frame",
    "add_compact_personas",
    "build_persona_lookup",
    "build_question_dataset",
    "build_prompt_dataset",
    "build_prompt_messages",
    "clean_text",
    "compact_persona_from_json",
    "format_compact_question",
    "format_question",
    "format_target",
    "NO_PERSONA_SYSTEM_PROMPT",
    "response_format_instruction",
    "SYSTEM_PROMPT",
    "truncate_persona",
]
