"""Build model prompts while keeping ground-truth responses separate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from datasets import Dataset, Features, Value
from tqdm.auto import tqdm

SYSTEM_PROMPT = """
You are an AI assistant. Answer the new survey question as if you are the
person described in the persona profile, which consists of their previous
survey responses. Remain consistent with the participant's previous answers
and stated characteristics. Return only a valid JSON object matching the
required answer structure. Do not include explanations or Markdown.
""".strip()


PROMPT_FEATURES = Features(
    {
        "pid": Value("string"),
        "block_name": Value("string"),
        "question_id": Value("string"),
        "question_type": Value("string"),
        "prompt_text": Value("large_string"),
        "target_text": Value("large_string"),
    }
)


def truncate_persona(
    persona: str,
    tokenizer: Any,
    *,
    max_tokens: int | None,
) -> str:
    """Retain the beginning and end of a persona within a token budget."""

    if max_tokens is None:
        return persona
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    token_ids = tokenizer.encode(persona, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return persona

    head_size = max_tokens // 2
    tail_size = max_tokens - head_size
    retained_ids = token_ids[:head_size] + token_ids[-tail_size:]
    return tokenizer.decode(retained_ids, skip_special_tokens=True)


def response_format_instruction(question_type: str) -> str:
    """Describe the JSON schema without revealing the held-out answer."""

    if question_type == "Matrix":
        return (
            '{"SelectedByPosition": [integer, ...], '
            '"SelectedText": ["response text", ...]}. Return one entry in each '
            "list for every matrix row, in row order."
        )
    if question_type == "MC":
        return (
            '{"SelectedByPosition": integer, '
            '"SelectedText": "response text"}. For a multiple-selection question, '
            "use lists of positions and response texts instead."
        )
    if question_type == "TE":
        return '{"Text": "response text"}.'
    if question_type in {"Slider", "CS"}:
        return '{"Values": [number, ...]} in question-row order.'
    raise ValueError(f"Unsupported question type: {question_type!r}.")


def build_prompt_messages(
    example: Mapping[str, Any],
    persona: str,
    *,
    include_target: bool,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Create train or inference messages for one survey question."""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "## Persona Profile\n\n"
                f"{persona}\n\n"
                "## New Survey Question\n\n"
                f"{example['question_prompt']}\n\n"
                "## Required Response Format\n\n"
                f"{response_format_instruction(str(example['question_type']))}\n\n"
                "Return only the JSON object."
            ),
        },
    ]
    if include_target:
        messages.append({"role": "assistant", "content": str(example["target"])})
    return messages


def build_prompt_dataset(
    question_dataset: Dataset,
    persona_lookup: Mapping[str, str],
    tokenizer: Any,
    *,
    max_persona_tokens: int | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    progress_desc: str | None = None,
) -> Dataset:
    """Build model prompts and ground-truth responses from participant data."""

    records: list[dict[str, str]] = []
    truncated_personas: dict[str, str] = {}
    examples = (
        tqdm(question_dataset, desc=progress_desc, unit="question")
        if progress_desc
        else question_dataset
    )
    for example in examples:
        pid = str(example["pid"])
        if pid not in persona_lookup:
            raise KeyError(f"No persona found for PID {pid!r}.")

        if pid not in truncated_personas:
            truncated_personas[pid] = truncate_persona(
                persona_lookup[pid], tokenizer, max_tokens=max_persona_tokens
            )
        persona = truncated_personas[pid]
        messages = build_prompt_messages(
            example,
            persona,
            include_target=False,
            system_prompt=system_prompt,
        )
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        records.append(
            {
                "pid": pid,
                "block_name": str(example["block_name"]),
                "question_id": str(example["question_id"]),
                "question_type": str(example["question_type"]),
                "prompt_text": prompt_text,
                "target_text": str(example["target"]),
            }
        )

    if not records:
        return Dataset.from_dict(
            {column: [] for column in PROMPT_FEATURES}, features=PROMPT_FEATURES
        )
    return Dataset.from_list(records, features=PROMPT_FEATURES)
