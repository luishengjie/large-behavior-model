"""Run model inference and validate generated responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeneratedResponse:
    text: str
    parsed: dict[str, Any] | None
    valid_json: bool
    valid_schema: bool
    format_repaired: bool


def parse_generated_response(text: str, *, question_type: str) -> GeneratedResponse:
    """Parse and validate a JSON response, removing one surrounding Markdown fence if needed."""

    valid_json = True
    format_repaired = False
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        valid_json = False
        fenced = re.fullmatch(
            r"\s*```(?:json)?\s*(.*?)\s*```\s*",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced is None:
            return GeneratedResponse(text, None, False, False, False)
        try:
            parsed = json.loads(fenced.group(1))
        except json.JSONDecodeError:
            return GeneratedResponse(text, None, False, False, False)
        format_repaired = True

    if not isinstance(parsed, dict):
        return GeneratedResponse(
            text,
            None,
            valid_json,
            False,
            format_repaired,
        )

    if question_type in {"MC", "Matrix"}:
        valid_schema = {"SelectedByPosition", "SelectedText"}.issubset(parsed)
        if question_type == "Matrix" and valid_schema:
            positions = parsed["SelectedByPosition"]
            selected_text = parsed["SelectedText"]
            valid_schema = (
                isinstance(positions, list)
                and isinstance(selected_text, list)
                and len(positions) == len(selected_text)
            )
    elif question_type == "TE":
        valid_schema = "Text" in parsed
    elif question_type in {"Slider", "CS"}:
        valid_schema = "Values" in parsed
    else:
        valid_schema = False

    return GeneratedResponse(
        text,
        parsed,
        valid_json,
        valid_schema,
        format_repaired,
    )


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    *,
    device: Any,
    question_type: str,
    max_new_tokens: int = 256,
) -> GeneratedResponse:
    """Generate response and validate its JSON schema."""

    import torch

    model_inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_ids = generated_ids[0, model_inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return parse_generated_response(text, question_type=question_type)


def load_lora_model(
    base_model: str,
    adapter_path: str,
    *,
    device: Any,
    dtype: Any = None,
) -> tuple[Any, Any]:
    """Load a pretrained model with LoRA adapter and tokenizer for inference."""

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install training dependencies with `uv sync --extra train`."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_path).to(device)
    model.eval()
    return model, tokenizer


def load_base_model(
    base_model: str,
    *,
    device: Any,
    dtype: Any = None,
) -> tuple[Any, Any]:
    """Load a pretrained model and tokenizer for inference."""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Install training dependencies with `uv sync --extra train`."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    return model, tokenizer
