import json

from datasets import Dataset

from behavior_modeling.data import build_prompt_dataset, build_prompt_messages
from behavior_modeling.models.sft import tokenize_sft_batch


class FakeTokenizer:
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(f"token-{token_id}" for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        assert not tokenize
        rendered = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        return rendered + ("\nassistant:" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}


def test_prompt_messages_exclude_target_during_inference() -> None:
    example = {
        "question_prompt": "Question",
        "question_type": "MC",
        "target": '{"answer": 1}',
    }

    messages = build_prompt_messages(example, "Persona", include_target=False)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert example["target"] not in json.dumps(messages)
    assert "Required Response Format" in messages[1]["content"]


def test_build_prompt_dataset_keeps_target_separate_from_prompt() -> None:
    questions = Dataset.from_list(
        [
            {
                "pid": "1",
                "block_name": "Block",
                "question_id": "Q1",
                "question_type": "MC",
                "question_prompt": "Choose one. Answer: [Masked]",
                "target": '{"SelectedByPosition": 1, "SelectedText": "Yes"}',
            }
        ]
    )

    prompt_data = build_prompt_dataset(
        questions,
        {"1": "Compact persona"},
        FakeTokenizer(),
    )

    assert len(prompt_data) == 1
    assert "Compact persona" in prompt_data[0]["prompt_text"]
    assert prompt_data[0]["target_text"] not in prompt_data[0]["prompt_text"]


def test_tokenization_masks_prompt_labels() -> None:
    tokenizer = FakeTokenizer()
    result = tokenize_sft_batch(
        {"prompt_text": ["one two"], "target_text": ["three"]},
        tokenizer,
        max_sequence_length=10,
    )

    prompt_length = len(tokenizer("one two")["input_ids"])
    assert result["labels"][0][:prompt_length] == [-100] * prompt_length
    assert any(label != -100 for label in result["labels"][0][prompt_length:])
    assert result["keep"] == [True]
