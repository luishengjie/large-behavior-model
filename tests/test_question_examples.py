"""Tests for question-level example construction."""

import json

from behavior_modeling.data import build_persona_lookup, build_question_dataset


def test_build_question_dataset_preserves_complete_matrix() -> None:
    matrix = {
        "QuestionID": "QID159",
        "QuestionText": "Please complete the statements below.",
        "QuestionType": "Matrix",
        "Rows": ["Linda is a teacher", "Linda is a bank teller"],
        "Columns": ["Improbable", "Probable"],
        "Answers": {
            "SelectedByPosition": [2, 1],
            "SelectedText": ["Probable", "Improbable"],
        },
    }
    dataset = [
        {
            "pid": "1710",
            "wave1_3_persona_text": "Persona text",
            "wave4_Q_wave4_A": json.dumps(
                [
                    {
                        "BlockName": "Linda",
                        "Questions": [matrix],
                    }
                ]
            ),
        }
    ]

    questions = build_question_dataset(dataset)

    assert len(questions) == 1
    prompt = questions[0]["question_prompt"]
    assert "Linda is a teacher" in prompt
    assert "Linda is a bank teller" in prompt
    assert "1 = Improbable" in prompt
    assert "2 = Probable" in prompt
    assert prompt.count("Answer: [Masked]") == 2
    assert json.loads(questions[0]["target"]) == matrix["Answers"]


def test_build_question_dataset_skips_descriptive_blocks() -> None:
    dataset = [
        {
            "pid": "1",
            "wave4_Q_wave4_A": json.dumps(
                [
                    {
                        "BlockName": "Introduction",
                        "Questions": [
                            {
                                "QuestionType": "DB",
                                "QuestionText": "Introduction",
                            }
                        ],
                    }
                ]
            ),
        }
    ]

    assert len(build_question_dataset(dataset)) == 0


def test_build_persona_lookup_keeps_one_persona_per_pid() -> None:
    lookup = build_persona_lookup(
        [
            {"pid": 1, "wave1_3_persona_text": "First persona"},
            {"pid": 2, "wave1_3_persona_text": "Second persona"},
        ]
    )

    assert lookup == {"1": "First persona", "2": "Second persona"}
