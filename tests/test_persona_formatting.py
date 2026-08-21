import json

from behavior_modeling.data import (
    compact_persona_from_json,
    format_compact_question,
)


def test_compact_mc_resolves_position_to_answer_text() -> None:
    question = {
        "QuestionType": "MC",
        "QuestionText": "Which best describes your spending habits?",
        "Options": [f"Spending option {index}" for index in range(1, 9)],
        "Answers": {"SelectedByPosition": 8},
    }

    compact = format_compact_question(question)

    assert compact.endswith("Answer: Spending option 8")
    assert "Answer: 8" not in compact


def test_compact_matrix_retains_rows_and_human_readable_answers() -> None:
    question = {
        "QuestionType": "Matrix",
        "QuestionText": "Rate each policy.",
        "Rows": ["Carbon tax", "Clean-energy investment"],
        "Columns": ["Oppose", "Neutral", "Support"],
        "Answers": {
            "SelectedByPosition": [1, 3],
            "SelectedText": ["Oppose", "Support"],
        },
    }

    compact = format_compact_question(question)

    assert "- Carbon tax: Oppose" in compact
    assert "- Clean-energy investment: Support" in compact


def test_compact_persona_omits_descriptive_blocks() -> None:
    raw = json.dumps(
        [
            {
                "BlockName": "Instructions",
                "Questions": [
                    {
                        "QuestionType": "DB",
                        "QuestionText": "Read the following instructions.",
                    }
                ],
            },
            {
                "BlockName": "Demographics",
                "Questions": [
                    {
                        "QuestionType": "MC",
                        "QuestionText": "How old are you?",
                        "Options": ["18-29", "30-49"],
                        "Answers": {"SelectedByPosition": 1},
                    }
                ],
            },
        ]
    )

    compact = compact_persona_from_json(raw)

    assert "Instructions" not in compact
    assert "## Demographics" in compact
    assert "Answer: 18-29" in compact
