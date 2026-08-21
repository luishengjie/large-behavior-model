"""Tests for response parsing during model inference."""

from behavior_modeling.evaluation import parse_generated_response


def test_parse_complete_matrix_response() -> None:
    response = parse_generated_response(
        '{"SelectedByPosition":[2,4],"SelectedText":["Oppose","Support"]}',
        question_type="Matrix",
    )

    assert response.valid_json
    assert response.valid_schema
    assert not response.format_repaired


def test_reject_matrix_response_with_mismatched_list_lengths() -> None:
    response = parse_generated_response(
        '{"SelectedByPosition":[2,4],"SelectedText":["Oppose"]}',
        question_type="Matrix",
    )

    assert response.valid_json
    assert not response.valid_schema


def test_reject_non_json_response() -> None:
    response = parse_generated_response(
        "Let me reason through the answer.",
        question_type="MC",
    )

    assert not response.valid_json
    assert not response.valid_schema
    assert not response.format_repaired


def test_parse_json_inside_markdown_fence() -> None:
    response = parse_generated_response(
        """```json
        {"SelectedByPosition": 2, "SelectedText": "No"}
        ```""",
        question_type="MC",
    )

    assert response.parsed == {
        "SelectedByPosition": 2,
        "SelectedText": "No",
    }
    assert not response.valid_json
    assert response.valid_schema
    assert response.format_repaired
