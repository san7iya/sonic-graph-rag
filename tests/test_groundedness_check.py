import pytest

from src.evaluation.groundedness_check import parse_judge_output, unretrieved_titles_mentioned

CATALOG = ["Billie Jean", "Juicy", "Spring (The Four Seasons)", "Spring Day"]


def test_flags_unretrieved_title_mentioned_in_answer():
    leaked = unretrieved_titles_mentioned("Try Billie Jean too.", ["Juicy"], CATALOG)
    assert leaked == ["Billie Jean"]


def test_retrieved_and_shortened_titles_are_not_flagged():
    answer = "Juicy is great, and so is Spring."
    assert unretrieved_titles_mentioned(answer, ["Juicy", "Spring (The Four Seasons)"], ["Juicy", "Spring"]) == []


def test_parse_judge_output_handles_code_fences():
    raw = '```json\n[{"claim": "x", "supported": true}]\n```'
    assert parse_judge_output(raw) == [{"claim": "x", "supported": True}]


def test_parse_judge_output_rejects_garbage():
    with pytest.raises(ValueError):
        parse_judge_output("I cannot do that")
