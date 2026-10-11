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


CONTEXT = ('[1] "No Woman No Cry" by Bob Marley (genre: reggae)\nBio: Released in 1974 on the album Natty Dread, '
           'the song was recorded in Kingston.')


def test_supported_answer_has_no_unsupported_specifics():
    from src.evaluation.groundedness_check import unsupported_specifics
    answer = "No Woman No Cry by Bob Marley was released in 1974 on Natty Dread."
    assert unsupported_specifics(answer, CONTEXT) == []


def test_flags_invented_year_and_name():
    from src.evaluation.groundedness_check import unsupported_specifics
    flagged = unsupported_specifics("Bob Marley released it in 1975 with Peter Tosh in Jamaica.", CONTEXT)
    assert "1975" in flagged and "Peter Tosh" in flagged and "Jamaica" in flagged
    assert "Bob Marley" not in flagged


def test_sentence_starter_words_are_not_flagged():
    from src.evaluation.groundedness_check import unsupported_specifics
    assert unsupported_specifics("According to the context, it is reggae.", CONTEXT) == []
