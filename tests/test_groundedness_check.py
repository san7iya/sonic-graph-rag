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


def test_numbers_match_spelled_out_forms():
    from src.evaluation.groundedness_check import number_in_words, unsupported_specifics
    assert number_in_words(41) == "forty one" and number_in_words(11) == "eleven" and number_in_words(300) == "three hundred"
    ctx = "It spent forty-one weeks in the Top 10 and is certified eleven times Platinum."
    assert unsupported_specifics("It spent 41 weeks and is 11x Platinum.", ctx) == []
    assert unsupported_specifics("It spent 42 weeks.", ctx) == ["42"]


def test_titles_not_in_context_counts_graph_facts_as_context():
    from src.evaluation.groundedness_check import titles_not_in_context
    ctx = '[1] "Song A" by X\nGraph facts: other songs by this artist: Song B'
    catalog = ["Song A", "Song B", "Song C"]
    assert titles_not_in_context("Song B and Song C are similar.", ctx, catalog) == ["Song C"]


def test_bold_labels_and_sentence_openers_are_not_flagged_as_names():
    from src.evaluation.groundedness_check import unsupported_specifics
    ctx = "Bob Marley recorded reggae songs in Kingston."
    answer = "* **Solo Innovation:** Bob Marley recorded reggae.\nAdditionally, he worked in Kingston. Because of this."
    assert unsupported_specifics(answer, ctx) == []


def test_real_unsupported_name_in_a_sentence_is_still_flagged():
    from src.evaluation.groundedness_check import unsupported_specifics
    ctx = "Bob Marley recorded reggae songs in Kingston."
    assert "Peter Tosh" in unsupported_specifics("He recorded with Peter Tosh in Kingston.", ctx)
    assert "Jamaica" in unsupported_specifics("Born in Jamaica, Bob Marley recorded reggae.", ctx)


def test_one_word_titles_only_count_when_quoted():
    from src.evaluation.groundedness_check import titles_not_in_context
    ctx = "Some other context"
    assert titles_not_in_context("It includes one piano album.", ctx, ["One"]) == []
    assert titles_not_in_context('It includes "One" by Metallica.', ctx, ["One"]) == ["One"]
    assert titles_not_in_context("Listen to Billie Jean now.", ctx, ["Billie Jean"]) == ["Billie Jean"]
