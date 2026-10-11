"""`requested_rounding` reads every rounding phrasing the generators use, and only those."""

import pytest

from mathgen import make
from mathgen.grading.typed import grade_number
from mathgen.text_quality import requested_rounding


@pytest.mark.parametrize(
    ("text", "places"),
    [
        ("Find P(X = 3) to 4 d.p.", 4),
        ("Find the area (1 d.p.)", 1),
        ("Give x (2 d.p.).", 2),
        ("Round to 2dp.", 2),
        ("Round to 3 dp", 3),
        ("to two d.p.", 2),
        ("correct to 2 decimal places", 2),
        ("to the nearest cent", 2),
        ("to the nearest dollar", 0),
        ("to the nearest degree", 0),
        ("to the nearest newton", 0),
        ("to the nearest kilometre", 0),
        ("to the nearest minute", 0),
        ("To the nearest kilojoule, how much", 0),
        ("to the nearest metre", 0),
        ("to the nearest nautical mile", 0),
        ("to the nearest whole number", 0),
        ("to the nearest tenth", 1),
    ],
)
def test_recognised_phrasings(text, places):
    assert requested_rounding(text) == places


@pytest.mark.parametrize(
    "text",
    [
        "Round to the nearest ten.",
        "to the nearest hundred",
        "to the nearest half",
        "to the nearest quarter",
        "the nearest multiple of 5",
        "the nearest lower value",
        "the nearest to 4",
        "the nearest such number",
        "the nearest option",
        "the nearest 1/2",
        "within 1-2 dp",
        "Find x.",
        # two different counts in one text stay ambiguous
        "x to 1 d.p. and y to 2 d.p.",
        "x to the nearest cent and y to 1 decimal place",
    ],
)
def test_other_text_is_not_a_place_count(text):
    assert requested_rounding(text) is None


def test_repeated_same_count_still_resolves():
    assert requested_rounding("x (1 d.p.) and y (1 d.p.)") == 1


def test_unrounded_reply_to_a_dp_question_is_wrong():
    q = make("binomial_probability", seed=0)
    assert requested_rounding(q.text) == 4
    assert grade_number(q, q.answer_text).correct
    assert not grade_number(q, q.answer_text + "37").correct


@pytest.mark.parametrize(
    ("text", "places"),
    [
        ("The ratios (each rounded to 4 d.p.) are: tan 35 = 0.7002. Find EF, correct to 2 decimal places.", 2),
        ("Given values to 3 d.p., find x to the nearest tenth.", 1),
        ("Given values to 3 dp, give the cost to the nearest cent.", 2),
    ],
)
def test_data_precision_in_an_abbreviation_does_not_cancel_the_answer_instruction(text, places):
    """A question gives its data "to 4 d.p." and asks for the answer to N decimal places: the answer's instruction
    is the one a reply must meet. Pooling the two counts left no instruction at all, and an unrounded reply passed."""
    assert requested_rounding(text) == places
