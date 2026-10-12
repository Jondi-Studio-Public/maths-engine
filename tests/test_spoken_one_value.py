"""A spoken single-number answer states one value, not negated or hedged (`mathgen.grading.spoken.check_answer`).

"not twelve", "twelve or thirteen" and "twelve and a half" each hold a twelve but are not the answer 12. The
number-word reading and the +/-0.051 transcript tolerance are unchanged, and answers of several values
(a dict of components, a list) still match on each component being present.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mathgen.grading.spoken import check_answer


def _q(answer):
    return SimpleNamespace(answer=answer, answer_text=str(answer))


@pytest.mark.parametrize(
    "transcript",
    [
        "twelve",
        "12",
        "um twelve",
        "uh, twelve",
        "it's twelve",
        "it is twelve",
        "the answer is twelve",
        "the answer is 12",
        "I think it's twelve",
        "twelve centimetres",
        "x equals twelve",
        "twelve point oh",
        "twelve point oh five",  # within the 0.051 tolerance
        "12.04",
    ],
)
def test_one_value_with_fillers_is_right(transcript):
    assert check_answer(_q(12), transcript)
    assert check_answer(_q(12.0), transcript)


@pytest.mark.parametrize(
    "transcript",
    [
        "not twelve",
        "it's not twelve",
        "it isn't twelve",
        "never twelve",
        "neither twelve",
        "twelve or thirteen",
        "twelve or 13",
        "twelve and a half",
        "twelve, thirteen",
        "twelve twelve",
        "twelve plus or minus one",
        "twelve plus minus one",
        "twelve point five",
        "thirteen",
        "minus twelve",
        "negative twelve",
        "twelve not thirteen",
        "",
        "um",
    ],
)
def test_negated_hedged_or_wrong_is_not_twelve(transcript):
    assert not check_answer(_q(12), transcript)


def test_the_fallback_transcript_follows_the_same_rule():
    assert check_answer(_q(12), "not twelve", fallback="twelve")
    assert not check_answer(_q(12), "not twelve", fallback="twelve or thirteen")


def test_twelve_point_five_is_12_5():
    assert check_answer(_q(12.5), "twelve point five")
    assert check_answer(_q(12.5), "12.5")
    assert not check_answer(_q(12.5), "twelve")  # 0.5 away
    assert not check_answer(_q(12.5), "not twelve point five")


def test_negative_and_zero_answers():
    assert check_answer(_q(-3), "minus three")
    assert check_answer(_q(-3), "negative three")
    assert not check_answer(_q(-3), "three")
    assert not check_answer(_q(-3), "not minus three")
    assert check_answer(_q(0), "zero")
    assert not check_answer(_q(0), "not zero")


def test_number_words_still_read():
    assert check_answer(_q(144), "a hundred and forty four")
    assert check_answer(_q(36.9), "thirty six point nine")
    assert not check_answer(_q(144), "a hundred and forty four or a hundred and forty five")


def test_fraction_answers():
    answer = {"p": 3, "q": 5}
    assert check_answer(_q(answer), "three fifths")
    assert check_answer(_q(answer), "three over five")
    assert check_answer(_q(answer), "3/5")
    assert not check_answer(_q(answer), "not three fifths")
    assert not check_answer(_q(answer), "three fifths or two fifths")


def test_tolerance_is_unchanged():
    assert check_answer(_q(6.1), "six point one five")  # 0.05 away
    assert not check_answer(_q(6.1), "six point two")  # 0.1 away


def test_several_valued_answers_still_match_on_each_component():
    assert check_answer(_q({"x": 2, "y": 3}), "x is two and y is three")
    assert check_answer(_q([2, 3]), "a is two and b is three")
    assert not check_answer(_q([2, 3]), "two")
