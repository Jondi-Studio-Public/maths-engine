"""The spoken-number reader keeps the point and sign of a decimal and does not join unrelated number words.

`extract_values` reads "point five" and ".5" as 0.5 (not 5), "-.5" as -0.5, "three and three quarters" as
the mixed number 15/4, and "three and four" or "twelve twelve" as two values, not a sum or a product. Only the
British "a hundred and twenty" / "one thousand and five" pattern joins words with "and". A long run of scale
words reads as nothing and never overflows, and the reader is linear in the length of the reply.
"""

from __future__ import annotations

import time
from fractions import Fraction
from types import SimpleNamespace

import pytest

from mathgen.grading import grade
from mathgen.grading.spoken import check_answer, extract_values


def _q(answer, text=None):
    return SimpleNamespace(answer=answer, answer_text=text if text is not None else str(answer))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("point five", [Fraction(1, 2)]),
        ("zero point five", [Fraction(1, 2)]),
        ("minus point five", [Fraction(-1, 2)]),
        ("point oh five", [Fraction(1, 20)]),
        ("three point one four", [Fraction(314, 100)]),
        (".5", [Fraction(1, 2)]),
        ("-.5", [Fraction(-1, 2)]),
        ("0.5", [Fraction(1, 2)]),
        ("a hundred and twenty", [Fraction(120)]),
        ("one thousand and five", [Fraction(1005)]),
        ("two hundred and fifty six thousand three hundred and four", [Fraction(256304)]),
        ("forty-two", [Fraction(42)]),
        ("twelve hundred", [Fraction(1200)]),
        ("three and three quarters", [Fraction(15, 4)]),
        ("two and a half", [Fraction(5, 2)]),
        ("minus three and a half", [Fraction(-7, 2)]),
        ("3 and 3/4", [Fraction(15, 4)]),
        ("three and four", [Fraction(3), Fraction(4)]),
        ("twelve twelve", [Fraction(12), Fraction(12)]),
        ("three four", [Fraction(3), Fraction(4)]),
        ("twenty twelve", [Fraction(20), Fraction(12)]),
        ("five a hundred", [Fraction(5), Fraction(100)]),
        ("three and five quarters", [Fraction(3), Fraction(5, 4)]),
        ("three over five", [Fraction(3, 5)]),
    ],
)
def test_reading(text, expected):
    assert extract_values(text) == expected


@pytest.mark.parametrize("said", ["point five", "zero point five", ".5", "0.5"])
def test_half_is_not_five(said):
    assert check_answer(_q(0.5), said)
    assert not check_answer(_q(5), said)


def test_negative_decimal():
    for said in ("minus point five", "-.5", "negative zero point five"):
        assert check_answer(_q(-0.5), said)
        assert not check_answer(_q(-5), said)
        assert not check_answer(_q(5), said)


def test_three_and_four_is_two_values_not_seven():
    assert not check_answer(_q(7), "three and four")
    assert check_answer(_q({"x": 3, "y": 4}), "three and four")
    assert check_answer(_q([3, 4]), "three and four")


def test_mixed_number_is_one_value():
    assert check_answer(_q({"p": 15, "q": 4}), "three and three quarters")
    assert check_answer(_q(3.75), "three and three quarters")
    assert not check_answer(_q(1.5), "three and three quarters")
    assert not check_answer(_q(3), "three and a half")


def test_adjacent_words_are_not_multiplied():
    assert not check_answer(_q(24), "twelve twelve")
    assert not check_answer(_q(3), "one two")
    assert not check_answer(_q(32), "twenty twelve")
    assert not check_answer(_q(100), "five a hundred")


@pytest.mark.parametrize("repeats", [20, 200, 3000])
def test_scale_word_runs_do_not_overflow(repeats):
    said = "hundred " * repeats
    assert extract_values(said) == []
    for answer in (5, 5.5, {"p": 1, "q": 2}, [1, 2], {"x": 1, "y": 2}):
        assert check_answer(_q(answer), said) is False
    assert check_answer(_q("x**2"), "x to the power of " + said) is False


def test_short_scale_run_still_reads():
    assert extract_values("hundred " * 3) == [Fraction(100)] * 3


@pytest.mark.parametrize("word", ["hundred ", "one ", "one and ", "point one ", "three and a half "])
def test_reader_is_linear(word):
    said = word * (400_000 // len(word))
    start = time.perf_counter()
    extract_values(said)
    assert time.perf_counter() - start < 2.0


def _question(answer, text):
    return SimpleNamespace(
        answer=answer, answer_text=text, parts=None, distractors=[], answer_alt=None, answer_form=None
    )


def test_number_grader_reads_a_point():
    for said in (".5", "0.5", "point five", "zero point five"):
        assert grade("number", _question(0.5, "0.5"), said).correct, said
        assert not grade("number", _question(5, "5"), said).correct, said
    for said in ("-.5", "minus point five"):
        assert grade("number", _question(-0.5, "-0.5"), said).correct, said
        assert not grade("number", _question(-5, "-5"), said).correct, said
        assert not grade("number", _question(5, "5"), said).correct, said


def test_fraction_grader_reads_a_typed_point():
    half = _question({"p": 1, "q": 2}, "1/2")
    assert grade("fraction", half, ".5").correct
    assert not grade("fraction", _question({"p": 5, "q": 1}, "5"), ".5").correct
    assert not grade("fraction", _question({"p": -5, "q": 1}, "-5"), "-.5").correct
    assert grade("fraction", _question({"p": -1, "q": 2}, "-1/2"), "-.5").correct


@pytest.mark.parametrize("kind", ["pair", "list"])
def test_pair_and_list_graders_read_a_point_and_two_values(kind):
    answer = {"x": 0.5, "y": 2} if kind == "pair" else [0.5, 2]
    q = _question(answer, "0.5, 2")
    for said in ("x = .5, y = 2", "point five and two", "0.5 and two"):
        assert grade(kind, q, said).correct, said
    swapped = _question({"x": 5, "y": 2} if kind == "pair" else [5, 2], "5, 2")
    assert not grade(kind, swapped, "point five and two").correct
    three_four = _question({"x": 3, "y": 4} if kind == "pair" else [3, 4], "3, 4")
    assert grade(kind, three_four, "three and four").correct
    assert not grade(kind, _question({"x": 7, "y": 7} if kind == "pair" else [7, 7], "7, 7"), "three and four").correct
