"""A value reply may end with a full stop or begin with "=" in every value kind, as a number reply always could."""

from __future__ import annotations

import pytest
import sympy

from mathgen.core import Question
from mathgen.grading import grade


def _q(kind_answer, answer_text, text="Find the value."):
    return Question(topic="t", level=0, seed=1, text=text, latex="", answer=kind_answer, answer_text=answer_text)


FRACTION = _q({"p": 3, "q": 13}, "3/13")
NUMBER = _q(3.5, "3.5")
WHOLE = _q(3, "3")
ORDERED = _q([1, 2, 3], "1, 2, 3")
EXPRESSION = _q(sympy.Symbol("x") ** 2, "x^2")


@pytest.mark.parametrize("reply", ["3/13", "3/13.", "= 3/13", "=3/13.", "  3/13 . "])
def test_fraction_accepts_a_sentence_end_and_a_leading_equals(reply):
    assert grade("fraction", FRACTION, reply).correct


@pytest.mark.parametrize("reply", ["3.5", "3.5.", "= 3.5", "=3.5."])
def test_a_trailing_full_stop_does_not_disturb_a_decimal(reply):
    assert grade("number", NUMBER, reply).correct


@pytest.mark.parametrize("reply", ["3", "3.", "= 3"])
def test_whole_number(reply):
    assert grade("number", WHOLE, reply).correct


@pytest.mark.parametrize("reply", ["1, 2, 3", "1, 2, 3.", "(1, 2, 3).", "= 1, 2, 3."])
def test_ordered_list(reply):
    assert grade("ordered_list", ORDERED, reply).correct


@pytest.mark.parametrize("reply", ["x^2", "x^2.", "= x^2", "= x^2."])
def test_expression(reply):
    assert grade("expression", EXPRESSION, reply).correct


@pytest.mark.parametrize("reply", ["3/14.", "= 3/14", "3.5 .5", "35.", "1, 2, 4."])
def test_a_wrong_value_is_still_wrong(reply):
    assert not grade("fraction", FRACTION, reply).correct
    assert not grade("number", NUMBER, reply).correct
    assert not grade("ordered_list", ORDERED, reply).correct


def test_only_one_full_stop_and_one_equals_are_dropped():
    assert not grade("fraction", FRACTION, "3/13..").correct
    assert not grade("fraction", FRACTION, "== 3/13").correct
