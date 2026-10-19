"""An interval of convergence written as |x - a| < r is the interval a - r < x < a + r."""

from __future__ import annotations

import pytest

from mathgen.core import Question
from mathgen.grading import grade
from mathgen.grading.inequality import read_solution_set


def _q(answer):
    return Question(topic="t", level=0, seed=1, text="Find the interval.", latex="", answer=answer, answer_text=answer)


@pytest.mark.parametrize("reply", ["|x| < 3", "abs(x) < 3", "|x|<3", "3 > |x|", "-3 < x < 3", "(-3, 3)", "|x| < 9/3"])
def test_open_interval_about_zero(reply):
    assert grade("inequality", _q("-3 < x < 3"), reply).correct


@pytest.mark.parametrize("reply", ["|x-1| < 4", "|x - 1|<4", "abs(x-1) < 4", "|x−1| < 4", "(-3, 5)"])
def test_open_interval_about_one(reply):
    assert grade("inequality", _q("-3 < x < 5"), reply).correct


def test_closed_version_with_less_or_equal():
    assert grade("inequality", _q("-3 <= x <= 5"), "|x - 1| <= 4").correct
    assert not grade("inequality", _q("-3 < x < 5"), "|x - 1| <= 4").correct
    assert not grade("inequality", _q("-3 <= x <= 5"), "|x - 1| < 4").correct


def test_centre_with_a_plus_sign():
    assert grade("inequality", _q("-3 < x < -1"), "|x + 2| < 1").correct


@pytest.mark.parametrize("reply", ["|x| < 4", "|x - 1| < 3", "|x + 1| < 4", "|y| < 3"])
def test_wrong_interval_or_variable_is_wrong(reply):
    assert not grade("inequality", _q("-3 < x < 3"), reply).correct


@pytest.mark.parametrize("reply", ["|x| < 0", "|x| < -2", "|x| < y", "|2x| < 3", "|x| < ", "|x|"])
def test_radius_must_be_a_positive_number(reply):
    assert read_solution_set(reply) is None
