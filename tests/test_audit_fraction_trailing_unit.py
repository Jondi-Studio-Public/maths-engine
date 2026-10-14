"""An exact-fraction answer may carry a unit, as a typed number may; the unit must still be an allowed one."""

from __future__ import annotations

import pytest

from mathgen.core import Question
from mathgen.grading import grade


def _q(text, p, q, answer_text):
    return Question(topic="t", level=0, seed=1, text=text, latex="", answer={"p": p, "q": q}, answer_text=answer_text)


@pytest.mark.parametrize(
    "reply", ["1185/8", "1185/8 m^2", "1185/8 m²", "148.125 m^2", "148.125m²", "1185/8 square metres", "1185/8 sq m"]
)
def test_fraction_with_area_unit_is_accepted(reply):
    question = _q("Find the area in m^2.", 1185, 8, "1185/8")
    assert grade("fraction", question, reply).correct


@pytest.mark.parametrize("reply", ["109.5 m/s", "219/2 m/s", "109.5 metres per second", "109.5m/s"])
def test_fraction_with_speed_unit_is_accepted(reply):
    assert grade("fraction", _q("Find the speed.", 219, 2, "219/2"), reply).correct


@pytest.mark.parametrize("reply", ["71", "$71", "71 dollars", "71 $", "$71.00"])
def test_whole_dollar_answer_held_as_a_fraction(reply):
    assert grade("fraction", _q("What is the minimum cost?", 71, 1, "71"), reply).correct


@pytest.mark.parametrize("reply", ["7.2%", "7.2 %", "36/5 percent"])
def test_an_unasked_percent_is_refused_as_grade_number_refuses_it(reply):
    assert not grade("fraction", _q("Find the error.", 36, 5, "36/5"), reply).correct


@pytest.mark.parametrize("reply", ["1185/8 kg", "148.125 cm", "1185/8 s", "1185/8 dollars"])
def test_a_unit_the_question_rules_out_is_rejected(reply):
    question = _q("Find the area in m^2.", 1185, 8, "1185/8")
    assert not grade("fraction", question, reply).correct


@pytest.mark.parametrize("reply", ["1186/8 m^2", "148 m^2", "1/8 m", "m^2", "5 m 7 m"])
def test_a_wrong_value_with_a_unit_is_still_wrong(reply):
    assert not grade("fraction", _q("Find the area in m^2.", 1185, 8, "1185/8"), reply).correct
