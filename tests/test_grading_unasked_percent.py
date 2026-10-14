"""A "%" or "percent" after a value is refused unless the question is about percent, in the fraction kind as in number."""

from __future__ import annotations

import pytest

from mathgen.core import Question
from mathgen.grading import grade


def _q(answer, answer_text, text):
    return Question(topic="t", level=0, seed=1, text=text, latex="", answer=answer, answer_text=answer_text)


@pytest.mark.parametrize("reply", ["3/4%", "0.75%", "3/4 percent", "3/4 per cent"])
def test_fraction_refuses_an_unasked_percent(reply):
    assert not grade("fraction", _q({"p": 3, "q": 4}, "3/4", "Find the value."), reply).correct


@pytest.mark.parametrize("reply", ["0.27%", "0.27 percent"])
def test_number_refuses_an_unasked_percent(reply):
    assert not grade("number", _q(0.27, "0.27", "Find the probability."), reply).correct


def test_a_plain_reply_is_still_right():
    assert grade("fraction", _q({"p": 3, "q": 4}, "3/4", "Find the value."), "3/4").correct
    assert grade("number", _q(0.27, "0.27", "Find the probability."), "0.27").correct


def test_percent_is_fine_when_the_question_mentions_it():
    assert grade("fraction", _q({"p": 3, "q": 4}, "3/4", "What fraction is 75 percent?"), "3/4%").correct
    assert grade("number", _q(27, "27%", "Find the rate."), "27%").correct
    assert grade("number", _q(27, "27", "What percentage is it?"), "27 percent").correct
