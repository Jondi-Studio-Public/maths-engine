"""Reply forms a learner or an app sends that used to raise or be refused: a JSON number where text was expected,
the curly apostrophe phone and Mac keyboards type for a prime, a label copied from the question, and a stem
that says its expression is "not yet simplified"."""

from __future__ import annotations

import pytest

from mathgen import engine
from mathgen.core import Question
from mathgen.grading import grade

_SKILLS = {s.id: s for s in engine.load_skills()[1]}


@pytest.mark.parametrize("skill_id", ["inequalities-negative-mult-div", "ucv1-diagonalisation"])
@pytest.mark.parametrize("reply", [5, 2.5, True, [1, 2], None])
def test_a_reply_that_is_not_text_is_marked_wrong_without_raising(skill_id, reply):
    sk = _SKILLS[skill_id]
    q = engine.make_for_skill(sk, 0)
    assert not engine.grade(sk, q, reply).correct


@pytest.mark.parametrize("label", ["f’(x) = ", "y’ = ", "y’’ = "])
def test_a_curly_apostrophe_prime_label_is_read_like_a_straight_one(label):
    sk = _SKILLS["differentiate-trig"]
    q = engine.make_for_skill(sk, 1)
    assert engine.grade(sk, q, label + q.answer_text).correct
    assert engine.grade(sk, q, label.replace("’", "'") + q.answer_text).correct


def test_a_label_written_as_the_question_writes_it_is_dropped():
    sk = _SKILLS["simple-event-probability"]
    q = engine.make_for_skill(sk, 1, source=0)
    assert "P(red)" in q.text
    answer = q.answer_text
    assert engine.grade(sk, q, f"P(red) = {answer}").correct
    assert engine.grade(sk, q, f"P( red ) = {answer}").correct
    # A label the question never writes is still not dropped, and the value is still checked.
    assert not engine.grade(sk, q, f"P(blue) = {answer}").correct
    assert not engine.grade(sk, q, f"x = {answer}").correct
    assert not engine.grade(sk, q, "P(red) = 99/100").correct


def _frac(text):
    return Question(topic="t", level=0, seed=1, text=text, latex="", answer={"p": 3, "q": 4}, answer_text="3/4")


@pytest.mark.parametrize(
    "text",
    [
        "The function f(x) = 6/8 + x (not yet simplified) has one root. Find it.",
        "Do not simplify. Find the probability.",
        "Find the probability; don't simplify the working.",
    ],
)
def test_negated_simplify_wording_does_not_ask_for_lowest_terms(text):
    assert grade("fraction", _frac(text), "0.75").correct
    assert grade("fraction", _frac(text), "6/8").correct


def test_simplify_still_asks_for_lowest_terms():
    assert not grade("fraction", _frac("Simplify 6/8."), "6/8").correct
    assert not grade("fraction", _frac("Write the answer as a fully simplified fraction."), "0.75").correct
