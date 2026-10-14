"""A reply that denies or doubts its own value is wrong in every kind but inequality and matrix: "not 12",
"12 is wrong", "no 12", "maybe 12". A word the answer or an option uses itself is not a hedge there."""

from __future__ import annotations

import pytest

from mathgen import engine
from mathgen.core import Question
from mathgen.grading import grade

_SKILLS = {s.id: s for s in engine.load_skills()[1]}


@pytest.mark.parametrize(
    ("skill_id", "seed"),
    [
        ("differentiate-trig", 1),
        ("simple-event-probability", 1),
        ("ucv1-fundamental-group-map", 1),
        ("inequalities-negative-mult-div", 0),
    ],
)
@pytest.mark.parametrize(
    "wrap", ["not {}", "{} is wrong", "{} (wrong)", "{} is incorrect", "isn't {}", "maybe {}", "probably {}"]
)
def test_a_hedged_reply_is_wrong(skill_id, seed, wrap):
    sk = _SKILLS[skill_id]
    q = engine.make_for_skill(sk, seed, source=0)
    assert engine.grade(sk, q, q.answer_text).correct
    if sk.answer == "inequality":
        return  # inequality reads its own words; it is not checked here
    assert not engine.grade(sk, q, wrap.format(q.answer_text)).correct


def _number(answer, text="Find the value."):
    return Question(topic="t", level=0, seed=1, text=text, latex="", answer=answer, answer_text=str(answer))


@pytest.mark.parametrize("reply", ["no 9", "no -9", "not 9", "9 is wrong", "9, maybe"])
def test_hedged_numbers_are_wrong(reply):
    assert not grade("number", _number(9 if "-" not in reply else -9), reply).correct


@pytest.mark.parametrize("reply", ["9", "no, it's 9", "9 (no units)", "No. of sides = 9", "9?"])
def test_no_before_words_or_a_question_mark_is_not_a_hedge(reply):
    assert grade("number", _number(9), reply).correct


def test_a_word_the_answer_uses_is_not_a_hedge():
    q = Question(
        topic="t",
        level=0,
        seed=1,
        text="Is it suitable?",
        latex="",
        answer="Not suitable",
        answer_text="Not suitable",
        distractors=["Suitable", "Cannot tell"],
    )
    assert grade("mcq", q, "Not suitable").correct


def _midpoint():
    return Question(
        topic="t",
        level=0,
        seed=1,
        text="Find the midpoint M of A(1, 2) and B(3, 4).",
        latex="",
        answer={"x": 2, "y": 3},
        answer_text="(2, 3)",
    )


@pytest.mark.parametrize("reply", ["M(2, 3)", "M (2, 3)", "(2, 3)", "M = (2, 3)"])
def test_a_point_written_with_the_name_the_question_gives_it_is_correct(reply):
    assert grade("pair", _midpoint(), reply).correct


@pytest.mark.parametrize("reply", ["f(2, 3)", "N(2, 3)", "sqrt((2, 3))", "(2, 3)!"])
def test_a_function_or_an_unnamed_point_around_a_pair_is_wrong(reply):
    assert not grade("pair", _midpoint(), reply).correct
