"""A multiple-choice reply scores only when it is exactly one option: its index, its letter or its text.

The grader used to take the first integer anywhere in the reply, so "not 1", "1 or 2", "1, actually 2", "x=1" and
"1.5" all scored for an answer at index 1. Now anything that is not a single option is wrong.
"""

from __future__ import annotations

import pytest

from mathgen import engine

SKILL = next(s for s in engine.load_skills()[1] if s.id == "ucv1-fundamental-group-map")
QUESTION = engine.make_for_skill(SKILL, 1)  # the correct option is at index 1, "multiplication by -4"


def _correct(reply) -> bool:
    return engine.grade(SKILL, QUESTION, reply).correct


@pytest.mark.parametrize(
    "reply",
    ["1", " 1 ", "1.", 1, "B", "b", "(b)", "B.", "b)", "multiplication by -4", " Multiplication  BY -4. "],
)
def test_one_option_is_correct(reply):
    assert _correct(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "not 1",
        "1 or 2",
        "1, actually 2",
        "x=1",
        "1.5",
        "option 1",
        "-1",
        "0",
        "2",
        "4",
        "01 2",
        "1 2",
        "",
        "a",
        "(b",
        "bb",
    ],
)
def test_anything_else_is_wrong(reply):
    assert not _correct(reply)


def test_adversarial_reply_is_graded_quickly():
    assert not _correct("(a" * 50_000)


def test_numeric_option_text_wins_over_index():
    skill = next(s for s in engine.load_skills()[1] if s.id == "count-to-10")
    question = engine.make_for_skill(skill, 2)
    from mathgen.core import mcq_options

    options = mcq_options(question)
    key = options.index(question.answer_text)
    wrong = next(i for i in range(len(options)) if i != key)
    assert engine.grade(skill, question, question.answer_text).correct
    assert not engine.grade(skill, question, options[wrong]).correct
    assert engine.grade(skill, question, key).correct  # an int is always the 0-based index
