"""Three marking rulings: thousands grouped by a space and "a × 10^n" are one number, and a vector's components
are marked in order."""

from __future__ import annotations

import pytest

from mathgen import engine
from mathgen.core import Question
from mathgen.grading import grade


def _q(answer, answer_text=None, text="q"):
    return Question(
        topic="test", level=0, seed=1, text=text, latex="", answer=answer, answer_text=answer_text or str(answer)
    )


# -- "12 345" is twelve thousand three hundred and forty-five ----------------------------


@pytest.mark.parametrize(
    "answer, reply",
    [
        (12345, "12 345"),
        (12345, "12 345"),  # thin space
        (12345, "12 345"),  # no-break space
        (1234567, "1 234 567"),
        (1234567.5, "1 234 567.5"),
        (-12345, "-12 345"),
        (12345, "$12 345"),
        (12345, "12 345 m"),
        (12345, "x = 12 345"),
    ],
)
def test_space_grouped_thousands_are_one_number(answer, reply):
    assert grade("number", _q(answer), reply).correct


@pytest.mark.parametrize(
    "answer, reply",
    [
        (12, "12 345"),
        (345, "12 345"),
        (345, "3 45"),
        (123456, "12 3456"),
        (9.5, "9 1/2"),
        (12345, "12 34 5"),
    ],
)
def test_space_that_does_not_group_thousands_still_splits_values(answer, reply):
    assert not grade("number", _q(answer), reply).correct


# -- "1.2 × 10^3" is 1200 -----------------------------------------------------------------


@pytest.mark.parametrize(
    "answer, reply",
    [
        (1200, "1.2 × 10^3"),
        (1200, "1.2 x 10^3"),
        (1200, "1.2 X 10^3"),
        (1200, "1.2*10**3"),
        (1200, "1.2×10³"),
        (1200, "12 × 10^2"),
        (1000000, "1 × 10^6"),
        (0.0012, "1.2 × 10^-3"),
        (0.0012, "1.2 × 10^−3"),
        (0.0012, "1.2 × 10⁻³"),
        (0.0012, "1.2 × 10^(-3)"),
        (0.0012, "1.2 × 10^{-3}"),
        (-1200, "-1.2 × 10^3"),
        (1200, "x = 1.2 × 10^3"),
    ],
)
def test_times_ten_to_a_power_is_read(answer, reply):
    assert grade("number", _q(answer), reply).correct


@pytest.mark.parametrize(
    "answer, reply",
    [
        (1000, "1e3"),  # calculator notation is not what is taught
        (1200, "1.2e3"),
        (1200, "1.2 × 10^3 × 2"),
        (1200, "1.2 × 10^4"),
        (12000, "1.2 × 10^3"),
        (1200, "1.2 × 10^999"),  # the power is at most two digits
        (6, "2 x 3"),
    ],
)
def test_other_forms_and_wrong_powers_are_wrong(answer, reply):
    assert not grade("number", _q(answer), reply).correct


def test_times_ten_reply_is_checked_for_the_rounding_asked():
    question = _q(0.33, "0.33", text="Find p to 2 decimal places.")
    assert grade("number", question, "3.3 × 10^-1").correct
    assert not grade("number", question, "3.333 × 10^-1").correct  # 0.3333 has not been rounded


# -- a vector's components are in order ---------------------------------------------------


@pytest.mark.parametrize(
    "reply, correct",
    [
        ("(5, 6, -5)", True),
        ("5, 6, -5", True),
        ("[5, 6, -5]", True),
        ("<5, 6, -5>", True),
        ("⟨5, 6, −5⟩", True),
        ("5i + 6j - 5k", True),
        ("6j + 5i - 5k", True),  # component form names each component, so its terms may come in any order
        ("(-5, 6, 5)", False),
        ("(6, 5, -5)", False),
        ("(5, 6)", False),
        ("not (5, 6, -5)", False),
    ],
)
def test_vector_components_are_marked_in_order(reply, correct):
    question = _q([5, 6, -5], "(5, 6, -5)", text="Find u + v")
    assert grade("ordered_list", question, reply).correct is correct


@pytest.mark.parametrize("skill_id", ["vector-addition", "vector-cross-product"])
def test_vector_skills_reject_a_reversed_vector(skill_id):
    skill = {s.id: s for s in engine.load_skills()[1]}[skill_id]
    for seed in range(10):
        question = engine.make_for_skill(skill, seed=seed)
        components = list(question.answer)
        assert engine.grade(skill, question, ", ".join(map(str, components))).correct
        if components != components[::-1]:
            assert not engine.grade(skill, question, ", ".join(map(str, components[::-1]))).correct
