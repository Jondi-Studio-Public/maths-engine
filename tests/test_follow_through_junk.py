"""Pins that junk replies earn no follow-through marks and never make marking hang: a carried reply far from its
part's answer, the same number typed into two boxes, or a follow-through answer of 1, and two constants that
differ are told apart by evaluation rather than `simplify`."""

from __future__ import annotations

import time

import pytest
import sympy

from mathgen import engine
from mathgen.core import symbolically_equal
from mathgen.grading import typed

_SKILLS = {s.id: s for s in engine.load_skills()[1]}


def _exam(skill_id, seed=0):
    skill = _SKILLS[skill_id]
    return skill, engine.make_for_skill(skill, seed, source="exam")


def _timed_grade(skill, question, reply):
    start = time.perf_counter()
    result = engine.grade(skill, question, reply)
    return result, time.perf_counter() - start


@pytest.mark.parametrize(
    "skill_id, reply",
    [
        ("solve-ode-exponential", "100000000"),
        ("senior-use-partial-fractions-where-necessary-for-integration-in", "1000000"),
        ("ucv1-heat-equation-mode", "0.0001"),
    ],
)
def test_huge_or_tiny_replies_are_marked_quickly_and_earn_nothing(skill_id, reply):
    skill, question = _exam(skill_id)
    result, seconds = _timed_grade(skill, question, {p.label: reply for p in question.parts})
    assert seconds < 10, seconds
    assert result.marks == 0


def test_a_small_decimal_against_a_constant_expression_is_not_sent_to_simplify():
    _, question = _exam("ucv1-heat-equation-mode")
    part = next(p for p in question.parts if p.label == "b")
    start = time.perf_counter()
    assert not typed.grade(part.answer_kind, part, "0.0001").correct
    assert time.perf_counter() - start < 5
    assert typed.grade(part.answer_kind, part, part.answer_text).correct


def test_constants_that_differ_are_told_apart_and_equal_ones_still_match():
    assert not symbolically_equal(sympy.log(2) / sympy.pi**2, sympy.Rational(1, 10000))
    assert not symbolically_equal(sympy.pi, sympy.Rational(355, 113))
    assert symbolically_equal(sympy.sqrt(2) + sympy.sqrt(3), sympy.sqrt(5 + 2 * sympy.sqrt(6)))


@pytest.mark.parametrize("value", ["0", "1", "2", "-1", "1000000", "0.0001"])
@pytest.mark.parametrize(
    "skill_id",
    [
        "senior-use-partial-fractions-where-necessary-for-integration-in",
        "solve-ode-exponential",
        "ucv1-heat-equation-mode",
    ],
)
def test_the_same_junk_in_every_box_earns_nothing(skill_id, value):
    skill, question = _exam(skill_id)
    if any(p.answer_text == value for p in question.parts):
        pytest.skip("the junk value is one part's real answer")
    result, _ = _timed_grade(skill, question, {p.label: value for p in question.parts})
    assert result.marks == 0


def test_the_same_wrong_number_in_two_boxes_earns_no_difference_even_when_it_is_a_distractor():
    # (c) integrates from (a) to (b); "1" for both limits gives an integral of 0 with no working done.
    skill, question = _exam("definite-integral")
    result, _ = _timed_grade(skill, question, {"a": "1", "b": "1", "c": "0", "d": "1"})
    assert result.marks == 0


@pytest.mark.parametrize("skill_id", ["ucv1-double-integral-rectangle", "modular-multiplicative-inverse"])
def test_no_two_parts_of_these_chains_share_an_answer(skill_id):
    skill = _SKILLS[skill_id]
    for seed in range(40):
        question = engine.make_for_skill(skill, seed, source="exam")
        answers = [p.answer_text for p in question.parts]
        assert len(set(answers)) == len(answers), (seed, answers)
        result, _ = _timed_grade(skill, question, {p.label: "1" for p in question.parts})
        assert all(r.marks == 0 for label, r in result.parts.items() if _part(question, label).answer_text != "1"), seed


def _part(question, label):
    return next(p for p in question.parts if p.label == label)
