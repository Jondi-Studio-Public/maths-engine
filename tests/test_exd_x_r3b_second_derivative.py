"""Tests the multi-part exam tier of `second-derivative`: f'', its value at a point, a point of inflection."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade
from tests import exd_x_r3b_support as S
from tests.exd_x_c1_support import parse_math, same_function, x

TOPIC = "exd_x_r3b_second_derivative"
SKILL_ID = "second-derivative"
_SKILL = S.skill(SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return S.exam_questions(TOPIC, 300)


def _function(q):
    return parse_math(re.search(r"f\(x\) = (.+)\.$", q.stem).group(1))


def _inflection_points(f2):
    poly = sympy.Poly(sympy.expand(f2), x)
    return sorted(r for r, mult in sympy.roots(poly).items() if mult % 2 == 1)


def test_source_wired():
    S.check_wiring(TOPIC, SKILL_ID)


def test_prompt_variety_and_new_shape_over_hard_tier():
    S.check_shapes(TOPIC, SKILL_ID)


def test_structure():
    S.check_structure(TOPIC, "abc", [2, 1, 2])


def test_keys_match_independent_derivation_over_300_seeds(questions):
    for q in questions:
        f2 = sympy.diff(_function(q), x, 2)
        assert same_function(f2, parse_math(q.parts[0].answer_text)), q.stem
        x0 = int(re.search(r"f''\((-?\d+)\)", q.parts[1].prompt).group(1))
        assert q.parts[1].answer_text == str(f2.subs(x, x0)), q.stem
        points = _inflection_points(f2)
        expected = points[-1]
        assert Fraction(int(expected.p), int(expected.q)) == Fraction(q.parts[2].answer_text), q.stem
        assert ("greater" in q.parts[2].prompt) == (len(points) == 2), q.stem


def test_grading_accepts_key_and_rejects_every_distractor():
    S.check_grading(TOPIC, SKILL_ID, 120)


def test_distractors_are_tagged_and_formatted():
    S.check_tags(TOPIC)


def test_follow_through_from_a_wrong_second_derivative(questions):
    checked = 0
    for q in questions[:80]:
        wrong = q.parts[0].distractors[0]  # the first derivative, a polynomial of one degree higher
        x0 = int(re.search(r"f''\((-?\d+)\)", q.parts[1].prompt).group(1))
        value = parse_math(wrong).subs(x, x0)
        if value == 0 or str(value) == q.parts[1].answer_text:
            continue
        result = grade(_SKILL, q, {"a": wrong, "b": str(value), "c": q.parts[2].answer_text})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, q.stem
        assert result.parts["c"].marks == 2
        checked += 1
    assert checked >= 40


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "x^2", "b": "12345", "c": "999"})
        assert result.marks == 0 and not result.correct


def test_answer_space_over_300_seeds():
    S.check_answer_space(TOPIC)


def test_clean_and_deterministic():
    S.check_clean_and_deterministic(TOPIC)
    assert make(TOPIC, seed=7, difficulty="exam").to_dict() == make(TOPIC, seed=7, difficulty="exam").to_dict()
