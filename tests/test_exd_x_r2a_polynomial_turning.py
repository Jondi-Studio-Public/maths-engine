"""Tests the multi-part exam tier of `differentiate-polynomial`: a cubic given as a product, its turning points."""

from __future__ import annotations

import sympy

from mathgen.engine import grade
from tests.exd_x_r2a_support import (
    check_all_wrong_earns_nothing,
    check_answer_space,
    check_clean_and_deterministic,
    check_graded,
    check_shapes,
    check_structure,
    check_wired,
    frac,
    questions,
    rational_text,
    same_function,
    skill,
    stem_function,
    value,
    x,
)

TOPIC = "exd_x_r2a_polynomial_turning"
SKILL_ID = "differentiate-polynomial"


def oracle(q):
    f = stem_function(q)
    fp, f2 = sympy.diff(f, x), sympy.diff(f, x, 2)
    roots = sympy.solve(fp, x)
    x_max = next(r for r in roots if f2.subs(x, r) < 0)
    inflection = sympy.solve(f2, x)[0]
    return fp, x_max, f.subs(x, x_max), inflection


def test_structure_and_tags():
    check_structure(TOPIC, ["expression", "fraction", "fraction", "fraction"], [2, 2, 1, 1], [False, True, True, True])


def test_prompt_shapes_and_new_shape_over_hard_tier():
    check_shapes(SKILL_ID, TOPIC)


def test_every_part_matches_an_independent_derivation_over_300_seeds():
    bad = []
    for q in questions(TOPIC, 300):
        fp, x_max, v_max, inflection = oracle(q)
        a, b, c, d = (p.answer_text for p in q.parts)
        if not (same_function(value(a), fp) and frac(b) == x_max and frac(c) == v_max and frac(d) == inflection):
            bad.append((q.text, q.answer_text))
    assert bad == [], bad[:3]


def test_key_accepted_and_every_distractor_rejected():
    check_graded(SKILL_ID, TOPIC, 120)


def test_follow_through_from_a_wrong_quadratic_derivative():
    """A wrong f'(x) that is still a quadratic: the learner's own maximum, its value and the inflection earn the marks."""
    sk = skill(SKILL_ID)
    used = 0
    for q in questions(TOPIC, 150):
        f = stem_function(q)
        for wrong in q.parts[0].distractors:
            g = sympy.Poly(value(wrong), x)
            if g.degree() != 2:
                continue
            a_, b_, c_ = g.all_coeffs()
            disc = b_**2 - 4 * a_ * c_
            if disc < 0 or not sympy.sqrt(disc).is_Rational:
                continue
            x_max = (-b_ - sympy.sqrt(disc)) / (2 * a_)
            replies = {
                "a": wrong,
                "b": rational_text(x_max),
                "c": rational_text(f.subs(x, x_max)),
                "d": rational_text(-b_ / (2 * a_)),
            }
            result = grade(sk, q, replies)
            assert result.parts["a"].marks == 0
            for label in "bcd":
                assert result.parts[label].marks == result.parts[label].max_marks, (q.text, wrong, label, replies)
            used += 1
            break
    assert used >= 15, used


def test_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "x", "b": "999", "c": "123", "d": "-77"})


def test_text_is_clean_and_generation_is_deterministic():
    check_clean_and_deterministic(TOPIC)
