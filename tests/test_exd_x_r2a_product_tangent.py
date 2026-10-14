"""Tests the multi-part exam tier of `differentiate-product-rule`: f'(x), the gradient, the tangent, the triangle."""

from __future__ import annotations

import sympy

from tests.exd_x_r2a_support import (
    check_all_wrong_earns_nothing,
    check_answer_space,
    check_clean_and_deterministic,
    check_follow_through_chain,
    check_graded,
    check_shapes,
    check_structure,
    check_wired,
    frac,
    questions,
    same_function,
    stem_function,
    stem_point,
    value,
    x,
)

TOPIC = "exd_x_r2a_product_tangent"
SKILL_ID = "differentiate-product-rule"


def oracle(q):
    f, x0 = stem_function(q), stem_point(q)
    fp = sympy.diff(f, x)
    m, f0 = fp.subs(x, x0), f.subs(x, x0)
    tangent = m * (x - x0) + f0
    c0 = tangent.subs(x, 0)
    return fp, m, tangent, c0**2 / (2 * abs(m))


def test_structure_and_tags():
    check_structure(
        TOPIC, ["expression", "fraction", "expression", "fraction"], [2, 1, 2, 2], [False, True, True, True]
    )


def test_prompt_shapes_and_new_shape_over_hard_tier():
    check_shapes(SKILL_ID, TOPIC)


def test_every_part_matches_an_independent_derivation_over_300_seeds():
    bad = []
    for q in questions(TOPIC, 300):
        fp, m, tangent, area = oracle(q)
        a, b, c, d = (p.answer_text for p in q.parts)
        if not (same_function(value(a), fp) and frac(b) == m and same_function(value(c), tangent) and frac(d) == area):
            bad.append((q.text, q.answer_text))
    assert bad == [], bad[:3]


def test_key_accepted_and_every_distractor_rejected():
    check_graded(SKILL_ID, TOPIC, 120)


def test_follow_through_from_a_wrong_derivative():
    check_follow_through_chain(SKILL_ID, TOPIC, "area")


def test_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "x", "b": "999", "c": "y = 5", "d": "123456"})


def test_text_is_clean_and_generation_is_deterministic():
    check_clean_and_deterministic(TOPIC)
