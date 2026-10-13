"""Tests the multi-part exam tier of `differentiate-chain-rule`: f'(x), the gradient, the tangent, its x-intercept."""

from __future__ import annotations

import re

import sympy

from mathgen.engine import grade
from tests.exd_x_r2a_support import (
    check_answer_space,
    check_clean_and_deterministic,
    check_graded,
    check_shapes,
    check_structure,
    check_wired,
    frac,
    parse_math,
    questions,
    same_function,
    skill,
    value,
    x,
)

TOPIC = "exd_x_r2a_chain_tangent"
SKILL_ID = "differentiate-chain-rule"


def rendered(q):
    """The function and the point of contact, read back from the rendered stem alone."""
    f = parse_math(re.search(r"f\(x\) = (.+?)\.\s", q.stem).group(1))
    x0 = int(re.findall(r"x = (-?\d+)", q.stem)[-1])
    return f, x0


def oracle(q):
    f, x0 = rendered(q)
    fp = sympy.diff(f, x)
    m, f0 = fp.subs(x, x0), f.subs(x, x0)
    tangent = m * (x - x0) + f0
    return fp, m, tangent, x0 - f0 / m


def test_source_wired():
    check_wired(SKILL_ID, TOPIC)


def test_structure_and_tags():
    check_structure(
        TOPIC, ["expression", "fraction", "expression", "fraction"], [2, 1, 2, 1], [False, True, True, True]
    )


def test_prompt_shapes_and_new_shape_over_hard_tier():
    check_shapes(SKILL_ID, TOPIC)


def test_every_part_matches_an_independent_derivation_over_300_seeds():
    bad = []
    for q in questions(TOPIC, 300):
        fp, m, tangent, x_int = oracle(q)
        a, b, c, d = (p.answer_text for p in q.parts)
        if not (same_function(value(a), fp) and frac(b) == m and same_function(value(c), tangent) and frac(d) == x_int):
            bad.append((q.text, q.answer_text))
    assert bad == [], bad[:3]


def test_key_accepted_and_every_distractor_rejected():
    check_graded(SKILL_ID, TOPIC, 120)


def test_follow_through_from_a_wrong_derivative():
    sk = skill(SKILL_ID)
    for q in questions(TOPIC, 100):
        f, x0 = rendered(q)
        f0 = f.subs(x, x0)
        wrong_a = q.parts[0].distractors[0]
        m = value(wrong_a).subs(x, x0)
        if m == 0 or m == sympy.zoo or m == frac(q.parts[1].answer_text):
            continue
        line = m * (x - x0) + f0
        replies = {"a": wrong_a, "b": str(m), "c": "y = " + str(sympy.expand(line)), "d": str(x0 - f0 / m)}
        result = grade(sk, q, replies)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label, replies)
            assert result.parts[label].follow_through is True
        assert result.marks == 4 and not result.correct


def test_follow_through_from_a_wrong_gradient_only():
    sk = skill(SKILL_ID)
    for q in questions(TOPIC, 60):
        f, x0 = rendered(q)
        f0 = f.subs(x, x0)
        m = sympy.Rational(frac(q.parts[1].answer_text)) + 2
        if m == 0:
            continue
        line = m * (x - x0) + f0
        replies = {
            "a": q.parts[0].answer_text,
            "b": str(m),
            "c": "y = " + str(sympy.expand(line)),
            "d": str(x0 - f0 / m),
        }
        result = grade(sk, q, replies)
        assert result.parts["a"].marks == 2 and result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["d"].marks == 1


def test_all_wrong_earns_nothing():
    sk = skill(SKILL_ID)
    for q in questions(TOPIC, 30):
        result = grade(sk, q, {"a": "x", "b": "999", "c": "y = 5", "d": "123456"})
        assert result.marks == 0


def test_answer_space():
    check_answer_space(TOPIC)


def test_text_is_clean_and_generation_is_deterministic():
    check_clean_and_deterministic(TOPIC)
