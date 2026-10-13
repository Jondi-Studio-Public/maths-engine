"""Tests the multi-part exam tier of `stationary-points`: a cubic with a repeated root, its second stationary point."""

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

TOPIC = "exd_x_r2a_stationary_double_root"
SKILL_ID = "stationary-points"


def oracle(q):
    f = stem_function(q)
    fp = sympy.diff(f, x)
    x0 = next(r for r in sympy.solve(fp, x) if f.subs(x, r) != 0)
    return fp, x0, sympy.diff(f, x, 2).subs(x, x0), f.subs(x, x0)


def test_source_wired():
    check_wired(SKILL_ID, TOPIC)


def test_structure_and_tags():
    check_structure(TOPIC, ["expression", "fraction", "fraction", "fraction"], [2, 2, 1, 1], [False, True, True, True])


def test_prompt_shapes_and_new_shape_over_hard_tier():
    check_shapes(SKILL_ID, TOPIC)


def test_every_part_matches_an_independent_derivation_over_300_seeds():
    bad = []
    for q in questions(TOPIC, 300):
        fp, x0, f2, v = oracle(q)
        a, b, c, d = (p.answer_text for p in q.parts)
        if not (same_function(value(a), fp) and frac(b) == x0 and frac(c) == f2 and frac(d) == v):
            bad.append((q.text, q.answer_text))
    assert bad == [], bad[:3]


def test_key_accepted_and_every_distractor_rejected():
    check_graded(SKILL_ID, TOPIC, 120)


def test_follow_through_from_a_wrong_quadratic_derivative():
    """A wrong f'(x) that is still a quadratic: the learner's other root, f'' there and f there earn the marks."""
    sk = skill(SKILL_ID)
    used = 0
    for q in questions(TOPIC, 150):
        f = stem_function(q)
        p = next(r for r in sympy.solve(sympy.diff(f, x), x) if f.subs(x, r) == 0)
        for wrong in q.parts[0].distractors:
            g = sympy.Poly(value(wrong), x)
            if g.degree() != 2:
                continue
            roots = sympy.roots(g)
            if p not in roots or any(not r.is_Rational for r in roots):
                continue
            others = [r for r in roots if r != p]
            x_b = others[0] if others else p
            if 0 in (x_b, sympy.diff(g.as_expr(), x).subs(x, x_b)):
                continue  # the grader does not carry a stated zero forward as follow-through
            replies = {
                "a": wrong,
                "b": rational_text(x_b),
                "c": rational_text(sympy.diff(g.as_expr(), x).subs(x, x_b)),
                "d": rational_text(f.subs(x, x_b)),
            }
            result = grade(sk, q, replies)
            assert result.parts["a"].marks == 0
            # every wrong quadratic here keeps f's root, so the learner's value of f is a stated zero: not carried
            for label in "bc":
                part = result.parts[label]
                assert part.marks == part.max_marks, (q.text, wrong, label, replies)
            used += 1
            break
    assert used >= 15, used


def test_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "x", "b": "999", "c": "123", "d": "-77"})


def test_answer_space():
    check_answer_space(TOPIC)


def test_text_is_clean_and_generation_is_deterministic():
    check_clean_and_deterministic(TOPIC)
