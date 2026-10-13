"""Tests the multi-part exam tier of `differentiate-trig`: a linear plus trigonometric function and its local maximum."""

from __future__ import annotations

import re

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
    parse_math,
    questions,
    skill,
    stem_function,
    value,
    x,
)

TOPIC = "exd_x_r2a_trig_turning"
SKILL_ID = "differentiate-trig"


def domain_end(q):
    return parse_math(re.search(r"≤ x ≤ (.+?)\.?$", q.stem.rstrip(".")).group(1))


def local_max(g, f, end):
    """The solution of g(x) = 0 in [0, end] where g changes from positive to negative (g' < 0), or None."""
    solutions = sympy.solveset(g, x, sympy.Interval(0, end))
    if not isinstance(solutions, sympy.FiniteSet):
        return None
    maxima = [s for s in solutions if sympy.diff(g, x).subs(x, s) < 0]
    return maxima[0] if len(maxima) == 1 else None


def close(left, right) -> bool:
    return abs(complex(sympy.N(sympy.sympify(left) - sympy.sympify(right), 30))) < 1e-9


def oracle(q):
    f, end = stem_function(q), domain_end(q)
    fp = sympy.diff(f, x)
    x_max = local_max(fp, f, end)
    return fp, x_max, f.subs(x, x_max)


def test_source_wired():
    check_wired(SKILL_ID, TOPIC)


def test_structure_and_tags():
    check_structure(TOPIC, ["expression", "expression", "expression"], [2, 2, 2], [False, True, True])


def test_prompt_shapes_and_new_shape_over_hard_tier():
    check_shapes(SKILL_ID, TOPIC)


def test_every_part_matches_an_independent_derivation_over_300_seeds():
    bad = []
    for q in questions(TOPIC, 300):
        fp, x_max, v_max = oracle(q)
        a, b, c = (p.answer_text for p in q.parts)
        if not (
            close(value(a).subs(x, sympy.Rational(3, 7)), fp.subs(x, sympy.Rational(3, 7)))
            and close(value(a).subs(x, 5), fp.subs(x, 5))
            and x_max is not None
            and close(value(b), x_max)
            and close(value(c), v_max)
        ):
            bad.append((q.text, q.answer_text))
    assert bad == [], bad[:3]


def test_key_accepted_and_every_distractor_rejected():
    check_graded(SKILL_ID, TOPIC, 120)


def test_follow_through_from_a_derivative_with_a_slip_in_the_coefficient_or_sign():
    """A wrong f'(x) that still has the right trigonometric function: the learner's own maximum and value earn marks."""
    sk = skill(SKILL_ID)
    used = 0
    for q in questions(TOPIC, 150):
        f, end = stem_function(q), domain_end(q)
        part_a = q.parts[0]
        for wrong in part_a.distractors:
            if part_a.option_notes.get(wrong) not in ("forgot-inner-derivative", "sign-error-trig-derivative"):
                continue
            x_max = local_max(value(wrong), f, end)
            if x_max is None:
                continue
            replies = {"a": wrong, "b": str(x_max), "c": str(f.subs(x, x_max))}
            result = grade(sk, q, replies)
            assert result.parts["a"].marks == 0
            for label in "bc":
                part = result.parts[label]
                assert part.marks == part.max_marks, (q.text, wrong, label, replies)
            used += 1
            break
    assert used >= 30, used


def test_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "x", "b": "123", "c": "-77"})


def test_answer_space():
    check_answer_space(TOPIC)


def test_text_is_clean_and_generation_is_deterministic():
    check_clean_and_deterministic(TOPIC)
