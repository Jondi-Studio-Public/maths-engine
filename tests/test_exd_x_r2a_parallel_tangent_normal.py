"""Tests the multi-part exam tier of `senior-find-the-slope-of-a-tangent-and-the`: a tangent parallel to a line."""

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
    frac,
    parse_math,
    questions,
    rational_text,
    same_function,
    skill,
    stem_function,
    value,
    x,
)

TOPIC = "exd_x_r2a_parallel_tangent_normal"
SKILL_ID = "senior-find-the-slope-of-a-tangent-and-the"


def read_stem(q):
    """The cubic, the gradient of the line it is parallel to, and the sign asked for, from the rendered stem."""
    f = stem_function(q)
    line = parse_math(re.search(r"parallel to (?:the line )?y = (.+?)\.$", q.stem).group(1))
    positive = "positive" in q.stem
    return f, sympy.Poly(line, x).coeff_monomial(x), positive


def oracle(q):
    f, m, positive = read_stem(q)
    roots = [r for r in sympy.solve(sympy.diff(f, x) - m, x) if (r > 0) == positive]
    assert len(roots) == 1, (q.text, roots)
    target = roots[0]
    tangent = m * (x - target) + f.subs(x, target)
    normal = -1 / m * (x - target) + f.subs(x, target)
    return target, tangent, normal, normal.subs(x, 0)


def test_structure_and_tags():
    check_structure(
        TOPIC, ["fraction", "expression", "expression", "fraction"], [2, 2, 2, 1], [False, True, True, True]
    )


def test_prompt_shapes_and_new_shape_over_hard_tier():
    check_shapes(SKILL_ID, TOPIC)


def test_every_part_matches_an_independent_derivation_over_300_seeds():
    bad = []
    for q in questions(TOPIC, 300):
        target, tangent, normal, y_int = oracle(q)
        a, b, c, d = (p.answer_text for p in q.parts)
        if not (
            frac(a) == target
            and same_function(value(b), tangent)
            and same_function(value(c), normal)
            and frac(d) == y_int
        ):
            bad.append((q.text, q.answer_text))
    assert bad == [], bad[:3]


def test_key_accepted_and_every_distractor_rejected():
    check_graded(SKILL_ID, TOPIC, 120)


def test_follow_through_from_a_wrong_contact_point():
    """A wrong x-coordinate for P, with the tangent, the normal and its intercept worked out correctly from it."""
    sk = skill(SKILL_ID)
    used = 0
    for q in questions(TOPIC, 100):
        f, m, _ = read_stem(q)
        for wrong in q.parts[0].distractors:
            a = sympy.Rational(frac(wrong))
            if a == 0:
                continue
            f_a = f.subs(x, a)
            tangent = m * (x - a) + f_a
            normal = -1 / m * (x - a) + f_a
            replies = {
                "a": wrong,
                "b": "y = " + str(sympy.expand(tangent)),
                "c": "y = " + str(sympy.expand(normal)),
                "d": rational_text(normal.subs(x, 0)),
            }
            result = grade(sk, q, replies)
            assert result.parts["a"].marks == 0
            for label in "bcd":
                part = result.parts[label]
                assert part.marks == part.max_marks and part.follow_through is True, (q.text, wrong, label, replies)
            assert result.marks == 5 and not result.correct
            used += 1
            break
    assert used >= 60, used


def test_all_wrong_earns_nothing():
    check_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "999", "b": "y = 5", "c": "y = 7", "d": "-77"})


def test_text_is_clean_and_generation_is_deterministic():
    check_clean_and_deterministic(TOPIC)
