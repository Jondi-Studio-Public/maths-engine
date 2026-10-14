"""Exam source of `lagrange-constrained-point`: a weighted objective under a linear constraint, in four parts."""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from tests.exd_x_u5_support import (
    as_fraction,
    canonical_reply,
    check_key_and_distractors,
    check_no_text_defects,
    check_shapes,
    exam_question,
    exam_sources,
    parse_expression,
    rational,
    skill_of,
    survives_retarget,
)

from mathgen.engine import grade

SKILL = "lagrange-constrained-point"
x, y, lam = sympy.symbols("x y lam")


def read(q):
    """(weight k, constraint expression g, constant c) parsed from the rendered stem alone."""
    stem = q.text.split(" (a)")[0]
    k = int(re.search(r"x² \+ (\d+)y²", stem).group(1))
    g_text = re.search(r"g\(x, y\) = ([^.,]+?)(?= and use|[.,])", stem).group(1)
    g = parse_expression(g_text, x=x, y=y)
    c = int(re.search(re.escape(g_text) + r" = (-?\d+)", stem).group(1))
    return k, g, c


def oracle(q):
    k, g, c = read(q)
    f = x**2 + k * y**2
    solution = sympy.solve(
        [sympy.diff(f, x) - lam * sympy.diff(g, x), sympy.diff(f, y) - lam * sympy.diff(g, y), g - c],
        [x, y, lam],
        dict=True,
    )
    assert len(solution) == 1
    s = solution[0]
    return {"a": rational(s[lam]), "b": rational(s[x]), "c": rational(s[y]), "d": rational(f.subs(s))}


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam_question(SKILL, seed)
        got = {p.label: as_fraction(p.answer_text) for p in q.parts}
        assert got == oracle(q), (seed, q.text)


def test_grader_accepts_key_rejects_distractors():
    check_key_and_distractors(SKILL)


def test_follow_through_from_wrong_multiplier():
    skill = skill_of(SKILL)
    checked = 0
    for seed in range(60):
        q = exam_question(SKILL, seed)
        k, g, c = read(q)
        a_coef, b_coef = (rational(g.coeff(v)) for v in (x, y))
        wrong = as_fraction(q.parts[0].distractors[0])
        xs, ys = wrong * a_coef / 2, wrong * b_coef / (2 * k)
        fs = xs * xs + k * ys * ys
        if 0 in (wrong, xs, ys, fs) or len({wrong, xs, ys, fs}) < 4 or not survives_retarget(xs, ys, fs):
            continue
        checked += 1
        reply = {"a": q.parts[0].distractors[0], "b": str(xs), "c": str(ys), "d": str(fs)}
        result = grade(skill, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == result.parts[label].max_marks, (seed, reply)
        assert result.marks == 4 and result.max_marks == 6
    assert checked >= 20


def test_follow_through_from_wrong_coordinates():
    skill = skill_of(SKILL)
    for seed in range(40):
        q = exam_question(SKILL, seed)
        k, _, _ = read(q)
        xs = as_fraction(q.parts[1].distractors[0])
        ys = as_fraction(q.parts[2].distractors[0])
        fs = xs * xs + k * ys * ys
        if 0 in (xs, ys, fs) or len({xs, ys, fs}) < 3 or not survives_retarget(fs):
            continue
        reply = {**canonical_reply(q), "b": q.parts[1].distractors[0], "c": q.parts[2].distractors[0], "d": str(fs)}
        result = grade(skill, q, reply)
        assert result.parts["d"].marks == result.parts["d"].max_marks, (seed, reply)
        assert result.parts["b"].marks == 0 and result.parts["c"].marks == 0


def test_all_wrong_earns_nothing():
    q = exam_question(SKILL, 5)
    assert grade(skill_of(SKILL), q, {"a": "999", "b": "998", "c": "997", "d": "996"}).marks == 0


def test_chain_is_not_copyable():
    """No part repeats an earlier answer or follows through as a restatement, and junk replies earn nothing."""
    from tests.exd_junk_probe import assert_chain_is_not_copyable

    assert_chain_is_not_copyable("exd_x_u5_lagrange_weighted", "lagrange-constrained-point")
