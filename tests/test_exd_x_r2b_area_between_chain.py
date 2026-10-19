"""The multi-part exam tier `exd_x_r2b_area_between_chain` of definite-integral: the two intersections, the exact
area between a parabola and a line, and the average vertical distance.

The oracle reads the two curves out of the rendered stem and re-derives every key with SymPy (solve, then integrate
the larger curve minus the smaller); it never imports the generator module.
"""

from __future__ import annotations

import re

import sympy

import exd_x_r2b_support as sup
from mathgen.engine import grade

TOPIC = "exd_x_r2b_area_between_chain"
SKILL_ID = "definite-integral"
SEEDS = range(300)
x = sympy.symbols("x")


def read(q):
    p_text, l_text = re.search(r"y = (.+?) and the line y = (.+?)(?:\.| intersect)", q.stem).groups()
    return sup.parse_poly(p_text, x=x), sup.parse_poly(l_text, x=x)


def oracle(q):
    parabola, line = read(q)
    lo, hi = sorted(sympy.solve(sympy.Eq(parabola, line), x))
    mid = (lo + hi) / 2
    top, bottom = (line, parabola) if line.subs(x, mid) > parabola.subs(x, mid) else (parabola, line)
    area = sympy.integrate(top - bottom, (x, lo, hi))
    return lo, hi, area, area / (hi - lo)


def test_registered():
    sup.assert_registered(SKILL_ID, TOPIC)


def test_shape():
    sup.assert_chain_shape(TOPIC, ["fraction"] * 4, [1, 1, 3, 1], [0, 0, 1, 1])


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_prompt_shapes():
    sup.assert_shapes(SKILL_ID, TOPIC)


def test_key_matches_independent_derivation():
    for seed in SEEDS:
        q = sup.exam_question(TOPIC, seed)
        got = [sympy.Rational(p.answer_text) for p in q.parts]
        assert got == list(oracle(q)), (seed, q.stem)


def test_distractors_tagged():
    sup.assert_tagged(TOPIC)


def test_grading():
    sup.assert_grading(SKILL_ID, TOPIC)


def test_follow_through_from_a_wrong_limit():
    sk = sup.skill_object(SKILL_ID)
    checked = 0
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        parabola, line = read(q)
        lo, hi, _, _ = oracle(q)
        wrong_lo = lo - 1  # any wrong left-hand limit
        mid = (wrong_lo + hi) / 2
        top, bottom = (
            (line, parabola) if line.subs(x, (lo + hi) / 2) > parabola.subs(x, (lo + hi) / 2) else (parabola, line)
        )
        area = sympy.integrate(top - bottom, (x, wrong_lo, hi))
        if (
            wrong_lo == 0
            or area <= 0
            or area == sympy.Rational(q.parts[2].answer_text)
            or mid == 0
            or area == area / (hi - wrong_lo)
        ):
            continue
        mean = area / (hi - wrong_lo)
        if mean == sympy.Rational(q.parts[3].answer_text):
            continue
        checked += 1
        reply = {"a": str(wrong_lo), "b": str(hi), "c": str(area), "d": str(mean)}
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 1
        assert result.parts["c"].marks == 3 and result.parts["c"].follow_through, (seed, reply)
        assert result.parts["d"].marks == 1 and result.parts["d"].follow_through, (seed, reply)
    assert checked >= 30
