"""The multi-part exam tier `exd_x_r2b_leibniz_chain` of senior-use-the-leibniz-notation-for-the-derivative-frac:
dy/du, du/dx, dy/dx by the chain rule and dy/dt for a given dx/dt.

The oracle reads the composite and the instant out of the rendered text and re-derives every key by differentiating
the composite with SymPy; it never imports the generator module.
"""

from __future__ import annotations

import re

import sympy

import exd_x_r2b_support as sup
from mathgen.engine import grade

TOPIC = "exd_x_r2b_leibniz_chain"
SKILL_ID = "senior-use-the-leibniz-notation-for-the-derivative-frac"
SEEDS = range(300)
x, u = sympy.symbols("x u")


def read(q):
    u_text = re.search(r"u = (.+?)\.$", q.stem).group(1)
    shown = re.search(r"y = (u\^\d|1/u\^\d|1/u)\b", q.stem).group(1)
    if shown.startswith("u^"):
        power = int(shown[2:])
    else:
        power = -int(shown[4:]) if shown.startswith("1/u^") else -1
    inner = sup.parse_poly(u_text, x=x)
    r = int(re.search(r"(?:rate of|constant) (\d+) units? per second", q.parts[3].prompt).group(1))
    x1 = int(re.search(r"x = (-?\d+)", q.parts[3].prompt).group(1))
    return power, inner, r, x1


def oracle(q):
    n, inner, r, x1 = read(q)
    composite = inner**n
    d_total = sympy.diff(composite, x)
    return n, inner, d_total, d_total.subs(x, x1) * r


def test_registered():
    sup.assert_registered(SKILL_ID, TOPIC)


def test_shape():
    sup.assert_chain_shape(TOPIC, ["expression", "expression", "expression", "number"], [1, 1, 2, 2], [0, 0, 1, 1])


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_prompt_shapes():
    sup.assert_shapes(SKILL_ID, TOPIC)


def test_answer_space():
    sup.assert_answer_space(TOPIC)


def test_key_matches_independent_derivation():
    for seed in SEEDS:
        q = sup.exam_question(TOPIC, seed)
        n, inner, d_total, rate = oracle(q)
        a, b, c, d = (p.answer_text for p in q.parts)
        assert sympy.simplify(sup.parse_poly(a, u=u) - n * u ** (n - 1)) == 0, (seed, a)
        assert sympy.simplify(sup.parse_poly(b, x=x) - sympy.diff(inner, x)) == 0, (seed, b)
        assert sympy.simplify(sup.parse_poly(c, x=x) - d_total) == 0, (seed, c)
        assert sympy.Rational(d) == rate, (seed, d, rate)


def test_distractors_tagged():
    sup.assert_tagged(TOPIC)


def test_grading():
    sup.assert_grading(SKILL_ID, TOPIC)


def test_follow_through_from_a_wrong_dy_du():
    sk = sup.skill_object(SKILL_ID)
    checked = 0
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        n, inner, d_total, rate = oracle(q)
        _, _, r, x1 = read(q)
        wrong_a = q.parts[0].distractors[0]
        wrong_outer = sup.parse_poly(wrong_a, u=u)
        wrong_total = sympy.expand(wrong_outer.subs(u, inner) * sympy.diff(inner, x))
        wrong_rate = wrong_total.subs(x, x1) * r
        if wrong_rate == 0 or wrong_rate == rate or not wrong_rate.is_integer:
            continue
        checked += 1
        reply = {"a": wrong_a, "b": q.parts[1].answer_text, "c": str(wrong_total), "d": str(wrong_rate)}
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (seed, reply)
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through, (seed, reply)
    assert checked >= 30
