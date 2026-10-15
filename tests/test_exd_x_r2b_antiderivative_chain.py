"""The multi-part exam tier `exd_x_r2b_antiderivative_chain` of integrate-polynomial: a curve from its gradient
function with an unknown constant, a point, and a tangent's y-intercept.

The oracle reads the gradient function, the stated gradient and the point out of the rendered stem and re-derives
every key with SymPy (solve for k, integrate, fix the constant); it never imports the generator module.
"""

from __future__ import annotations

import re

import sympy

import exd_x_r2b_support as sup
from mathgen.engine import grade

TOPIC = "exd_x_r2b_antiderivative_chain"
SKILL_ID = "integrate-polynomial"
SEEDS = range(300)
x, k = sympy.symbols("x k")


def read(q):
    g_text = re.search(r"f'\(x\) = (.+?), where k", q.stem).group(1)
    tail = q.stem.split("where k is a constant.")[1]
    p = int(re.search(r"x = (-?\d+)", tail).group(1))
    grad = int(re.search(r"(?:has gradient|curve is|is) (-?\d+),", tail).group(1))
    qx, ty = (int(v) for v in re.search(r"\((-?\d+), (-?\d+)\)", tail).groups())
    x1 = int(re.search(r"f\((-?\d+)\)", q.parts[2].prompt).group(1))
    x2 = int(re.search(r"x = (-?\d+)", q.parts[3].prompt).group(1))
    return sup.parse_poly(g_text, x=x, k=k), p, grad, qx, ty, x1, x2


def oracle(q):
    gradient, p, grad, qx, ty, x1, x2 = read(q)
    k_val = sympy.solve(gradient.subs(x, p) - grad, k)[0]
    anti = sympy.integrate(gradient.subs(k, k_val), x)
    f = sympy.expand(anti + (ty - anti.subs(x, qx)))
    slope2, value2 = sympy.diff(f, x).subs(x, x2), f.subs(x, x2)
    return k_val, f, f.subs(x, x1), value2 - slope2 * x2


def test_registered():
    sup.assert_registered(SKILL_ID, TOPIC)


def test_shape():
    sup.assert_chain_shape(TOPIC, ["fraction", "expression", "fraction", "fraction"], [2, 2, 1, 2], [0, 1, 1, 1])


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_prompt_shapes():
    sup.assert_shapes(SKILL_ID, TOPIC)


def test_key_matches_independent_derivation():
    for seed in SEEDS:
        q = sup.exam_question(TOPIC, seed)
        k_val, f, value1, intercept = oracle(q)
        a, b, c, d = (p.answer_text for p in q.parts)
        assert sympy.Rational(a) == k_val, (seed, a)
        assert sympy.expand(sup.parse_poly(b, x=x) - f) == 0, (seed, b)
        assert (sympy.Rational(c), sympy.Rational(d)) == (value1, intercept), (seed, q.stem)


def test_distractors_tagged():
    sup.assert_tagged(TOPIC)


def test_grading():
    sup.assert_grading(SKILL_ID, TOPIC)


def test_follow_through_from_a_wrong_k():
    sk = sup.skill_object(SKILL_ID)
    checked = 0
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        gradient, _, _, qx, ty, x1, x2 = read(q)
        wrong = q.parts[0].distractors[0]
        wk = sympy.Rational(wrong)
        anti = sympy.integrate(gradient.subs(k, wk), x)
        f = sympy.expand(anti + (ty - anti.subs(x, qx)))
        value1 = f.subs(x, x1)
        intercept = f.subs(x, x2) - sympy.diff(f, x).subs(x, x2) * x2
        canon = [sympy.Rational(q.parts[2].answer_text), sympy.Rational(q.parts[3].answer_text)]
        if 0 in (wk, value1, intercept) or value1 == canon[0] or intercept == canon[1] or wk in (value1, intercept):
            continue
        checked += 1
        reply = {"a": wrong, "b": str(f).replace("**", "^"), "c": str(value1), "d": str(intercept)}
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0
        for label, marks in (("b", 2), ("c", 1), ("d", 2)):
            assert result.parts[label].marks == marks and result.parts[label].follow_through, (seed, label, reply)
    assert checked >= 30
