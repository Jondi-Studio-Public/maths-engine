"""The multi-part exam tier `exd_x_r2b_increments_chain` of senior-use-the-increments-formula-delta-y-cong-frac:
dy/dx, the estimated change, the estimated value and the error against the exact value.

The oracle reads y, x0 and dx out of the rendered stem and re-derives every key with SymPy and exact rationals; it
never imports the generator module.
"""

from __future__ import annotations

import re

import sympy

import exd_x_r2b_support as sup
from mathgen.engine import grade

TOPIC = "exd_x_r2b_increments_chain"
SKILL_ID = "senior-use-the-increments-formula-delta-y-cong-frac"
SEEDS = range(300)
x = sympy.symbols("x")


def read(q):
    f_text = re.search(r"y = (.+?)\. ", q.stem).group(1)
    x0 = int(re.search(r"x = (\d+)", q.stem).group(1))
    dx = sympy.Rational(re.search(r"δx = (-?\d+(?:\.\d+)?)", q.stem).group(1))
    return sup.parse_poly(f_text, x=x), x0, dx


def exact(text):
    return sympy.Rational(text)


def oracle(q):
    f, x0, dx = read(q)
    slope = sympy.diff(f, x).subs(x, x0)
    est_change = slope * dx
    est_new = f.subs(x, x0) + est_change
    error = f.subs(x, x0 + dx) - est_new
    return sympy.diff(f, x), est_change, est_new, error


def test_registered():
    sup.assert_registered(SKILL_ID, TOPIC)


def test_shape():
    sup.assert_chain_shape(TOPIC, ["expression", "number", "number", "number"], [2, 1, 1, 2], [0, 1, 1, 1])


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_prompt_shapes():
    sup.assert_shapes(SKILL_ID, TOPIC)


def test_answer_space():
    sup.assert_answer_space(TOPIC)


def test_key_matches_independent_derivation():
    for seed in SEEDS:
        q = sup.exam_question(TOPIC, seed)
        deriv, est_change, est_new, error = oracle(q)
        a, b, c, d = (p.answer_text for p in q.parts)
        assert sympy.expand(sup.parse_poly(a, x=x) - deriv) == 0, (seed, a)
        assert (exact(b), exact(c), exact(d)) == (est_change, est_new, sup.rounded(error, 3)), (seed, q.stem)


def test_distractors_tagged():
    sup.assert_tagged(TOPIC)


def test_grading():
    sup.assert_grading(SKILL_ID, TOPIC)


def test_follow_through_from_a_wrong_derivative():
    sk = sup.skill_object(SKILL_ID)
    checked = 0
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        f, x0, dx = read(q)
        wrong = q.parts[0].distractors[0]
        slope = sup.parse_poly(wrong, x=x).subs(x, x0)
        est_change = slope * dx
        est_new = f.subs(x, x0) + est_change
        error = f.subs(x, x0 + dx) - est_new
        values = [est_change, est_new, error]
        canon = [exact(part.answer_text) for part in q.parts[1:]]
        if 0 in values or len(set(values)) < 3 or any(v == slope for v in values) or values[:2] == canon[:2]:
            continue
        checked += 1
        reply = {"a": wrong, "b": str(sympy.N(est_change, 30)), "c": str(sympy.N(est_new, 30)), "d": sup.dp(error, 3)}
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0
        for label, value, want in zip("bcd", values, canon):
            assert result.parts[label].marks == result.parts[label].max_marks, (seed, label, reply)
            assert result.parts[label].follow_through or sup.rounded(value, 3) == want
    assert checked >= 40
