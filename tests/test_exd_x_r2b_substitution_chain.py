"""The multi-part exam tier `exd_x_r2b_substitution_chain` of integration-by-substitution: the new limit, the
integrand in u, the value of the definite integral and the mean value.

The oracle reads the integrand and the limit out of the rendered stem and re-derives every key with SymPy (the
integral by SymPy's own `integrate` over the x-limits); it never imports the generator module.
"""

from __future__ import annotations

import re

import sympy

import exd_x_r2b_support as sup
from mathgen.engine import grade

TOPIC = "exd_x_r2b_substitution_chain"
SKILL_ID = "integration-by-substitution"
SEEDS = range(300)
x, u = sympy.symbols("x u")


def read(q):
    text = re.search(r"integral of (.+?) dx", q.stem).group(1)
    b = int(re.search(r"(?:and|to) x = (\d+)", q.stem).group(1))
    inner = sup.parse_poly(re.search(r"u = (.+?)\.", q.stem).group(1), x=x)
    if "/" in text:  # m x^p / (inner)^e
        lead, bracket = text.split("/")
        integrand = sup.parse_poly(lead, x=x) / inner ** int(bracket.split("^")[-1])
    else:  # m x^p (inner)^e
        head = re.match(r"(\d+x(?:\^\d)?)\(", text).group(1)
        integrand = sup.parse_poly(head, x=x) * inner ** int(text.rsplit("^", 1)[1])
    return integrand, inner, b


def oracle(q):
    integrand, inner, b = read(q)
    value = sympy.integrate(integrand, (x, 0, b))
    return inner.subs(x, b), value, value / b


def test_registered():
    sup.assert_registered(SKILL_ID, TOPIC)


def test_shape():
    sup.assert_chain_shape(TOPIC, ["fraction", "expression", "number", "number"], [1, 2, 2, 1], [0, 0, 1, 1])


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_prompt_shapes():
    sup.assert_shapes(SKILL_ID, TOPIC)


def test_answer_space():
    sup.assert_answer_space(TOPIC, distinct=100)


def test_key_matches_independent_derivation():
    for seed in SEEDS:
        q = sup.exam_question(TOPIC, seed)
        integrand, inner, b = read(q)
        u2, value, mean = oracle(q)
        a, g, c, d = (p.answer_text for p in q.parts)
        assert sympy.Rational(a) == u2, (seed, a)
        # g(u) du must reproduce the integrand: substitute u = inner and multiply by du/dx
        lhs = sup.parse_poly(g, u=u).subs(u, inner) * sympy.diff(inner, x)
        assert sympy.simplify(lhs - integrand) == 0, (seed, g, q.stem)
        assert (c, d) == (sup.dp(value, 2), sup.dp(mean, 2)), (seed, c, d)


def test_distractors_tagged():
    sup.assert_tagged(TOPIC)


def test_grading():
    sup.assert_grading(SKILL_ID, TOPIC)


def test_follow_through_from_a_wrong_coefficient():
    sk = sup.skill_object(SKILL_ID)
    checked = 0
    for seed in range(60):
        q = sup.exam_question(TOPIC, seed)
        integrand, inner, b = read(q)
        u2, _, _ = oracle(q)
        wrong_g = q.parts[1].distractors[0]
        g = sup.parse_poly(wrong_g, u=u)
        coefficient = g.subs(u, 1)
        exponent = sympy.degree(g * u**3, u) - 3 if g.is_polynomial(u) else None
        if exponent is None:
            exponent = -2
        k = inner.subs(x, 0)
        value = coefficient * (u2 ** (exponent + 1) - k ** (exponent + 1)) / (exponent + 1)
        mean = value / b
        canon = [sympy.Rational(q.parts[2].answer_text), sympy.Rational(q.parts[3].answer_text)]
        if (
            sup.rounded(value, 2) == canon[0]
            or sup.rounded(mean, 2) == canon[1]
            or sup.rounded(mean, 2) == sup.rounded(value, 2)
        ):
            continue
        checked += 1
        reply = {
            "a": str(u2),
            "b": wrong_g,
            "c": sup.dp(value, 2),
            "d": sup.dp(sympy.Rational(sup.dp(value, 2)) / b, 2),
        }
        result = grade(sk, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (seed, reply)
        assert result.parts["d"].marks == 1 and result.parts["d"].follow_through, (seed, reply)
    assert checked >= 30
