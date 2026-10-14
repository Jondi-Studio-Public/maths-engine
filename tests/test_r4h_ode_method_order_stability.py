"""Tests `uc2-ode-method-order-and-stability`: `r4h_ode_method_order_stability` and its exam chain.

The oracle works from the text: it reads the matrix, the theta, or the amplification factor, and answers with SymPy
(eigenvalues; the exact solution of ``|R(z)| <= 1``; the Taylor series of ``R(z) - exp(-z)``).
"""

from __future__ import annotations

import re

import sympy

from mathgen import make
from tests import r4h_support as sup

SKILL_ID, TOPIC, EXAM = (
    "uc2-ode-method-order-and-stability",
    "r4h_ode_method_order_stability",
    "exd_x_r4h_ode_method_order_stability",
)
SKILL = sup.skill_of(SKILL_ID)
Z, H = sympy.symbols("z h", positive=True)
R = sympy.Rational


def series_order(r_expr):
    diff = sympy.series(r_expr - sympy.exp(-Z), Z, 0, 9).removeO()
    poly = sympy.Poly(sympy.expand(diff), Z)
    low = min(m[0] for m in poly.monoms())
    return low - 1, poly.coeff_monomial(Z**low) * -1


def stable_limit(r_expr):
    """Largest z > 0 with |R(z)| <= 1, as a SymPy rational: the boundary is where R = -1."""
    sols = sympy.solve(sympy.Eq(r_expr, -1), Z)
    return min(s for s in sols if s.is_real and s > 0)


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM)


def test_easy_key_is_two_over_the_largest_eigenvalue():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="easy")
        rows = re.search(r"A = \[\[(.*?)\], \[(.*?)\]\]", sup.clean(q.text))
        a = sympy.Matrix([[int(v) for v in rows.group(i).split(", ")] for i in (1, 2)])
        biggest = max(abs(v) for v in a.eigenvals())
        assert R(q.answer_text) == 2 / biggest, (seed, q.text, q.answer_text)


def test_medium_key_is_the_boundary_of_the_stable_region():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="medium")
        theta = R(re.search(r"θ = (\d+/\d+)", q.text).group(1))
        lam = int(re.search(r"y′ = −(\d+)y", q.text).group(1))
        r_expr = (1 - (1 - theta) * Z) / (1 + theta * Z)
        assert R(q.answer_text) == stable_limit(r_expr) / lam, (seed, q.text, q.answer_text)


def test_hard_key_is_the_order_from_the_series():
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty="hard")
        text = sup.clean(re.search(r"R\(z\) = (.*?)(?:\. |, where| Compare|\.$)", q.text).group(1))
        for k in "²³⁴⁵⁶":
            text = text.replace("z" + k, "z**" + str("²³⁴⁵⁶".index(k) + 2))
        text = re.sub(r"(\)|\d)z", r"\1*z", text)
        expr = sympy.sympify(text, locals={"z": Z})
        assert int(q.answer_text) == series_order(expr)[0], (seed, q.text, q.answer_text)


def test_tiers_differ_in_shape():
    for seed in range(30):
        easy, medium, hard = (make(TOPIC, seed=seed, difficulty=t).text for t in sup.TIERS)
        assert "explicit Euler" in easy or "Euler" in easy
        assert "θ" in medium and "θ" not in easy
        assert "R(" in hard and "R(" not in medium


def test_options_text_variety():
    sup.check_key_and_distractors(SKILL, TOPIC)
    sup.check_text_and_determinism(TOPIC)
    sup.check_variety(TOPIC, distinct=3, repeats=60, by="text")


def method_r(stem):
    text = sup.clean(stem)
    th = re.search(r"θ = (\d+/\d+)", text)
    if th:
        t = R(th.group(1))
        return (1 - (1 - t) * Z) / (1 + t * Z)
    if "f(yₙ + h f(yₙ))" in text:
        return 1 - Z + Z**2 / 2
    if "f(yₙ₊₁)]" in text and "(h/2)" in text:
        return (1 - Z / 2) / (1 + Z / 2)
    if "h f(yₙ₊₁)" in text:
        return 1 / (1 + Z)
    return 1 - Z


def test_exam_shape_and_keys():
    sup.check_exam_shape(EXAM, ["mcq", "fraction", "fraction", "fraction"], [2, 1, 2, 2], [False, False, True, False])
    for seed in range(60):
        q = sup.exam(EXAM, seed)
        r_expr = method_r(q.stem)
        order, const = series_order(r_expr)
        lam = int(re.search(r"y′ = −(\d+)y", q.parts[1].prompt).group(1))
        h = R(re.search(r"h = (\d+/\d+|\d+)", q.parts[1].prompt).group(1))
        r_val = r_expr.subs(Z, lam * h)
        y0 = int(re.search(r"y₀ = (\d+)", q.parts[2].prompt).group(1))
        a, b, c, d = q.parts
        stable_always = sympy.limit(abs(r_expr), Z, sympy.oo) <= 1 and all(
            abs(r_expr.subs(Z, v)) <= 1 for v in (1, 5, 50, 500)
        )
        expected = (
            ("First" if order == 1 else "Second")
            + " order, "
            + ("stable for every step size" if stable_always else "stable only for small enough steps")
        )
        assert a.answer_text == expected, (seed, q.stem, a.answer_text, expected)
        assert R(b.answer_text) == r_val and R(c.answer_text) == y0 * r_val**3 and R(d.answer_text) == const, seed


def test_exam_grading_and_follow_through():
    sup.check_exam_grading(SKILL, EXAM)
    sup.check_follow_through(SKILL, EXAM, minimum=20)
    sup.check_exam_variety(EXAM, distinct=30)
