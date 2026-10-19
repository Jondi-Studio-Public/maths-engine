"""``families-of-curves-and-linearity``: answers grade right, every distractor wrong, and the answers are recomputed from the stem."""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from r3h_support import add_common_tests, questions

PRACTICE = "r3h_families"
EXAM = "exd_x_r3h_families"
add_common_tests(globals(), PRACTICE, EXAM)

x = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def _parse(text):
    text = re.sub(r"(?<![A-Za-z])e\^(\w+)", r"exp(\1)", text.strip())
    text = re.sub(r"(?<![A-Za-z])e\^\(", "exp(", text)
    return parse_expr(
        text, local_dict={"x": x, **{n: sympy.Symbol(n) for n in ("fx", "gx", "fprime")}}, transformations=_TRANSFORMS
    )


def _zero(expr):
    """Whether `expr` vanishes at several probe points (a numerical test; SymPy's simplifier is slow and not needed)."""
    return all(abs(complex(sympy.N(expr.subs(x, v)))) < 1e-9 for v in (sympy.Rational(3, 10), sympy.Rational(7, 5), 2))


def _combination(text, names=("f", "g")):
    """The coefficients of the named symbols in a printed combination like ``-2f(x) - 3g(x)``."""
    swapped = text.replace("f'(x)", "fprime").replace("f(x)", "fx").replace("g(x)", "gx")
    expr = _parse(swapped)
    lookup = {"f": "fx", "g": "gx", "fprime": "fprime", "g(x)": "gx"}
    return [int(expr.coeff(sympy.Symbol(lookup.get(n, n)))) for n in names]


def test_easy_derivative_matches_concrete_functions():
    for q in questions(PRACTICE, "easy", range(1, 25)):
        x0, p, _, q2 = re.search(r"f'\((-?\d+)\) = (-?\d+), g'\((-?\d+)\) = (-?\d+)", q.text).groups()[0:4]
        x0, p, q2 = int(x0), int(p), int(q2)
        body = re.search(r"h\(x\) = (.*?)\. Find", q.text).group(1)
        r, s = _combination(body)
        u = re.search(r"([+-]) (\d*)x\^2", body)
        u = (-1 if u.group(1) == "-" else 1) * (int(u.group(2)) if u.group(2) else 1)
        # concrete functions with the stated slopes at x0 (their other values are irrelevant to h')
        f = p * (x - x0) + 3
        g = q2 * (x - x0) - 7
        c = int(re.search(r"([+-]) (\d+)$", body).group(2)) * (-1 if re.search(r"- \d+$", body) else 1)
        h = r * f + s * g + u * x**2 + c
        assert sympy.diff(h, x).subs(x, x0) == int(q.answer), q.text


def test_medium_integral_matches_concrete_functions():
    for q in questions(PRACTICE, "medium", range(1, 25)):
        P = re.search(r"∫ f\(x\) dx = (-?\d+)", q.text).group(1)
        Q = re.search(r"∫ g\(x\) dx = (-?\d+)", q.text).group(1)
        a, b = re.search(r"from x = (-?\d+) to x = (-?\d+)", q.text).groups()
        a, b, P, Q = int(a), int(b), int(P), int(Q)
        body = re.search(r"value of ∫ \((.*)\) dx over", q.text).group(1)
        r, s = _combination(body)
        tail = body.split("g(x)")[1].strip()
        extra = _parse(tail) if tail else 0
        # constant integrands with the stated integrals: linearity must not depend on which functions they are
        f = sympy.Rational(P, b - a)
        g = sympy.Rational(Q, b - a)
        assert sympy.integrate(r * f + s * g + extra, (x, a, b)) == int(q.answer), q.text


def test_hard_answer_differentiates_to_the_combination_and_hits_the_values():
    for q in questions(PRACTICE, "hard", range(1, 25)):
        fp = _parse(re.search(r"f'\(x\) = (.*?), f\(", q.text).group(1))
        gp = _parse(re.search(r"g'\(x\) = (.*?) and g\(", q.text).group(1))
        x0, yf = (int(v) for v in re.search(r"f\((-?\d+)\) = (-?\d+)", q.text).groups())
        yg = int(re.search(r"g\((-?\d+)\) = (-?\d+)", q.text).group(2))
        r, s = _combination(re.search(r"h\(x\) = (.*?)\.", q.text).group(1))
        h = _parse(q.answer_text)
        assert _zero(sympy.diff(h, x) - (r * fp + s * gp)), q.text
        assert abs(float(sympy.N(h.subs(x, x0) - (r * yf + s * yg)))) < 1e-9, q.text


def test_exam_chain_is_consistent_with_the_stem():
    for q in questions(EXAM, "exam", range(1, 25)):
        fp = _parse(re.search(r"f'\(x\) = (.*?) and f\(", q.text).group(1))
        x0, y0 = (int(v) for v in re.search(r"and f\((-?\d+)\) = (-?\d+)\.", q.text).groups())
        parts = {p.label: p for p in q.parts}
        F = _parse(parts["a"].answer_text)
        assert _zero(sympy.diff(F, x) - fp), q.text
        C = y0 - F.subs(x, x0)
        assert abs(float(parts["b"].answer) - float(C)) < 1e-9, q.text
        x1 = int(re.search(r"Find f\((-?\d+)\)", q.text).group(1))
        dp = len(parts["c"].answer_text.split(".")[1]) if "." in parts["c"].answer_text else 0
        assert abs(float(F.subs(x, x1) + C) - float(parts["c"].answer)) < 0.5 * 10**-dp + 1e-9, q.text
        m = re.search(r"∫ g\(x\) dx = (-?\d+)", q.text).group(1)
        a, b = re.search(r"from x = (-?\d+) to x = (-?\d+)", q.text).groups()
        body = re.search(r"Find ∫ \((.*?)\) dx over", q.text).group(1)
        r, s = _combination(body, names=("g", "fprime"))
        step = (float(b) - float(a)) / 4000
        slope = sympy.lambdify(x, fp, "math")
        mid = sum(slope(float(a) + (i + 0.5) * step) for i in range(4000)) * step
        want = r * int(m) + s * mid
        assert abs(want - float(parts["d"].answer)) < 0.5 * 10**-dp + 1e-3, q.text


def test_written_down_antiderivatives_match_sympy():
    import random

    from mathgen.generators import r3h_families_build as build

    rng = random.Random(7)
    for kind in ("poly", "exp", "cos", "sin"):
        for _ in range(40):
            fp, F = build._draw_fprime(rng, kind)
            assert sympy.expand(F - sympy.integrate(fp, x)) == 0, (kind, fp, F)
