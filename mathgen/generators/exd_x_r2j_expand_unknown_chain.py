"""Exam chain on expanding (x + a)(p x^2 + k x + c) when k is unknown: find k from a stated coefficient, expand,
then use the expansion.

The stem gives a, p and c and the coefficient n of x in the expansion. Three parts:

(a) k, from n = c + a k;
(b) the expansion of P(x) = (x + a)(p x^2 + k x + c) with that k, fully expanded and collected;
(c) P(x0) for a stated x0, read off the expansion.

The key for (b) is the sympy expansion; (a) is solved the other way round (k is drawn, n is built from it).
"""

from __future__ import annotations

import random

import sympy

from mathgen.core import check, draw_until, generator, join_terms, phrase

from .exd_x_r2j_support import expression_part, number_part

_X = sympy.Symbol("x")

_STEM = [
    "Let P(x) = (x {a_term})({quad}), where k is a constant. In the expansion of P(x), the coefficient of x is {n}.",
    "The polynomial P(x) = (x {a_term})({quad}) has k as a constant. When P(x) is expanded, the coefficient of x "
    "is {n}.",
    "k is a constant and P(x) = (x {a_term})({quad}). The coefficient of x in the expanded form of P(x) is {n}.",
]


def _draw(rng: random.Random):
    a = rng.choice([v for v in range(-7, 8) if abs(v) >= 2])
    p = rng.choice([1, 2, 3, 4, -1, -2, -3])
    c = rng.choice([v for v in range(-9, 10) if v != 0])
    k = rng.choice([v for v in range(-8, 9) if v != 0])
    x0 = rng.choice([v for v in range(-3, 4) if v not in (0,)])
    return a, p, c, k, x0


def _accept(draw) -> bool:
    a, p, c, k, x0 = draw
    n = c + a * k
    value = (x0 + a) * (p * x0**2 + k * x0 + c)
    return n != 0 and abs(n) <= 60 and n != c and k + a * p != 0 and value != 0


def _poly_terms(p, q, r, s):
    return [(p, "x^3"), (q, "x^2"), (r, "x"), (s, "")]


@generator("exd_x_r2j_expand_unknown_chain", level=6, presets={"exam": dict(tier="exam")})
def exd_x_r2j_expand_unknown_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Find the unknown coefficient from the expansion's x coefficient, expand, then evaluate."""
    check(tier == "exam", "exd_x_r2j_expand_unknown_chain: only the exam tier exists")
    a, p, c, k, x0 = draw_until(rng, _draw, _accept, what="an expansion chain")
    n = c + a * k
    quad = join_terms([(p, "x^2"), (1, "kx"), (c, "")])
    a_term = f"+ {a}" if a > 0 else f"- {-a}"
    stem = phrase(rng, _STEM, a_term=a_term, quad=quad, n=n)

    expanded = sympy.expand((_X + a) * (p * _X**2 + k * _X + c))
    coeffs = [expanded.coeff(_X, e) for e in (3, 2, 1, 0)]
    check(coeffs == [p, k + a * p, c + a * k, a * c], "expansion coefficients disagree with sympy")
    key_text = join_terms(_poly_terms(*coeffs))
    value = expanded.subs(_X, x0)
    check(value == (x0 + a) * (p * x0**2 + k * x0 + c), "P(x0) disagrees with the factored form")

    def poly(q, r, s, lead=p):
        expr = lead * _X**3 + q * _X**2 + r * _X + s
        return expr, join_terms(_poly_terms(lead, q, r, s))

    wrong_b = []
    expr, text = poly(k, c, 0)  # x times the quadratic only: the a is never distributed
    wrong_b.append((expr, text, "only-multiplied-first-terms"))
    expr, text = poly(k + a * p, -c + a * k, a * c)  # sign flipped on c in the x-term only
    wrong_b.append((expr, text, "sign-error-on-one-term"))
    expr, text = poly(k + a * p, c, a * c)  # the a k x term left out
    wrong_b.append((expr, text, "dropped-cross-term"))
    expr, text = poly(k + a * p, c + a * k, a + c)  # a + c where a c belongs
    wrong_b.append((expr, text, "adds-instead-of-multiplies"))

    part_a = sympy.Symbol("part_a")
    ft_b = sympy.sstr(p * _X**3 + (part_a + a * p) * _X**2 + (c + a * part_a) * _X + a * c)
    ft_c = f"Subs(part_b, (x,), ({x0},))"
    leading_only = p * x0**3
    wrong_a = [
        (n - c, "forgot-to-divide-by-coefficient"),
        (sympy.Rational(n, a), "ignored-constant-term"),
        (-k, "sign-error-moving-term"),
        ((n - c) * a, "multiplies-instead-of-dividing"),
        (sympy.Rational(n + c, a), "sign-error-moving-term"),
    ]
    wrong_a = [(v, t) for v, t in wrong_a if sympy.Integer(v) == v]
    parts = [
        number_part(
            prompt="Find the value of k.",
            exact=k,
            places=0,
            marks=2,
            working=[
                f"The x terms come from x · ({c}) and ({a}) · kx, so the coefficient of x is {join_terms([(c, ''), (a, 'k')])}.",
                f"{join_terms([(c, ''), (a, 'k')])} = {n}, so k = {k}.",
            ],
            wrong=[(int(v), t) for v, t in wrong_a],
        ),
        expression_part(
            prompt="Hence expand and simplify P(x).",
            key=expanded,
            key_text=key_text,
            marks=3,
            working=[
                f"With k = {k}: P(x) = (x {a_term})({join_terms([(p, 'x^2'), (k, 'x'), (c, '')])}).",
                f"= {key_text}.",
            ],
            wrong=wrong_b,
            follow_through=ft_b,
            answer_form="expanded",
        ),
        number_part(
            prompt=f"Hence find the value of P({x0}).",
            exact=int(value),
            places=0,
            marks=1,
            working=[f"P({x0}) = {key_text.replace('x', f'({x0})')} = {value}."],
            wrong=[
                (int(w), tag)
                for w, tag in (
                    (expanded.subs(_X, -x0), "substitution-sign-error"),
                    (value - a * c, "dropped-constant-term"),
                    (leading_only, "ignores-lower-terms"),
                    (wrong_b[0][0].subs(_X, x0), "only-multiplied-first-terms"),
                )
            ],
            follow_through=ft_c,
        ),
    ]
    return dict(text=stem, parts=parts)
