"""Exam-tier source for factoring out a common factor: the common factor has variables as well as a number.

In the practice tiers the expression is linear (or has one obvious power of x), so the common factor is a
number, or a number times x. Here there are three terms in TWO variables, each with its own powers, and the
highest common factor is a number times the lowest power of each variable that appears in every term:

    Factorise 12x^3y - 18x^2y^2 + 30xy   ->   6xy(2x^2 - 3xy + 5)

The learner has to take the gcd of the coefficients, then for each variable pick the smallest exponent across
all three terms, and divide each term by the whole factor (subtracting exponents).

Built backwards: a coefficient gcd ``g``, a variable factor ``x^i y^j`` (at least one variable) and three
bracket monomials with coprime coefficients and no variable common to all three are drawn first. The key is
``g x^i y^j (bracket)`` and the prompt is its expansion. The independent check recomputes the highest common
factor from the expanded terms alone (integer gcd and minimum exponents) and requires it to equal ``g x^i y^j``.

Distractors (each a named misconception; all are value-different from the question, so none is a merely
under-factored form)
    divide-one-term-only   only the first term is divided by the common factor
    exponent-not-reduced   one bracket term is divided by the number but keeps the factor's variable powers
    sign-error             the last bracket term has the wrong sign

The answer is graded with ``answer_form="factored"``, which needs a bracket and value-equality.
"""

from __future__ import annotations

import math
import random

from sympy import Poly, expand, symbols

from ..core import check, fmt_signed, generator, join_terms, latex_answer, pick_distractors, symbolically_equal

x, y = symbols("x y")


def _txt(expr) -> str:
    """Student-style text for a polynomial: no '*', '^' for powers."""
    return str(expr).replace("**", "^").replace("*", "")


def _var_text(a: int, b: int) -> str:
    xp = "" if a == 0 else ("x" if a == 1 else f"x^{a}")
    yp = "" if b == 0 else ("y" if b == 1 else f"y^{b}")
    return xp + yp


def _draw_inner(rng: random.Random) -> list[tuple[int, int, int]]:
    """Three distinct monomials (coef, x-exp, y-exp): coprime coefficients, and for each variable at least one
    term free of it, so no variable factor is left to take out of the bracket."""
    pool = [(a, b) for a in range(3) for b in range(3)]
    while True:
        exps = rng.sample(pool, 3)
        if min(a for a, _ in exps) == 0 and min(b for _, b in exps) == 0:
            break
    exps.sort(key=lambda e: (-(e[0] + e[1]), -e[0]))
    while True:
        coefs = [rng.choice([-7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]) for _ in range(3)]
        if math.gcd(math.gcd(abs(coefs[0]), abs(coefs[1])), abs(coefs[2])) == 1:
            break
    if coefs[0] < 0:  # lead with a positive term, as a textbook would
        coefs = [-c for c in coefs]
    return [(c, a, b) for c, (a, b) in zip(coefs, exps)]


def _exam(rng: random.Random) -> dict:
    g = rng.randint(2, 6)
    i, j = rng.choice([(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)])
    inner = _draw_inner(rng)
    factor = g * x**i * y**j
    terms = [c * x**a * y**b for c, a, b in inner]
    bracket = sum(terms)
    expr = expand(factor * bracket)

    # Independent check: the highest common factor, recomputed from the expanded terms alone.
    poly_terms = Poly(expr, x, y).terms()
    check(len(poly_terms) == 3, "the expanded expression must keep three terms")
    gcd_coef = 0
    for _, coef in poly_terms:
        gcd_coef = math.gcd(gcd_coef, abs(int(coef)))
    min_x = min(m[0] for m, _ in poly_terms)
    min_y = min(m[1] for m, _ in poly_terms)
    check(gcd_coef * x**min_x * y**min_y == factor, "recomputed highest common factor must match the drawn one")

    factor_text = fmt_signed(g, _var_text(i, j))
    bracket_text = join_terms([(c, _var_text(a, b)) for c, a, b in inner])
    answer_text = f"{factor_text}({bracket_text})"

    def wrapped(items) -> str:
        return f"{factor_text}({join_terms([(c, _var_text(a, b)) for c, a, b in items])})"

    def value(items):
        return expand(factor * sum(c * x**a * y**b for c, a, b in items))

    head, tail, last = inner[0], inner[1:], inner[-1]
    slips = [
        # divide-one-term-only: the other terms are left undivided
        [head] + [(c * g, a + i, b + j) for c, a, b in tail],
        # exponent-not-reduced: the last term loses the number but keeps the factor's variables
        inner[:-1] + [(last[0], last[1] + i, last[2] + j)],
        # sign-error on the last term
        inner[:-1] + [(-last[0], last[1], last[2])],
    ]
    candidates = [wrapped(items) for items in slips if not symbolically_equal(value(items), expr)]

    text = f"Factorise {_txt(expr)} by taking out the highest common factor."
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=pick_distractors(answer_text, candidates, n=3),
        answer_form="factored",
    )


@generator(
    "exd_x_a5_factor_monomial_gcf",
    level=9,
    presets={"exam": dict(tier="exam")},
)
def exd_x_a5_factor_monomial_gcf(rng: random.Random, tier: str = "exam") -> dict:
    """Factor a number-and-variables highest common factor out of a three-term expression in x and y."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
