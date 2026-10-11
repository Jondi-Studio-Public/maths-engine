"""Exam-tier source for expanding a single bracket: the multiplier is a monomial in two variables.

In the practice tiers the multiplier is a plain number (an integer or a simple fraction), so expanding is
repeated multiplication of coefficients. Here the multiplier carries variables and powers, ``-3x^2y``, and the
bracket holds three terms in x and y with their own powers. Every product needs the index law (add the
exponents of a repeated variable) as well as the coefficient arithmetic and the sign.

    Expand -3x^2y(2xy - 4x + 5y^2)   ->   -6x^3y^2 + 12x^3y - 15x^2y^3

Built backwards: the multiplier ``c x^i y^j`` and the three bracket monomials are drawn first; the key is the
sympy expansion. An independent route multiplies term by term with explicit coefficient and exponent
bookkeeping, and the two must agree.

Distractors (each a named misconception, filtered to differ from the key)
    partial-distribution  only the first bracket term is multiplied
    dropped-variable-part only the multiplier's coefficient is applied, its variables are lost
    exponent-not-added    a variable already in the bracket term keeps its old exponent instead of gaining the multiplier's
    sign-error            the sign of the last product is flipped

The answer is graded with ``answer_form="expanded"``: restating the question is rejected.
"""

from __future__ import annotations

import random

from sympy import expand, symbols

from ..core import (
    check_equivalent,
    fmt_signed,
    generator,
    join_terms,
    latex_answer,
    pick_distractors,
    symbolically_equal,
)

x, y = symbols("x y")


def _txt(expr) -> str:
    """Student-style text for a polynomial: no '*', '^' for powers."""
    return str(expr).replace("**", "^").replace("*", "")


def _var_text(a: int, b: int) -> str:
    """'x^2y' style variable part; '' when both exponents are zero."""
    xp = "" if a == 0 else ("x" if a == 1 else f"x^{a}")
    yp = "" if b == 0 else ("y" if b == 1 else f"y^{b}")
    return xp + yp


def _draw_bracket(rng: random.Random) -> list[tuple[int, int, int]]:
    """Three distinct monomials (coef, x-exp, y-exp), each with total degree at least 1."""
    pool = [(a, b) for a in range(3) for b in range(3) if a + b >= 1]
    exps = rng.sample(pool, 3)
    exps.sort(key=lambda e: (-(e[0] + e[1]), -e[0]))
    return [(rng.choice([-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7]), a, b) for a, b in exps]


def _exam(rng: random.Random) -> dict:
    c = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
    i, j = rng.choice([(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)])
    bracket = _draw_bracket(rng)
    mono = c * x**i * y**j
    terms = [k * x**a * y**b for k, a, b in bracket]

    key = expand(mono * sum(terms))
    # Independent route: explicit coefficient / exponent bookkeeping, one product at a time.
    by_hand = sum((c * k) * x ** (i + a) * y ** (j + b) for k, a, b in bracket)
    check_equivalent(key, by_hand, "term-by-term expansion must agree with sympy's expansion")

    first, rest = terms[0], terms[1:]
    candidates = [
        # partial-distribution
        expand(mono * first + sum(rest)),
        # dropped-variable-part
        expand(c * sum(terms)),
        # exponent-not-added
        expand(sum((c * k) * x ** (a if a else i) * y ** (b if b else j) for k, a, b in bracket)),
        # sign-error on the last product
        expand(mono * sum(terms[:-1]) - mono * terms[-1]),
    ]
    candidates = [cand for cand in candidates if not symbolically_equal(cand, key)]

    multiplier_text = fmt_signed(c, _var_text(i, j))
    bracket_text = join_terms([(k, _var_text(a, b)) for k, a, b in bracket])
    text = f"Expand {multiplier_text}({bracket_text})."
    answer_text = _txt(key)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(key),
        distractors=pick_distractors(answer_text, [_txt(cand) for cand in candidates], n=3),
        answer_form="expanded",
    )


@generator(
    "exd_x_a5_expand_monomial_multiplier",
    level=9,
    presets={"exam": dict(tier="exam")},
)
def exd_x_a5_expand_monomial_multiplier(rng: random.Random, tier: str = "exam") -> dict:
    """Expand a monomial (coefficient and powers of x and y) over a three-term bracket."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
