"""Exam-tier source for `factorise-quadratic`: factorising where the quadratic is not handed over in the plain
one-variable form.

Two shapes, chosen per seed, both of which ask for the same kind of answer as the practice tiers (a product
of two integer binomials) but need an extra idea before the usual search for factors can start:

* ``two_variable`` -- a homogeneous quadratic in ``x`` and ``y`` such as ``6x^2 + 5xy - 4y^2``. The
  binomials carry a ``y`` term, so the learner has to see that the single-variable method transfers.
* ``substitution`` -- a quadratic in ``x^2``, such as ``2x^4 + 5x^2 - 3``. The learner must treat ``x^2`` as
  a single unknown, factorise, and keep ``x^2`` in each binomial: ``(2x^2 - 1)(x^2 + 3)``.

Both are built backwards from the two binomials, then re-checked with ``sympy.factor_list`` (an independent
route that knows nothing about the construction).
"""

from __future__ import annotations

import math
import random

import sympy
from sympy import expand, symbols

from ..core import check, check_distinct, check_equivalent, fmt_signed, generator, join_terms, pick_distractors

x, y = symbols("x y")


def _binomial(p: int, q: int, var2: str = "") -> str:
    """'px + q' (or 'px + qy' when ``var2`` is given) with the sign written the way a student writes it."""
    sign = "+" if q >= 0 else "-"
    tail = fmt_signed(abs(q), var2) if var2 else str(abs(q))
    return f"{fmt_signed(p, 'x')} {sign} {tail}"


def _pair(p: int, q: int, r: int, s: int, var2: str = "") -> str:
    return f"({_binomial(p, q, var2)})({_binomial(r, s, var2)})"


def _draw_binomials(rng: random.Random, p_choices, r_choices, k_range: int) -> tuple[int, int, int, int]:
    """Draw (p, q, r, s) with nonzero constants, no common factor inside either binomial, and the two
    binomials not proportional (so the trinomial is not a perfect square)."""
    while True:
        p = rng.choice(p_choices)
        r = rng.choice(r_choices)
        q = rng.randint(1, k_range) * rng.choice([1, -1])
        s = rng.randint(1, k_range) * rng.choice([1, -1])
        if math.gcd(p, abs(q)) != 1 or math.gcd(r, abs(s)) != 1:
            continue
        if p * s == q * r:
            continue
        return p, q, r, s


def _two_variable(rng: random.Random) -> dict:
    while True:
        p, q, r, s = _draw_binomials(rng, [1, 2, 3, 4], [2, 3, 4], 5)
        expr = expand((p * x + q * y) * (r * x + s * y))
        a, b, c = p * r, p * s + q * r, q * s
        if b == 0 or math.gcd(a, math.gcd(abs(b), abs(c))) != 1:
            continue
        break
    # sympy normally orders x before y, so the first-bracket/second-bracket order below is the
    # one the learner sees when they list the factors; any order is accepted by the grader.
    text_expr = join_terms([(a, "x^2"), (b, "xy"), (c, "y^2")])
    answer = _pair(p, q, r, s, "y")
    check_equivalent(expr, a * x**2 + b * x * y + c * y**2, "two_variable expansion")
    factors = sympy.factor_list(expr)[1]
    check(
        len(factors) == 2 and all(sympy.Poly(f, x, y).total_degree() == 1 for f, _ in factors), "not two linear factors"
    )
    check_equivalent(sympy.factor(expr), (p * x + q * y) * (r * x + s * y), "sympy.factor disagrees")

    cands = []
    for label, cand in (
        ("flip one y sign", (p, -q, r, s)),
        ("flip both y signs", (p, -q, r, -s)),
        ("swap leading coefficients", (r, q, p, s)),
    ):
        value = expand((cand[0] * x + cand[1] * y) * (cand[2] * x + cand[3] * y))
        if value != expr:
            check_distinct(expr, value, msg=label)
            cands.append(_pair(*cand, "y"))
    return dict(
        text=f"Factorise {text_expr}.",
        latex=f"plain:Factorise {text_expr}.",
        answer=answer,
        answer_text=answer,
        answer_latex=answer,
        distractors=pick_distractors(answer, cands),
        answer_form="factored",
    )


def _substitution(rng: random.Random) -> dict:
    """A quadratic in x^2, such as ``2x^4 + 5x^2 - 3``, factorised as ``(2x^2 - 1)(x^2 + 3)``.

    The prompt has no brackets: a prompt with a bracketed substitution (``2(x + 1)^2 + ...``) would be
    graded correct when typed straight back, because it already has a bracket and equals the key.
    """
    while True:
        p, q, r, s = _draw_binomials(rng, [1, 2, 3], [1, 2, 3], 6)
        if p * r == 1 and rng.random() < 0.7:
            continue
        # Each factor a*x^2 + c must be irreducible over the integers, otherwise it could be factorised
        # further (a negative constant with a*|c| a perfect square is a difference of squares).
        if any(c < 0 and math.isqrt(a_ * -c) ** 2 == a_ * -c for a_, c in ((p, q), (r, s))):
            continue
        break
    a, b, c = p * r, p * s + q * r, q * s
    if b == 0:
        return _substitution(rng)
    text_expr = join_terms([(a, "x^4"), (b, "x^2"), (c, "")])
    expr = expand(a * x**4 + b * x**2 + c)
    answer = f"({_binomial(p, q).replace('x', 'x^2')})({_binomial(r, s).replace('x', 'x^2')})"
    check_equivalent(expr, (p * x**2 + q) * (r * x**2 + s), "factors disagree")
    factors = sympy.factor_list(expr)[1]
    check(len(factors) == 2 and all(sympy.Poly(f, x).degree() == 2 for f, _ in factors), "not two quadratic factors")

    cands = []
    for label, cand, power in (
        ("factorised as if x^2 were x", (p, q, r, s), 1),
        ("kept x^2 in one factor only", (p, q, r, s), None),
        ("flipped one constant", (p, -q, r, s), 2),
    ):
        if power is None:
            value = expand((p * x**2 + q) * (r * x + s))
            text_cand = f"({_binomial(p, q).replace('x', 'x^2')})({_binomial(r, s)})"
        else:
            value = expand((cand[0] * x**power + cand[1]) * (cand[2] * x**power + cand[3]))
            text_cand = "(" + _binomial(cand[0], cand[1]).replace("x", f"x^{power}" if power > 1 else "x") + ")("
            text_cand += _binomial(cand[2], cand[3]).replace("x", f"x^{power}" if power > 1 else "x") + ")"
        if value != expr:
            check_distinct(expr, value, msg=label)
            cands.append(text_cand)
    return dict(
        text=f"Factorise {text_expr}.",
        latex=f"plain:Factorise {text_expr}.",
        answer=answer,
        answer_text=answer,
        answer_latex=answer,
        distractors=pick_distractors(answer, cands),
        answer_form="factored",
    )


@generator(
    "exd_x_a1_factorise_quadratic",
    level=9,
    presets={
        "exam": dict(tier="exam"),
    },
)
def exd_x_a1_factorise_quadratic(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: a two-variable or bracket-substitution quadratic to factorise."""
    check(tier == "exam", f"unknown tier {tier!r}")
    return _two_variable(rng) if rng.random() < 0.5 else _substitution(rng)
