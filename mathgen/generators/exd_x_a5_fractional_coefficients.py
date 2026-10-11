"""Exam-tier source for simplifying multi-operation expressions: algebraic fractions with different denominators.

The practice tiers combine integer multiples of brackets. Here two terms are algebraic fractions -- a whole
bracket over a denominator -- with DIFFERENT denominators, plus a multiplied bracket, so the learner has to treat
each numerator as a unit, put everything over a common denominator (or divide through term by term), take care of
the sign of the subtracted fraction, and leave fractional coefficients in the answer.

    Simplify (2x + 1)/3 - (x - 4)/6 + 2(x - 1)   ->   11x/6 - 1/2

Half the questions replace the constants with a second variable y. The answer must be written ``ax + b`` (or
``ax + by``) with each fractional coefficient written like ``11x/6``.

Built backwards from the drawn numerators and denominators, with the key taken from sympy's exact expansion over
the rationals. An independent route recomputes each coefficient with ``fractions.Fraction`` and must agree.

Distractors (each a named misconception, filtered to differ from the key)
    sign-not-carried      a subtracted fraction has only its first numerator term negated
    divided-first-only    each fraction's denominator is applied to the first numerator term only
    multiplier-first-only the integer multiplier of the last bracket reaches only its first term
    denominators-added    the two fractions are combined over the sum of their denominators

The answer is graded with ``answer_form="expanded"``: leaving it as a bracketed fraction is rejected.
"""

from __future__ import annotations

import random
from fractions import Fraction

from sympy import Rational, expand, symbols

from ..core import check_equivalent, generator, join_terms, latex_answer, pick_distractors, symbolically_equal

x, y = symbols("x y")


def _term(coef: Rational, var: str) -> tuple[str, bool]:
    """(unsigned text, is_negative) for one term like 11x/6, x/2, 3y or 5/2."""
    mag = abs(coef)
    if not var:
        body = str(mag.p) if mag.q == 1 else f"{mag.p}/{mag.q}"
    elif mag.q == 1:
        body = var if mag.p == 1 else f"{mag.p}{var}"
    else:
        body = f"{var}/{mag.q}" if mag.p == 1 else f"{mag.p}{var}/{mag.q}"
    return body, coef < 0


def _txt(expr) -> str:
    """Answer text in the order the prompt asks for: x-term, then y-term or constant, e.g. '11x/6 - 1/2'."""
    expr = expand(expr)
    parts = []
    for var in ("x", "y", ""):
        coef = expr.coeff(x, 1) if var == "x" else expr.coeff(y, 1) if var == "y" else expr.subs({x: 0, y: 0})
        if coef == 0:
            continue
        body, neg = _term(Rational(coef), var)
        parts.append((neg, body))
    out = ""
    for idx, (neg, body) in enumerate(parts):
        out += (f"-{body}" if neg else body) if idx == 0 else (f" - {body}" if neg else f" + {body}")
    return out


def _nonzero(rng: random.Random, lo: int, hi: int) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


def _build(rng: random.Random) -> dict:
    two_var = rng.random() < 0.5
    z, z_name = (y, "y") if two_var else (1, "")
    d1, d2 = rng.sample([2, 3, 4, 6], 2)
    m1, m2, m3 = (rng.randint(1, 5) for _ in range(3))  # lead with a positive x-term, as printed
    n1, n2, n3 = (_nonzero(rng, -5, 5) for _ in range(3))
    if (m1 % d1 == 0 and n1 % d1 == 0) or (m2 % d2 == 0 and n2 % d2 == 0):
        raise ValueError("a fraction whose numerator divides exactly is not worth the step")
    k = rng.choice([-3, -2, 2, 3])
    op = rng.choice((1, -1))

    a1, a2, a3 = m1 * x + n1 * z, m2 * x + n2 * z, m3 * x + n3 * z
    key = expand(Rational(1, d1) * a1 + op * Rational(1, d2) * a2 + k * a3)

    # Independent route: exact rational coefficients, no expression algebra.
    x_coef = Fraction(m1, d1) + op * Fraction(m2, d2) + k * m3
    z_coef = Fraction(n1, d1) + op * Fraction(n2, d2) + k * n3
    check_equivalent(
        key,
        Rational(x_coef.numerator, x_coef.denominator) * x + Rational(z_coef.numerator, z_coef.denominator) * z,
        "exact-fraction arithmetic must agree with sympy's expansion",
    )
    if key == 0 or (x_coef.denominator == 1 and z_coef.denominator == 1):
        raise ValueError("want a non-zero answer with a fractional coefficient")

    candidates = []
    if op == -1:
        candidates.append(expand(Rational(1, d1) * a1 - Rational(1, d2) * m2 * x + Rational(1, d2) * n2 * z + k * a3))
    candidates.append(expand(Rational(1, d1) * m1 * x + n1 * z + op * (Rational(1, d2) * m2 * x + n2 * z) + k * a3))
    candidates.append(expand(Rational(1, d1) * a1 + op * Rational(1, d2) * a2 + k * m3 * x + n3 * z))
    candidates.append(expand(Rational(1, d1 + d2) * (a1 + op * a2) + k * a3))
    candidates = [c for c in candidates if not symbolically_equal(c, key)]

    n1_text = join_terms([(m1, "x"), (n1, z_name)])
    n2_text = join_terms([(m2, "x"), (n2, z_name)])
    n3_text = join_terms([(m3, "x"), (n3, z_name)])
    joiner = "+" if op == 1 else "-"
    expr_text = f"({n1_text})/{d1} {joiner} ({n2_text})/{d2} " + (f"+ {k}" if k > 0 else f"- {abs(k)}") + f"({n3_text})"
    form = "ax + by" if two_var else "ax + b"
    text = f"Simplify {expr_text}. Write your answer as {form}, with any fractional coefficient written like 5x/6."
    answer_text = _txt(key)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(key),
        distractors=pick_distractors(answer_text, [_txt(c) for c in candidates], n=3),
        answer_form="expanded",
    )


def _exam(rng: random.Random) -> dict:
    # Redraw a zero or all-integer answer rather than ask for something the tier is not about.
    for _ in range(50):
        try:
            return _build(rng)
        except ValueError:
            continue
    raise RuntimeError("could not draw a usable expression")


@generator(
    "exd_x_a5_fractional_coefficients",
    level=9,
    presets={"exam": dict(tier="exam")},
)
def exd_x_a5_fractional_coefficients(rng: random.Random, tier: str = "exam") -> dict:
    """Combine two algebraic fractions with different denominators and a multiplied bracket."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
