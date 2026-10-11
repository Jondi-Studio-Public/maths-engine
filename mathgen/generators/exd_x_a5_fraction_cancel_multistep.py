"""Exam-tier source for simplifying algebraic fractions: the factors are not visible until you find them.

In the practice tiers the denominator is already one of the numerator's two linear factors, so the learner
only has to find the other one. Here the learner has to factor BOTH parts first -- pulling out a common
numerical or variable factor, spotting a difference of squares inside a quartic, or recognising a two-variable
trinomial -- and only then cancel. Every family still divides out exactly, so the answer is a polynomial.

Families (random choice per seed)
    common_then_squares  (a x^3 - a k^2 x) / (x^2 + s k x)   : cancel x, then a difference of squares
    numeric_denominator  (c m x^2 - c m k^2) / (c x + s c k)  : a numerical factor in both parts
    quadratic_in_x2      (x^4 + (w - t) x^2 - t w) / (x^2 - t) : a quartic that factors as two x^2 brackets
    two_variable         (x^2 + (p + q) x y + p q y^2) / (x + p y) : a trinomial in x and y

Built backwards from the factored form: the key is the polynomial quotient, the numerator and denominator are
the expanded products. An independent check divides with ``sympy.div`` / ``cancel`` and requires a zero remainder
and agreement with the key.

Distractors (each a named misconception)
    dropped-coefficient  the numerical factor of the quotient is lost
    wrong-root           the denominator's own factor, or the other root's sign, is given back
    sign-error           the quotient's constant has the wrong sign
    read-off-middle      the middle coefficient of the numerator is taken as the answer's constant

The answer is graded with ``answer_form="expanded"``: restating the fraction is rejected.
"""

from __future__ import annotations

import random

from sympy import cancel, div, expand, symbols

from ..core import check, check_equivalent, generator, latex_answer, phrase, pick_distractors, symbolically_equal

x, y = symbols("x y")

_PATTERNS = (
    "Simplify {expr}.",
    "Simplify the algebraic fraction {expr}.",
)


def _txt(expr) -> str:
    """Student-style text for a polynomial: no '*', '^' for powers."""
    return str(expr).replace("**", "^").replace("*", "")


def _common_then_squares(rng: random.Random):
    a = rng.randint(2, 5)
    k = rng.randint(2, 7)
    s = rng.choice((1, -1))
    num = expand(a * x * (x - k) * (x + k))
    den = expand(x * (x + s * k))
    key = expand(a * (x - s * k))
    cands = [
        x - s * k,  # dropped-coefficient
        expand(a * (x + s * k)),  # wrong-root
        expand(-a * (x - s * k)),  # sign-error (whole quotient negated)
    ]
    return num, den, key, cands, x


def _numeric_denominator(rng: random.Random):
    c = rng.randint(2, 4)
    m = rng.randint(1, 3)
    k = rng.randint(2, 6)
    s = rng.choice((1, -1))
    num = expand(c * m * (x - k) * (x + k))
    den = expand(c * (x + s * k))
    key = expand(m * (x - s * k))
    cands = [
        x - s * k,  # dropped-coefficient
        expand(c * m * (x - s * k)),  # forgot to divide the numerical factor
        expand(m * (x + s * k)),  # wrong-root
    ]
    return num, den, key, cands, x


def _quadratic_in_x2(rng: random.Random):
    t = rng.choice((1, 4, 9, 16))
    # w != t, so w != -t either: the quotient never equals the denominator.
    w = rng.choice([v for v in (1, 4, 9, 16, 25) if v != t])
    if rng.random() < 0.5:
        w = -w
    num = expand((x**2 - t) * (x**2 + w))
    den = x**2 - t
    key = x**2 + w
    cands = [
        x**2 - w,  # sign-error
        den,  # wrong-root: gives the denominator back
        x**2 + (w - t),  # read-off-middle
    ]
    return num, den, key, cands, x


def _two_variable(rng: random.Random):
    p = rng.choice((-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6))
    q = rng.choice([v for v in (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6) if v != p and v != -p])
    num = expand((x + p * y) * (x + q * y))
    den = x + p * y
    key = x + q * y
    cands = [
        den,  # wrong-root
        x - q * y,  # sign-error
        x + (p + q) * y,  # read-off-middle
    ]
    return num, den, key, cands, y


_FAMILIES = (_common_then_squares, _numeric_denominator, _quadratic_in_x2, _two_variable)


def _exam(rng: random.Random) -> dict:
    num, den, key, cands, _ = rng.choice(_FAMILIES)(rng)

    quotient, remainder = div(num, den, x)
    check(remainder == 0, "the fraction must divide out exactly")
    check_equivalent(expand(quotient), key, "polynomial division must agree with the constructed quotient")
    check_equivalent(cancel(num / den), key, "cancel() must agree with the constructed quotient")

    cands = [c for c in cands if not symbolically_equal(c, key)]
    check(len(cands) >= 2, "too few distinct misconceptions")

    answer_text = _txt(key)
    text = phrase(rng, _PATTERNS, expr=f"({_txt(num)})/({_txt(den)})")
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=latex_answer(key),
        distractors=pick_distractors(answer_text, [_txt(c) for c in cands], n=3),
        answer_form="expanded",
    )


@generator(
    "exd_x_a5_fraction_cancel_multistep",
    level=9,
    presets={"exam": dict(tier="exam")},
)
def exd_x_a5_fraction_cancel_multistep(rng: random.Random, tier: str = "exam") -> dict:
    """Factor numerator and denominator, then cancel to a polynomial."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
