"""Exam-tier source for simplifying bracketed expressions: nested brackets, worked from the inside out.

The practice tiers expand one bracket and combine like terms. Here a bracket sits inside a square bracket, the
inner one is multiplied and the other one is subtracted, and the whole square bracket has its own multiplier
(sometimes negative), with a loose term after it. The learner must choose an order -- inside out -- and carry the
sign of the subtraction through before the outer multiplier is applied.

    Simplify 3[2(4x - 5) - (x + 6)] - 4x   ->   17x - 48

Half the questions replace the constants with a second variable, so there are two kinds of like term to track.

Built backwards from the coefficients, with the key taken from sympy's expansion. An independent route computes
each coefficient from the drawn numbers by hand (``a * (b * p - r)`` for x and so on) and must agree.

Distractors (each a named misconception, filtered to differ from the key)
    minus-as-decoration   the subtracted inner bracket is added
    outer-multiplier-once the outer multiplier reaches only the first term of the square bracket
    partial-inner         the inner multiplier is applied to the x-term only
    one-sign-flipped      only the first term of the subtracted bracket is negated

The answer is graded with ``answer_form="expanded"``: restating the question is rejected.
"""

from __future__ import annotations

import random

from sympy import expand, symbols

from ..core import check_equivalent, generator, join_terms, latex_answer, pick_distractors, symbolically_equal

x, y = symbols("x y")


def _txt(expr) -> str:
    """Student-style text for a polynomial: no '*', '^' for powers."""
    return str(expr).replace("**", "^").replace("*", "")


def _nonzero(rng: random.Random, lo: int, hi: int) -> int:
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


def _build(rng: random.Random) -> dict:
    two_var = rng.random() < 0.5
    z, z_name = (y, "y") if two_var else (1, "")
    a = rng.choice([-4, -3, -2, 2, 3, 4])
    b = rng.randint(2, 4)
    p, q, r, s = (_nonzero(rng, -6, 6) for _ in range(4))
    e = _nonzero(rng, -5, 5)

    inner1 = p * x + q * z
    inner2 = r * x + s * z
    tail = e * x

    key = expand(a * (b * inner1 - inner2) + tail)
    # Independent route: coefficient arithmetic, no expression algebra.
    x_coef = a * (b * p - r) + e
    z_coef = a * (b * q - s)
    check_equivalent(key, x_coef * x + z_coef * z, "coefficient arithmetic must agree with sympy's expansion")
    if key == 0:
        raise ValueError("expression cancels to zero")

    candidates = [
        # minus-as-decoration
        expand(a * (b * inner1 + inner2) + tail),
        # outer-multiplier-once
        expand(a * b * inner1 - inner2 + tail),
        # partial-inner
        expand(a * (b * p * x + q * z - inner2) + tail),
        # one-sign-flipped
        expand(a * (b * inner1 - r * x + s * z) + tail),
    ]
    candidates = [c for c in candidates if not symbolically_equal(c, key)]

    inner1_text = join_terms([(p, "x"), (q, z_name)])
    inner2_text = join_terms([(r, "x"), (s, z_name)])
    outer = f"{a}[{b}({inner1_text}) - ({inner2_text})]"
    text = f"Simplify {join_terms([(1, outer), (e, 'x')])}."
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
    # Redraw the rare expression that cancels to zero rather than ask for "0".
    for _ in range(50):
        try:
            return _build(rng)
        except ValueError:
            continue
    raise RuntimeError("could not draw a non-zero expression")


@generator(
    "exd_x_a5_nested_brackets",
    level=9,
    presets={"exam": dict(tier="exam")},
)
def exd_x_a5_nested_brackets(rng: random.Random, tier: str = "exam") -> dict:
    """Simplify a multiplied square bracket containing a multiplied bracket and a subtracted bracket."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
