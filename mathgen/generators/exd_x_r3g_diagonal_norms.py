"""Exam tier for "compute a diagonal operator norm".

The practice tiers ask for the norm of one diagonal map, a composition, or a power. Here the learner meets one
invertible diagonal map on R^2 (Euclidean norm) and works through the quantities a numerical-analysis or functional
analysis course attaches to it:

(a) the operator norm ||T|| = the largest |entry|;
(b) the operator norm of a power, ||T^k||, which is ||T||^k for a diagonal map;
(c) the operator norm of the inverse, ||T^-1|| = 1 / (the smallest |entry|);
(d) the condition number ||T|| ||T^-1||, from (a) and (c).

All answers are exact fractions. The two entries have different absolute values, so max and min are distinct.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, generator, rational_text
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3g_kit import clashes, fraction_part

_STEMS = (
    "A linear map T on R² acts diagonally: T(x, y) = ({e1}, {e2}). Operator norms are taken with respect to the "
    "Euclidean norm.",
    "Let T be the diagonal linear map T(x, y) = ({e1}, {e2}) on R², with the Euclidean (2-) norm throughout.",
    "The map T: R² → R² is given by T(x, y) = ({e1}, {e2}). Use the Euclidean norm on R².",
)


def _entry(c: Fraction, var: str) -> str:
    sign = "-" if c < 0 else ""
    num, den = abs(c.numerator), c.denominator
    body = var if num == 1 else f"{num}{var}"
    return f"{sign}{body}" if den == 1 else f"{sign}{body}/{den}"


def _draw_entry(rng: random.Random) -> Fraction:
    return Fraction(rng.choice((-1, 1)) * rng.randint(1, 9), rng.randint(1, 5))


def _br(value) -> str:
    """A fraction in brackets, so it can be divided or multiplied: ``(1/2)``; a whole number is left alone."""
    return f"({value})" if "/" in str(value) else str(value)


@generator("exd_x_r3g_diagonal_norms", level=12, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3g_diagonal_norms(rng: random.Random, tier: str = "exam") -> dict:
    """Norm, norm of a power, norm of the inverse and condition number of a 2 x 2 diagonal map."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(1000, "two entries of different size with non-trivial norms"):
        d1, d2 = _draw_entry(rng), _draw_entry(rng)
        big, small = max(abs(d1), abs(d2)), min(abs(d1), abs(d2))
        if big == small or small == 1 or big > 4 or (big.denominator == 1 and small.denominator == 1):
            continue
        k = rng.choice((2, 3, 3, 4))
        norm_pow, inverse, cond = big**k, 1 / small, big / small
        stem = rng.choice(_STEMS).format(e1=_entry(d1, "x"), e2=_entry(d2, "y"))
        pb = f"Find the operator norm of T^{k}, the map T applied {k} times."
        if clashes(
            (rational_text(big), rational_text(norm_pow), rational_text(inverse), rational_text(cond)), stem, pb
        ):
            continue
        break

    signed_big = d1 if abs(d1) == big else d2
    parts = [
        fraction_part(
            "Find the operator norm ‖T‖.",
            big,
            2,
            [f"‖T‖ is the largest |entry|: max({abs(d1)}, {abs(d2)}) = {big}."],
            [
                (small, "uses-min-instead-of-max"),
                (abs(d1) + abs(d2), "operator-norm-sums-the-entries"),
                (signed_big, "sign-of-result-reversed") if signed_big < 0 else None,
                (abs(d1 * d2), "operator-norm-multiplies-the-entries"),
            ],
            sensible_range=(0, None),
        ),
        fraction_part(
            pb,
            norm_pow,
            2,
            [f"T^{k} = diag({d1**k}, {d2**k}), so ‖T^{k}‖ = {norm_pow}."],
            [
                (big * k, "multiplied-instead-of-exponentiated"),
                (small**k, "uses-min-instead-of-max"),
                (signed_big**k, "forgets-absolute-value-after-power") if signed_big < 0 and k % 2 else None,
                (big ** (k - 1), "off-by-one-count"),
            ],
            follow_through=f"part_a**{k}",
            sensible_range=(0, None),
        ),
        fraction_part(
            "Find the operator norm of the inverse map T⁻¹.",
            inverse,
            2,
            [f"T⁻¹ = diag({1 / d1}, {1 / d2}); the largest |entry| is 1/{_br(small)} = {inverse}."],
            [
                (1 / big, "operator-norm-inverse-uses-the-largest-entry"),
                (small, "operator-norm-inverse-not-reciprocated"),
                (1 / abs(d1) + 1 / abs(d2), "operator-norm-inverse-sums-the-reciprocals"),
                (big - small, "norm-from-a-difference"),
            ],
            sensible_range=(0, None),
        ),
        fraction_part(
            "Find the condition number κ(T) = ‖T‖ ‖T⁻¹‖.",
            cond,
            2,
            [f"κ(T) = {big} × {inverse} = {cond}."],
            [
                (big * small, "condition-number-uses-the-smallest-entry"),
                (big + inverse, "added-instead-of-multiplied"),
                (big - small, "norm-from-a-difference"),
                (small / big, "condition-number-inverted"),
            ],
            follow_through="part_a*part_c",
            sensible_range=(1, None),
        ),
    ]
    return dict(text=stem, parts=parts)
