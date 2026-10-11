"""Exam tier for ``y = tan(x + c)``: vertical asymptotes of ``y = tan(ax + c)``.

The practice tiers shift a unit-period tangent. Here the argument also has a horizontal dilation, written either
as ``tan(a x + c)`` or in the factored form ``tan(a(x + d))``. The learner must solve ``a x + c = pi/2 + k pi``
for x, which divides the whole right-hand side (period and shift) by ``a``, and then pick the smallest positive
value. Built forwards from the coefficient fractions so every answer is an exact multiple of pi.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, format_pi_multiple, generator, pick_distractors

_HALF = Fraction(1, 2)
_SHIFTS = [Fraction(n, 12) for n in range(-11, 12) if n != 0]


def _smallest_positive(offset: Fraction, period: Fraction) -> Fraction:
    """Smallest value > 0 of the form offset + k*period."""
    k = (-offset) // period + 1
    value = offset + k * period
    check(0 < value <= period, "not the smallest positive representative")
    return value


def _signed_pi(coeff: Fraction) -> tuple[str, str]:
    return ("-" if coeff < 0 else "+"), format_pi_multiple(abs(coeff))


@generator("exd_x_f5_tanasym", level=14, presets={"exam": dict(tier="exam")})
def exd_x_f5_tanasym(rng: random.Random, tier: str = "exam") -> dict:
    """Smallest positive vertical asymptote of tan(ax + c) / tan(a(x + d))."""
    if tier != "exam":
        raise ValueError(f"unknown tier: {tier}")
    while True:
        a = rng.choice([2, 3, 4])
        if rng.random() < 0.5:
            c = rng.choice(_SHIFTS)
            sign, mag = _signed_pi(c)
            arg = f"{a}x {sign} {mag}"
            d = c / a
        else:
            d = rng.choice(_SHIFTS)
            c = a * d
            sign, mag = _signed_pi(d)
            arg = f"{a}(x {sign} {mag})"
        text = (
            f"For y = tan({arg}), find the smallest positive value of x at which the graph has a vertical asymptote. "
            f"Give your answer as an exact value in terms of pi."
        )
        # a x + c = pi/2 + k pi   =>   x = (1/2 - c)/a + k/a   (in units of pi)
        key = _smallest_positive((_HALF - c) / a, Fraction(1, a))
        # independent check: cos(a x + c) vanishes at the key and at no smaller positive grid point
        check(((a * key + c) - _HALF).denominator == 1, "cos(ax + c) does not vanish at the key")
        answer_text = format_pi_multiple(key)
        candidates = [
            _smallest_positive(_HALF - c, Fraction(1)),  # ignores the dilation: solves x + c = pi/2
            _smallest_positive(_HALF / a - c, Fraction(1, a)),  # divides pi/2 by a but not the shift
            _smallest_positive((_HALF + c) / a, Fraction(1, a)),  # sign error on the shift
            _smallest_positive((_HALF - c) * a, Fraction(a)),  # multiplies by a instead of dividing
        ]
        distractors = pick_distractors(answer_text, [format_pi_multiple(v) for v in candidates], n=3)
        if len(distractors) < 3:
            continue
        break
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_text,
        answer_text=answer_text,
        answer_latex=f"plain:{answer_text}",
        working=[
            f"tan is undefined where the argument is pi/2 + k pi, so {a}x + c = pi/2 + k pi with c = {format_pi_multiple(c)}.",
            f"x = (pi/2 - c)/{a} + k pi/{a}; the smallest positive value is {answer_text}.",
        ],
        distractors=distractors,
    )
