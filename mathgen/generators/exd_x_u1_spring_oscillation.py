"""Exam-tier source for `simple-harmonic-angular-frequency`: from a static extension to the period of oscillation.

A spring hangs vertically and a known mass stretches it by a measured amount while at rest. The learner has to
chain three ideas, as a first-year mechanics paper does:

* (a) the spring constant from the equilibrium (weight = k x), with the extension given in centimetres and
  g = 9.8 m/s^2;
* (b) the angular frequency of a different mass on the same spring, omega = sqrt(k / m), to 2 decimal places;
* (c) the period T = 2 pi / omega, to 2 decimal places, following through from (b).

Built backwards from an integer spring constant: the mass and extension are drawn so that k = m g / x is a whole
number, so (a) is exact. In (c) the key is worked out from the exact omega, and the draw is kept only when the
value worked out from omega already rounded to 2 decimal places (as a learner does) gives the same 2-decimal
answer and every displayed value is clear of a rounding boundary.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from ..answers import Distractor, exact_value
from ..core import (
    attempts,
    check,
    generator,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

G = Fraction(98, 10)  # m/s^2
PI = Fraction(math.pi)
_MARGIN = Fraction(2, 1000)  # keep a displayed value this far from a 2-decimal-place rounding boundary
_SETUPS = [
    "A spring hangs vertically from a fixed support. A {m0} kg mass hung from it at rest stretches it by "
    "{x0} cm. Take g = 9.8 m/s².",
    "In a physics laboratory, a {m0} kg mass hangs at rest from a vertical spring and extends it by {x0} cm. "
    "Use g = 9.8 m/s².",
    "A vertical spring is stretched by {x0} cm when a {m0} kg mass hangs at rest from its end. "
    "Take g = 9.8 m/s² and ignore the mass of the spring.",
]
_B_PROMPTS = [
    "The {m0} kg mass is replaced by a {m} kg mass, which is pulled down slightly and released. "
    "Find the angular frequency of the oscillation, in rad/s, correct to 2 decimal places.",
    "A {m} kg mass is hung from the spring in place of the {m0} kg mass and set oscillating vertically. "
    "Find its angular frequency, in rad/s, correct to 2 decimal places.",
]
_C_PROMPTS = [
    "Hence find the period of the oscillation, in seconds, correct to 2 decimal places.",
    "Using your answer to (b), find the time for one complete oscillation, in seconds, correct to 2 decimal places.",
]


def _dp2(value: Fraction) -> Fraction:
    return round_half_up(value, 2)


def _clear(value: Fraction) -> bool:
    """True when `value` is at least `_MARGIN` from a 2-decimal-place rounding boundary."""
    offset = (value * 100) % 1
    return abs(offset - Fraction(1, 2)) * Fraction(1, 100) > _MARGIN


def _sqrt_fraction(value: Fraction) -> Fraction:
    """A rational approximation of sqrt(value) that is accurate far beyond any displayed digit."""
    scale = 10**12
    root = math.isqrt(value.numerator * scale * scale // value.denominator)
    return Fraction(root, scale)


def _tenths(j: int) -> str:
    return f"{j // 10}.{j % 10}"


def _draw(rng: random.Random):
    for _ in range(2000):
        j0 = rng.randint(5, 40)  # original mass, in tenths of a kilogram
        x0 = rng.randint(2, 30)  # extension in cm
        if (j0 * 98) % x0:
            continue
        k = j0 * 98 // x0
        if not 20 <= k <= 400:
            continue
        j = rng.randint(3, 40)
        if j == j0:
            continue
        m = Fraction(j, 10)
        omega = _sqrt_fraction(Fraction(k) / m)
        if not 2 <= omega <= 40 or not _clear(omega):
            continue
        period = 2 * PI / omega
        period_from_rounded = _dp2(2 * PI / _dp2(omega))
        if _dp2(period) != period_from_rounded or not _clear(period) or period < Fraction(1, 5):
            continue
        return j0, x0, k, j, m, omega, period
    raise RuntimeError("no suitable mass and extension in 2000 draws")


@generator("exd_x_u1_spring_oscillation", level=16, presets={"exam": dict(tier="exam")})
def exd_x_u1_spring_oscillation(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: spring constant from a static extension, then angular frequency and period for another mass."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(200, "a spring question with enough distractors"):
        question = _build(rng)
        if question is not None:
            return question


def _build(rng: random.Random) -> dict | None:
    j0, x0, k, j, m, omega, period = _draw(rng)
    m0 = Fraction(j0, 10)

    # Independent routes: k from weight = k x in SI units, omega and T from floats.
    check(Fraction(j0, 10) * G / Fraction(x0, 100) == k, "spring constant disagrees with weight = k x")
    check(abs(float(omega) - math.sqrt(k / float(m))) < 1e-9, "omega disagrees with the float route")
    check(abs(float(period) - 2 * math.pi / math.sqrt(k / float(m))) < 1e-9, "period disagrees with the float route")

    omega_text = stated_precision_string(_dp2(omega), 2)
    period_text = stated_precision_string(_dp2(period), 2)
    tol = stated_precision_tolerance(2) * 2

    def d2(value: Fraction, tag):
        shown = _dp2(value)
        return Distractor(stated_precision_string(shown, 2), tag, exact_value(shown))

    def whole_or_2dp(value: Fraction, tag):
        shown = _dp2(value)
        text = str(int(shown)) if shown.denominator == 1 else stated_precision_string(shown, 2)
        return Distractor(text, tag, exact_value(shown))

    a_cands = [
        whole_or_2dp(m0 / (Fraction(x0, 100)), "forgot-weight-is-mass-times-g"),
        whole_or_2dp(m0 * G * x0, "unit-slip-cm-for-m"),
        whole_or_2dp(m0 * G, None),
        whole_or_2dp(Fraction(k) / 2, None),
    ]
    b_cands = [
        d2(Fraction(k) / m, "omits-square-root"),
        d2(_sqrt_fraction(m / Fraction(k)), "inverted-ratio"),
        d2(_sqrt_fraction(Fraction(k) / m0), "uses-original-mass-in-formula"),
        d2(_sqrt_fraction(Fraction(k) * m), None),
    ]
    c_cands = [
        d2(2 * PI * omega, "period-multiplied-by-omega"),
        d2(omega / (2 * PI), "period-as-frequency"),
        d2(1 / omega, "period-omits-two-pi"),
        d2(2 * PI / (omega * omega), None),
    ]
    a_dist = [d for d in pick_distractors(str(k), a_cands, 3) if d.text != str(k)]
    b_dist = pick_distractors(omega_text, b_cands, 3, tolerance=tol)
    c_dist = pick_distractors(period_text, c_cands, 3, tolerance=tol)
    # a distractor that rounds to zero at 2 decimal places is not a believable option
    b_dist = [d for d in b_dist if Fraction(d.text) >= 1]
    c_dist = [d for d in c_dist if Fraction(d.text) >= 1]
    if min(len(a_dist), len(b_dist), len(c_dist)) < 2:
        return None

    m_text, m0_text = _tenths(int(m * 10)), _tenths(j0)
    return dict(
        text=rng.choice(_SETUPS).format(m0=m0_text, x0=x0),
        parts=[
            dict(
                prompt="Find the spring constant, in N/m.",
                answer_kind="number",
                answer=k,
                answer_text=str(k),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"At rest the weight balances the spring force: m g = k x, with x = {x0} cm = {x0 / 100:g} m.",
                    f"k = {m0_text} × 9.8 ÷ {x0 / 100:g} = {k} N/m.",
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=rng.choice(_B_PROMPTS).format(m0=m0_text, m=m_text),
                answer_kind="number",
                answer=numeric_answer(_dp2(omega)),
                answer_text=omega_text,
                marks=2,
                sensible_range=(0, None),
                follow_through=f"sqrt(part_a/({m.numerator}/{m.denominator}))",
                working=[f"ω = √(k/m) = √({k} ÷ {m_text}) = {omega_text} rad/s (2 d.p.)."],
                distractors=b_dist,
            ),
            dict(
                prompt=rng.choice(_C_PROMPTS),
                answer_kind="number",
                answer=numeric_answer(_dp2(period)),
                answer_text=period_text,
                marks=1,
                sensible_range=(0, None),
                follow_through="2*pi/part_b",
                working=[f"T = 2π/ω = 2π ÷ {omega_text} = {period_text} s (2 d.p.)."],
                distractors=c_dist,
            ),
        ],
    )
