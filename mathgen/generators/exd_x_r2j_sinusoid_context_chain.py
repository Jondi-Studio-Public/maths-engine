"""Exam chain on a periodic scenario modelled by a sinusoid that starts at its minimum: h(t) = D - A cos(2 pi t / T).

The stem states A, D and T in words (and a threshold height H for the later parts). Four parts:

(a) the height at a stated time, to 1 decimal place;
(b) the first time the height reaches H, t = T arccos((D - H) / A) / (2 pi), to 3 decimal places;
(c) hence how long in each cycle the quantity is above H, T - 2 t, to 2 decimal places;
(d) hence the percentage of each cycle spent above H, to the nearest whole number.

The draw is kept only when working through the rounded earlier answers lands on the same displayed answers.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from mathgen.core import check, draw_until, generator, phrase

from .exd_x_r2j_support import clear_of_boundary, number_part, shown

_SCENARIOS = [
    ("A Ferris wheel's height above the ground", "height", "metres", "minutes"),
    ("The tide's height above the sea floor at a jetty", "height", "metres", "hours"),
    ("A piston's distance from the base of its cylinder", "distance", "centimetres", "seconds"),
    ("The depth of water over a harbour sandbar", "depth", "metres", "hours"),
    ("The depth of water at a river mouth", "depth", "metres", "hours"),
]

_STEM = [
    "{what} varies sinusoidally with amplitude {a} {unit} about a mean of {d} {unit}, with a period of {period} "
    "{tunit}. At time t = 0 it is at its minimum.",
    "{what} follows a sinusoidal pattern with a period of {period} {tunit}. Its amplitude is {a} {unit}, its mean "
    "value is {d} {unit}, and at t = 0 it is at its lowest.",
    "A model for {what_lower} is sinusoidal, with mean {d} {unit}, amplitude {a} {unit} and period {period} "
    "{tunit}, starting from its minimum at t = 0.",
]


def _draw(rng: random.Random, shortest: int = 5):
    a = rng.randint(2, 6)
    d = a + rng.randint(1, 6)
    period = rng.randint(shortest, 24)
    h = Fraction(round(float(d + rng.uniform(-0.8, 0.8) * a) * 10), 10)
    t1 = Fraction(rng.randint(5, int(period * 20)), 10)
    return a, d, period, h, t1


def _chain(a, d, period, h, t1):
    height = d - a * math.cos(2 * math.pi * float(t1) / period)
    t = period * math.acos((d - float(h)) / a) / (2 * math.pi)
    above = period - 2 * t
    pct = 100 * above / period
    return height, t, above, pct


def _accept(draw) -> bool:
    a, d, period, h, t1 = draw
    if abs(float(h) - d) < 0.3 * a:
        return False
    height, t, above, pct = _chain(a, d, period, h, t1)
    if not all(clear_of_boundary(Fraction(v), p) for v, p in ((height, 1), (t, 3), (above, 2), (pct, 0))):
        return False
    t3 = shown(Fraction(t), 3)
    above2 = shown(Fraction(period) - 2 * t3, 2)
    return above2 == shown(Fraction(above), 2) and shown(above2 / period * 100, 0) == shown(Fraction(pct), 0)


@generator("exd_x_r2j_sinusoid_context_chain", level=14, presets={"exam": dict(tier="exam")})
def exd_x_r2j_sinusoid_context_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Height at a time, first time at a threshold, time above the threshold per cycle, and its share of a cycle."""
    check(tier == "exam", "exd_x_r2j_sinusoid_context_chain: only the exam tier exists")
    what, noun, unit, tunit = rng.choice(_SCENARIOS)
    a, d, period, h, t1 = draw_until(
        rng, lambda r: _draw(r, 12 if tunit == "hours" else 5), _accept, what="a stable sinusoid chain"
    )
    height, t, above, pct = _chain(a, d, period, h, t1)
    stem = phrase(
        rng,
        _STEM,
        what=what,
        what_lower=what[0].lower() + what[1:],
        a=a,
        d=d,
        period=period,
        unit=unit,
        tunit=tunit,
    )
    h_text = f"{float(h):g}"
    t1_text = f"{float(t1):g}"
    r = (d - float(h)) / a
    check(abs(d - a * math.cos(2 * math.pi * t / period) - float(h)) < 1e-9, "the time found does not give height H")
    check(abs(above - (period - 2 * t)) < 1e-12 and 0 < above < period, "time above the threshold out of range")

    parts = [
        number_part(
            prompt=f"Find the {noun} after {t1_text} {tunit}, in {unit}, correct to 1 decimal place.",
            exact=Fraction(height),
            places=1,
            marks=2,
            working=[f"h(t) = {d} - {a}cos(2πt/{period}).", f"h({t1_text}) = {height:.1f} {unit}."],
            wrong=[
                (v, tag)
                for v, tag in (
                    (Fraction(d + a * math.cos(2 * math.pi * float(t1) / period)), "reflected-about-mean"),
                    (Fraction(a * math.cos(2 * math.pi * float(t1) / period)), "dropped-constant-term"),
                    (Fraction(d - a * math.sin(2 * math.pi * float(t1) / period)), "sine-cosine-swapped"),
                    (Fraction(d - a * math.cos(float(t1))), "forgot-period-scaling"),
                )
                if v > 0
            ],
            sensible_range=(0, None),
        ),
        number_part(
            prompt=f"Find the first time the {noun} reaches {h_text} {unit}, in {tunit}, correct to 3 decimal places.",
            exact=Fraction(t),
            places=3,
            marks=3,
            working=[
                f"{d} - {a}cos(2πt/{period}) = {h_text}, so cos(2πt/{period}) = {r:.4f}.",
                f"t = {period} × arccos({r:.4f}) / (2π) = {t:.3f} {tunit}.",
            ],
            wrong=[
                (Fraction(period * math.acos(r) / math.pi), "period-factor-slip"),
                (Fraction(period - t), "later-root-instead-of-first"),
                (Fraction(period * math.acos(-r) / (2 * math.pi)), "arccos-sign-slip"),
                (Fraction(period, 2), "half-period-guess"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            prompt=f"Hence find how long in each cycle the {noun} is above {h_text} {unit}, in {tunit}, correct to "
            "2 decimal places.",
            exact=Fraction(above),
            places=2,
            marks=2,
            working=[
                f"The height is above {h_text} {unit} between t = {t:.3f} and t = {period} - {t:.3f}.",
                f"Time above = {period} - 2 × {t:.3f} = {above:.2f} {tunit}.",
            ],
            wrong=[
                (Fraction(period - t), "forgot-symmetry-of-cycle"),
                (Fraction(2 * t), "time-below-for-time-above"),
                (Fraction(t), "forgot-symmetry-of-cycle"),
                (Fraction(period / 2 - t), "forgot-symmetry-of-cycle"),
            ],
            follow_through=f"{period} - 2*part_b",
            sensible_range=(0, period),
        ),
        number_part(
            prompt=f"Hence find the percentage of each cycle for which the {noun} is above {h_text} {unit}, to the "
            "nearest whole number.",
            exact=Fraction(pct),
            places=0,
            marks=1,
            working=[f"{above:.2f} / {period} × 100 = {pct:.0f}%."],
            wrong=[
                (Fraction(100 * t / period), "percent-of-wrong-time"),
                (Fraction(100 - pct), "time-below-for-time-above"),
                (Fraction(100 * (period - t) / period), "forgot-symmetry-of-cycle"),
                (Fraction(above), "gave-time-not-percent"),
            ],
            follow_through=f"part_c/{period}*100",
            sensible_range=(0, 100),
        ),
    ]
    return dict(text=stem, parts=parts)
