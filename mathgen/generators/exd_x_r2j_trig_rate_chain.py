"""Exam chain on an oscillating quantity h(t) = a sin(bt) + c: its greatest rate of change, the first time the
rate takes a stated value, the quantity's value then, and how fast the rate is changing then.

(a) the maximum rate of change of h, a*b;
(b) the smallest t > 0 at which dh/dt = k, t = arccos(k / (a b)) / b, to 3 decimal places;
(c) hence h at that time, a sin(b t) + c, to 2 decimal places;
(d) hence d^2h/dt^2 at that time, -a b^2 sin(b t), to 1 decimal place.

Since b t lies in (0, pi), sin(b t) is positive and equals sqrt(1 - (k / (a b))^2), so (c) and (d) have an exact
route as well as the route through the rounded time that (b) hands on. The draw is kept only when both routes
display the same answer.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

from mathgen.core import check, draw_until, generator, phrase

from .exd_x_r2j_support import number_part, shown

_SCENARIOS = [
    ("The height of a piston in an engine cylinder", "h", "millimetre", "second"),
    ("The water level at a tidal gauge", "h", "metre", "hour"),
    ("The height above the ground of a point on a Ferris wheel", "h", "metre", "minute"),
    ("The depth of water over a reef", "h", "metre", "hour"),
    ("The height of a buoy above the sea floor", "h", "metre", "minute"),
]

_STEM = [
    "{what} is modelled by h(t) = {rule}, where t (t ≥ 0) is the time in {tunit}s and h is measured in {unit}s.",
    "{what} is modelled by h(t) = {rule}. Here t (t ≥ 0) is the time in {tunit}s and h is in {unit}s.",
]


def _rule(a: int, b: int, c: int) -> str:
    inner = "t" if b == 1 else f"{b}t"
    return f"{a}sin({inner}) + {c}"


def _draw(rng: random.Random):
    a, b, c = rng.randint(2, 6), rng.randint(1, 3), rng.randint(4, 20)
    k = rng.choice([v for v in range(-a * b + 1, a * b) if v != 0])
    return a, b, c, k


def _chain(a: int, b: int, c: int, k: int):
    r = k / (a * b)
    t = math.acos(r) / b
    s = math.sqrt(1 - r * r)
    return t, a * s + c, -a * b * b * s


def _accept(draw) -> bool:
    a, b, c, k = draw
    t, height, accel = _chain(a, b, c, k)
    if not (shown(Fraction(t), 3) > 0):
        return False
    t3 = float(shown(Fraction(t), 3))
    h_back = a * math.sin(b * t3) + c
    acc_back = -a * b * b * math.sin(b * t3)
    return (
        shown(Fraction(h_back), 2) == shown(Fraction(height), 2)
        and shown(Fraction(acc_back), 1) == shown(Fraction(accel), 1)
        and all(
            abs((Fraction(v) * 10**p) % 1 - Fraction(1, 2)) > Fraction(1, 40)
            for v, p in ((t, 3), (height, 2), (accel, 1))
        )
    )


@generator("exd_x_r2j_trig_rate_chain", level=17, presets={"exam": dict(tier="exam")})
def exd_x_r2j_trig_rate_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Maximum rate, first time at a given rate, the quantity's value then, and the rate's own rate of change."""
    check(tier == "exam", "exd_x_r2j_trig_rate_chain: only the exam tier exists")
    a, b, c, k = draw_until(rng, _draw, _accept, what="a stable trig rate chain")
    what, sym, unit, tunit = rng.choice(_SCENARIOS)
    stem = phrase(rng, _STEM, what=what, rule=_rule(a, b, c), unit=unit, tunit=tunit)
    t, height, accel = _chain(a, b, c, k)
    t3 = float(shown(Fraction(t), 3))
    check(abs(b * t - math.acos(k / (a * b))) < 1e-12, "time does not satisfy cos(bt) = k/(ab)")
    check(abs(a * b * math.cos(b * t) - k) < 1e-9, "the rate at the time found is not k")
    check(abs(a * math.sin(b * t) + c - height) < 1e-9, "height at the time found disagrees")
    check(abs(-a * b * b * math.sin(b * t) - accel) < 1e-9, "second derivative at the time found disagrees")

    unit_k = unit if abs(k) == 1 else unit + "s"
    ft_h = f"{a}*sin({b}*part_b) + {c}"
    ft_a = f"-{a * b * b}*sin({b}*part_b)"
    cos_swapped = a * b * b * math.cos(b * t)  # sin read as cos in the second derivative
    parts = [
        number_part(
            prompt=f"Find the greatest rate of change of h, in {unit}s per {tunit}.",
            exact=a * b,
            places=0,
            marks=1,
            working=[f"h'(t) = {a * b}cos({'t' if b == 1 else f'{b}t'}), which has greatest value {a * b}."],
            wrong=[
                (a, "dropped-chain-rule-factor") if b != 1 else (a + c, "evaluated-function-not-derivative"),
                (a + c, "evaluated-function-not-derivative"),
                (a * b * b, "differentiated-twice"),
                (a * b + c, "evaluated-function-not-derivative"),
                (2 * a * b, "gave-range-for-greatest-value"),
                (a + b, "adds-instead-of-multiplies"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            prompt=f"Find the smallest positive value of t at which the rate of change of h is {k} {unit_k} per "
            f"{tunit}, correct to 3 decimal places.",
            exact=Fraction(t),
            places=3,
            marks=2,
            working=[
                f"{a * b}cos({'t' if b == 1 else f'{b}t'}) = {k}, so cos({'t' if b == 1 else f'{b}t'}) = {k}/{a * b}.",
                f"The smallest positive solution has {'t' if b == 1 else f'{b}t'} = arccos({k}/{a * b}), so "
                f"t = {t3:.3f}.",
            ],
            wrong=[
                (Fraction(math.acos(k / (a * b))), "forgot-divide-angle-by-b")
                if b != 1
                else (Fraction(math.acos(-k / (a * b))), "arccos-sign-slip"),
                (Fraction(math.acos(-k / (a * b)) / b), "arccos-sign-slip"),
                (Fraction(2 * math.pi / b - t), "later-root-instead-of-first"),
                (Fraction(math.asin(k / (a * b)) / b), "sine-cosine-swapped"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            prompt=f"Hence find the value of h at that time, in {unit}s, correct to 2 decimal places.",
            exact=Fraction(height),
            places=2,
            marks=2,
            working=[f"h = {a}sin({'t' if b == 1 else f'{b}t'}) + {c} with t = {t3:.3f}.", f"h = {height:.2f}."],
            wrong=[
                (Fraction(k / b + c), "sine-cosine-swapped"),
                (Fraction(height - c), "dropped-constant-term"),
                (Fraction(2 * c - height), "reflected-about-mean"),
                (Fraction(a + c), "took-maximum-instead"),
            ],
            follow_through=ft_h,
            sensible_range=(0, None),
        ),
        number_part(
            prompt=f"Hence find the rate of change of dh/dt at that time (how fast the rate of change is itself "
            f"changing), in {unit}s per {tunit} squared, correct to 1 decimal place.",
            exact=Fraction(accel),
            places=1,
            marks=2,
            working=[
                f"h''(t) = -{a * b * b}sin({'t' if b == 1 else f'{b}t'}).",
                f"At t = {t3:.3f}: h'' = {accel:.1f}.",
            ],
            wrong=[
                (Fraction(-accel), "dropped-negative-from-derivative"),
                (Fraction(cos_swapped * -1), "sine-cosine-swapped"),
                (Fraction(accel / b), "dropped-chain-rule-factor"),
                (Fraction(-a * b * b), "ignores-trig-evaluation"),
            ],
            follow_through=ft_a,
        ),
    ]
    return dict(text=stem, parts=parts)
