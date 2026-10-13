"""Exam chain on a tangent and a secant drawn to a circle from an outside point.

A circle has centre O and radius r. From an outside point M, a tangent touches the circle at T and a secant meets it
at B then A (B between M and A). The stem gives r, TM and BM; four parts follow:

(a) AM, from AM x BM = TM^2;
(b) hence the chord AB = AM - BM;
(c) hence the distance from O to the chord, sqrt(r^2 - (AB / 2)^2);
(d) hence the angle between the secant and MO, arctan(distance / (BM + AB / 2)), to 1 decimal place.

Built backwards from a Pythagorean triple (half-chord c, distance h, radius r) so that (c) is whole. The lengths
BM and TM satisfy (BM + c)^2 - TM^2 = c^2, so BM + c = (u + v) / 2 and TM = (v - u) / 2 for a factor pair
u v = c^2 of the same parity.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction
from functools import lru_cache

from mathgen.core import check, draw_until, generator, phrase

from .exd_x_r2j_support import number_part


@lru_cache(maxsize=None)
def _configurations() -> tuple:
    """Every (c, h, r, m, t): half-chord, distance to the chord, radius, BM and TM, all whole."""
    found = []
    for r in range(5, 101):
        for c in range(2, r):
            h2 = r * r - c * c
            h = math.isqrt(h2)
            if h * h != h2 or h < 2:
                continue
            for u in range(1, c):
                if (c * c) % u:
                    continue
                v = c * c // u
                if (u + v) % 2:
                    continue
                m, t = (u + v) // 2 - c, (v - u) // 2
                if 2 <= m <= 150 and t <= 250:
                    found.append((c, h, r, m, t))
    return tuple(found)


_STEM = [
    "A circle has centre O and radius {r} cm. From a point M outside the circle, a tangent touches the circle at T "
    "and a secant meets the circle at B and then at A, so that B lies between M and A. TM = {t} cm and "
    "BM = {m} cm.",
    "M is a point outside a circle with centre O and radius {r} cm. The tangent from M touches the circle at T, "
    "and a secant from M meets the circle first at B and then at A. TM is {t} cm and BM is {m} cm.",
    "Two straight lines are drawn from a point M outside a circle of radius {r} cm and centre O. One touches the "
    "circle at T, with TM = {t} cm. The other cuts the circle at B and then at A, with BM = {m} cm.",
]


def _degrees(x: float) -> Fraction:
    return Fraction(math.degrees(x))


@generator("exd_x_r2j_tangent_secant_chain", level=14, presets={"exam": dict(tier="exam")})
def exd_x_r2j_tangent_secant_chain(rng: random.Random, tier: str = "exam") -> dict:
    """AM from the tangent-secant relation, then the chord, its distance from the centre, and an angle at M."""
    check(tier == "exam", "exd_x_r2j_tangent_secant_chain: only the exam tier exists")
    c, h, r, m, t = draw_until(
        rng,
        lambda g: g.choice(_configurations()),
        lambda cfg: abs((math.degrees(math.atan2(cfg[1], cfg[3] + cfg[0])) * 10) % 1 - 0.5) > 0.03,
        what="a configuration whose angle rounds clearly",
    )
    angle = math.atan2(h, m + c)
    am = t * t // m
    ab = am - m
    check(am * m == t * t and ab == 2 * c, "tangent-secant relation does not hold")
    check(abs(math.hypot(m + c, h) ** 2 - (r * r + t * t)) < 1e-9, "OM does not satisfy OM^2 = r^2 + TM^2")
    stem = phrase(rng, _STEM, r=r, t=t, m=m)

    ft_ab = f"part_a - {m}"
    ft_h = f"sqrt({r}**2 - (part_b/2)**2)"
    ft_angle = f"atan(part_c/({m} + part_b/2))*180/pi"
    parts = [
        number_part(
            prompt="Find the length of AM, in cm.",
            exact=am,
            places=0,
            marks=2,
            working=[f"AM x BM = TM², so AM = {t}² / {m} = {am} cm."],
            wrong=[
                (t * t, "power-of-point-not-divided"),
                (ab, "chord-for-secant"),
                (t * m, "multiplies-instead-of-dividing"),
                (t + m, "adds-instead-of-multiplies"),
            ],
            sensible_range=(0, None),
        ),
        number_part(
            prompt="Hence find the length of the chord AB, in cm.",
            exact=ab,
            places=0,
            marks=1,
            working=[f"AB = AM - BM = {am} - {m} = {ab} cm."],
            wrong=[
                (am, "whole-secant-when-part-asked"),
                (am + m, "added-instead-of-subtracted"),
                (Fraction(ab, 2), "half-chord-for-chord"),
            ],
            follow_through=ft_ab,
            sensible_range=(0, None),
        ),
        number_part(
            prompt=phrase(
                rng,
                [
                    "Hence find the perpendicular distance from O to the chord AB, in cm.",
                    "Using your answer to (b), find how far the chord AB is from the centre O, in cm.",
                ],
            ),
            exact=h,
            places=0,
            marks=2,
            working=[f"The perpendicular from O bisects AB, so each half is {c} cm.", f"d = √({r}² - {c}²) = {h} cm."],
            wrong=[
                (r - c, "linear-instead-of-pythagoras"),
                (c, "half-chord-for-chord"),
                (r * r - c * c, "no-square-root"),
                (math.isqrt(r * r + c * c), "added-squares-for-leg"),
            ],
            follow_through=ft_h,
            sensible_range=(0, None),
        ),
        number_part(
            prompt="Hence find the angle between the secant MA and the line MO, in degrees, correct to 1 decimal "
            "place.",
            exact=_degrees(angle),
            places=1,
            marks=2,
            working=[
                f"The foot of the perpendicular from O is {m} + {c} = {m + c} cm from M along MA.",
                f"tan(angle) = {h} / {m + c}, so the angle is {math.degrees(angle):.1f}°.",
            ],
            wrong=[
                (_degrees(math.atan2(m + c, h)), "inverted-ratio"),
                (_degrees(math.atan2(h, m)), "ignored-half-chord"),
                (_degrees(math.atan2(h, m + 2 * c)), "chord-for-half-chord"),
                *([(_degrees(math.asin(h / (m + c))), "wrong-trig-ratio")] if h < m + c else []),
            ],
            follow_through=ft_angle,
            sensible_range=(0, 90),
        ),
    ]
    return dict(text=stem, parts=parts)
