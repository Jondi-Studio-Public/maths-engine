"""Exam-tier source for `f10-trig-ratios-similarity`: two right triangles that share a side.

A loading platform is reached by two ramps at different angles. The learner is told the first ramp's length and
angle and only the second ramp's angle. The question is a chain:

(a) the height of the platform, from the first ramp (sine, the known side being the hypotenuse);
(b) the length of the second ramp, from that height (sine again, now the unknown side is the hypotenuse, so the
    ratio is divided into the known side);
(c) the distance between the feet of the two ramps, which needs the horizontal run of each (cosine) and then a
    sum or a difference depending on which sides of the platform the ramps are on.

Built backwards from a ramp length to one decimal place and whole-number angles. Every answer is stated to one
decimal place, and a draw is kept only if working through the rounded earlier answers gives the same rounded
later answer as working from the exact values, so a learner who rounds as asked is never marked down.
"""

from __future__ import annotations

import random
from fractions import Fraction
from functools import lru_cache

import sympy

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    draw_until,
    generator,
    latex_answer,
    numeric_answer,
    round_half_up,
    stated_precision_string,
)

_SETTINGS = [
    "A loading platform is raised above level ground.",
    "A raised stage at a school hall is built above the floor.",
    "A verandah floor stands above the garden.",
]
_FIRST = [
    "A ramp {L} m long rises at {theta}° to the horizontal up to it.",
    "The first ramp is {L} m long and makes an angle of {theta}° with the horizontal as it rises to the top.",
    "One ramp, {L} m long, climbs to the top at {theta}° above the horizontal.",
]
_SECOND_OPPOSITE = [
    "A second ramp, built on the opposite side, reaches the same level at a gentler angle of {phi}° to the horizontal.",
    "A gentler second ramp, at {phi}° to the horizontal, is built on the other side and reaches the same level.",
]
_SECOND_SAME = [
    "A second ramp, built on the same side and starting further out, reaches the same level at a gentler angle of "
    "{phi}° to the horizontal.",
    "A gentler second ramp, at {phi}° to the horizontal, starts further out along the same side and reaches the "
    "same level.",
]


@lru_cache(maxsize=None)
def _ratio(name: str, degrees: int) -> Fraction:
    fn = {"sin": sympy.sin, "cos": sympy.cos}[name]
    return Fraction(str(fn(sympy.pi * degrees / 180).evalf(40)))


def _dp1(value: Fraction) -> Fraction:
    return round_half_up(value, 1)


def _clear(value: Fraction) -> bool:
    """True when `value` is at least 0.01 from a one-decimal-place rounding boundary."""
    return abs((value * 10) % 1 - Fraction(1, 2)) > Fraction(1, 10)


def _draw(rng: random.Random):
    L = Fraction(rng.randint(40, 100), 10)
    theta = rng.randint(25, 48)
    same = rng.random() < 0.5
    phi = rng.randint(14, theta - 5)
    return L, theta, phi, same


def _consistent(t) -> bool:
    L, theta, phi, same = t
    if phi >= theta:
        return False
    h = L * _ratio("sin", theta)
    L2 = h / _ratio("sin", phi)
    run1, run2 = L * _ratio("cos", theta), L2 * _ratio("cos", phi)
    far = run2 - run1 if same else run1 + run2
    if far < 1 or not all(_clear(v) for v in (h, L2, far)):
        return False
    # Working through the answers rounded as asked must give the same rounded results.
    h_r = _dp1(h)
    L2_r = _dp1(h_r / _ratio("sin", phi))
    far_r = _dp1(L2_r * _ratio("cos", phi) - run1) if same else _dp1(L2_r * _ratio("cos", phi) + run1)
    return L2_r == _dp1(L2) and far_r == _dp1(far)


def _slips(key: Fraction, slips: list[tuple[Fraction, str]]) -> list[Distractor]:
    out, seen = [], {key}
    for value, tag in slips:
        v = _dp1(value)
        if v <= 0 or any(abs(v - s) <= Fraction(1, 10) for s in seen):
            continue
        seen.add(v)
        out.append(Distractor(stated_precision_string(v, 1), tag, v))
    return out


def _wrong_answers(L, theta, phi, same):
    """The keys and the tagged wrong answers for one draw: ((a, b, c), (a_dist, b_dist, c_dist))."""
    h = L * _ratio("sin", theta)
    L2 = h / _ratio("sin", phi)
    run1, run2 = L * _ratio("cos", theta), L2 * _ratio("cos", phi)
    far = run2 - run1 if same else run1 + run2
    keys = (_dp1(h), _dp1(L2), _dp1(far))
    a_dist = _slips(
        keys[0],
        [
            (L * _ratio("cos", theta), "swaps-opposite-and-adjacent"),
            (L / _ratio("sin", theta), "multiplies-instead-of-divides"),
        ],
    )
    b_dist = _slips(
        keys[1],
        [
            (h * _ratio("sin", phi), "multiplies-instead-of-divides"),
            (h / _ratio("cos", phi), "swaps-opposite-and-adjacent"),
        ],
    )
    other = run1 + run2 if same else abs(run1 - run2)
    c_dist = _slips(
        keys[2],
        [(other, "adds-where-subtracts-needed"), (run2, "used-one-ramp-only"), (run1, "used-one-ramp-only")],
    )
    return keys, (a_dist, b_dist, c_dist)


def _enough(L, theta, phi, same) -> bool:
    return all(len(d) >= 2 for d in _wrong_answers(L, theta, phi, same)[1])


@generator("exd_x_u7_two_ramps_chain", level=9, presets={"exam": dict(tier="exam")})
def exd_x_u7_two_ramps_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: platform height, second ramp length, then the gap between the ramps' feet."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(300, "a pair of ramps with enough distinct wrong answers"):
        L, theta, phi, same = draw_until(rng, _draw, _consistent, what="a pair of ramps clear of rounding boundaries")
        if _enough(L, theta, phi, same):
            break
    h = L * _ratio("sin", theta)
    L2 = h / _ratio("sin", phi)
    run1, run2 = L * _ratio("cos", theta), L2 * _ratio("cos", phi)
    far = run2 - run1 if same else run1 + run2
    h_k, L2_k, far_k = _dp1(h), _dp1(L2), _dp1(far)
    # Independent float cross-check of the three keys.
    import math

    fh = float(L) * math.sin(math.radians(theta))
    fl2 = fh / math.sin(math.radians(phi))
    ff = fl2 * math.cos(math.radians(phi)) + (-1 if same else 1) * float(L) * math.cos(math.radians(theta))
    check(abs(fh - float(h_k)) < 0.051 and abs(fl2 - float(L2_k)) < 0.051 and abs(ff - float(far_k)) < 0.051, "keys")

    second = rng.choice(_SECOND_SAME if same else _SECOND_OPPOSITE)
    stem = " ".join([rng.choice(_SETTINGS), rng.choice(_FIRST), second]).format(
        L=stated_precision_string(L, 1), theta=theta, phi=phi
    )
    a_text, b_text, c_text = (stated_precision_string(v, 1) for v in (h_k, L2_k, far_k))

    (_, _, _), (a_dist, b_dist, c_dist) = _wrong_answers(L, theta, phi, same)

    sign = "−" if same else "+"
    ft_c = f"part_b*cos({phi}*pi/180) {'-' if same else '+'} {L}*cos({theta}*pi/180)"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the height of the top above the ground, in metres, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(h_k),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"The ramp is the hypotenuse: height = {stated_precision_string(L, 1)} × sin {theta}° = {a_text} m."
                ],
                distractors=a_dist,
            ),
            dict(
                prompt="Hence find the length of the second ramp, in metres, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(L2_k),
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a/sin({phi}*pi/180)",
                working=[
                    f"Now the ramp is the hypotenuse: length = height ÷ sin {phi}° = {a_text} ÷ sin {phi}° = {b_text} m."
                ],
                distractors=b_dist,
            ),
            dict(
                prompt="Find the distance between the feet of the two ramps, in metres, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(far_k),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=ft_c,
                working=[
                    f"Horizontal run of the first ramp = {stated_precision_string(L, 1)} × cos {theta}°; "
                    f"of the second = {b_text} × cos {phi}°.",
                    f"Distance = second run {sign} first run = {c_text} m.",
                ],
                distractors=c_dist,
            ),
        ],
    )
