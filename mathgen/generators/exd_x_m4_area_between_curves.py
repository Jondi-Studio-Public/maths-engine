"""Exam tier for "calculate the area between curves in simple cases": a three-part chain.

A parabola and a line (or two parabolas) meet at two points. Part (a) finds the x-coordinate of the left-hand
point of intersection, part (b) the right-hand one, and part (c) uses both to find the enclosed area. The practice
tiers hand over the limits of integration; here the learner has to find them by equating the two curves, and then
choose to integrate the difference between the curves between those limits. Part (c) carries `follow_through`, so a
learner whose limits are wrong still earns the area marks for integrating correctly between their own limits.

## Construction: draw the difference first

The difference of the two curves is built as `s * (a x - b)(c x - d)` with `a, c` in {1, 2}, so the intersections
are at `b/a` and `d/c`, both multiples of one half and always exact decimals. The lower curve `g` is drawn first
(a line or a parabola) and the upper curve `f` is `g` plus that difference. The enclosed area is
`|s a c| (q - p)^3 / 6` for intersections `p < q`; it is rounded once, at display, and kept away from a rounding
boundary.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    VerificationError,
    attempts,
    check,
    exact_decimal_string,
    generator,
    join_terms,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
    stated_precision_tolerance,
)

_MARGIN = Fraction(1, 1000)

_STEMS = (
    "The curves y = {f} and y = {g} intersect at two points, A and B, with A to the left of B.",
    "The graphs of y = {f} and y = {g} cross at two points, A and B. A lies to the left of B.",
    "A region is enclosed between the curve y = {f} and the curve y = {g}, which meet at points A and B "
    "(A is the left-hand point).",
)


_SETTINGS = (
    "The region is a garden bed whose axes are marked in metres, and topsoil costs ${price} per square metre.",
    "The region is a pond liner laid out on axes marked in metres, priced at ${price} per square metre.",
    "The region is a concrete slab drawn on axes marked in metres; concreting costs ${price} per square metre.",
)


def _poly(c2: int, c1: int, c0: int) -> str:
    return join_terms([(c2, "x^2"), (c1, "x"), (c0, "")])


def _fmt_root(value: Fraction) -> str:
    return exact_decimal_string(value, 1)


def _area_from_limits(scale: int, lo: Fraction, hi: Fraction) -> Fraction:
    return abs(scale) * (hi - lo) ** 3 / 6


def _draw(rng: random.Random):
    for _ in attempts(1000, "curves with a clear area"):
        a, c = rng.choice([1, 2]), rng.choice([1, 2])
        b, d = rng.randint(-9, 9), rng.randint(-9, 9)
        p, q = sorted([Fraction(b, a), Fraction(d, c)])
        if not 1 <= q - p <= 6:
            continue
        sign = rng.choice([1, -1])
        # D(x) = sign * (a x - b)(c x - d), expanded
        d2, d1, d0 = sign * a * c, -sign * (a * d + c * b), sign * b * d
        if rng.random() < 0.5:
            u, v, w = 0, rng.randint(-4, 4), rng.randint(-6, 6)  # g is a line
        else:
            u, v, w = rng.choice([-2, -1, 1, 2]), rng.randint(-5, 5), rng.randint(-6, 6)
        f2, f1, f0 = u + d2, v + d1, w + d0
        if f2 == 0 or (u == 0 and v == 0):
            continue
        if u != 0 and (f2, f1, f0) == (u, v, w):
            continue
        area = _area_from_limits(a * c, p, q)
        shown = round_half_up(area, 2)
        scaled = area * 100
        if abs((scaled % 1) - Fraction(1, 2)) <= _MARGIN * 100:
            continue
        text = stated_precision_string(shown, 2)
        if text in {str(b), str(d), str(a), str(c)}:
            continue
        return a, c, b, d, p, q, sign, (f2, f1, f0), (u, v, w), area, shown
    raise VerificationError("unreachable")


@generator("exd_x_m4_area_between_curves", level=11, presets={"exam": dict(tier="exam")})
def exd_x_m4_area_between_curves(rng: random.Random, tier: str = "exam") -> dict:
    """Two intersection x-coordinates, then the area between the curves. See module docstring."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    a, c, b, d, p, q, sign, (f2, f1, f0), (u, v, w), area, shown = _draw(rng)
    scale = a * c
    f_text, g_text = _poly(f2, f1, f0), _poly(u, v, w)
    price = rng.randint(8, 95)
    stem = rng.choice(_STEMS).format(f=f_text, g=g_text) + " " + rng.choice(_SETTINGS).format(price=price)
    cost = shown * price
    cost_text = stated_precision_string(cost, 2)
    check(cost_text not in stem and cost != shown, "cost text must not appear in the stem")
    area_text = stated_precision_string(shown, 2)
    p_text, q_text = _fmt_root(p), _fmt_root(q)
    check(Fraction(area_text) == shown and area_text not in stem, "area text must not appear in the stem")

    # The raw constants of the two linear factors, as a learner who forgets to divide by the coefficient reads them.
    raw = {Fraction(b, a): b, Fraction(d, c): d}
    raw_p, raw_q = raw.get(p, b), raw.get(q, d)

    def lin_integral(lo: Fraction, hi: Fraction) -> Fraction:
        """Integral of the difference f - g between lo and hi."""
        e2, e1, e0 = sign * scale, -sign * (a * d + c * b), sign * b * d

        def prim(x):
            return Fraction(e2, 3) * x**3 + Fraction(e1, 2) * x**2 + e0 * x

        return prim(hi) - prim(lo)

    def money(value: Fraction) -> str:
        return stated_precision_string(round_half_up(value, 2), 2)

    def root_slip(value: Fraction, key: Fraction, tag: str | None) -> Distractor:
        """A wrong root shown to the places of the key (a whole number or one decimal place), never a
        different format."""
        places = 0 if key.denominator == 1 else 1
        shown_value = round_half_up(abs(value), places) * (1 if value >= 0 else -1)
        return Distractor(stated_precision_string(shown_value, places), tag, shown_value)

    a_candidates = [
        root_slip(-p, p, "sign-flip-both-roots"),
        root_slip(q, p, None),
        root_slip(-q, p, None),  # the other root, with its sign flipped as well
        root_slip(q - p, p, None),  # the distance between the crossing points
        root_slip(p + q, p, None),  # the two roots added
    ]
    b_candidates = [
        root_slip(-q, q, "sign-flip-both-roots"),
        root_slip(p, q, None),
        root_slip(-p, q, None),  # the other root, with its sign flipped as well
        root_slip(q - p, q, None),  # the distance between the crossing points
        root_slip(p + q, q, None),  # the two roots added
    ]
    if scale > 1:
        a_candidates.append(root_slip(Fraction(raw_p), p, "forgot-to-divide-by-coefficient"))
        b_candidates.append(root_slip(Fraction(raw_q), q, "forgot-to-divide-by-coefficient"))
    width = q - p
    c_wrong = [
        -area,  # the integral taken the wrong way round, a negative area
        abs(lin_integral(Fraction(0), q)),  # lower limit taken as 0
        width**3 / 6,  # the leading coefficient dropped
        abs(scale) * width**3 / 3,  # 1/3 in place of 1/6
        abs(scale) * width**2 / 2,  # width squared
    ]
    c_candidates = [Distractor(money(v), None, round_half_up(v, 2)) for v in c_wrong]
    tol = stated_precision_tolerance(2)
    a_distractors = pick_distractors(p_text, a_candidates, 3, tolerance=Fraction(1, 100))
    b_distractors = pick_distractors(q_text, b_candidates, 3, tolerance=Fraction(1, 100))
    c_distractors = pick_distractors(area_text, c_candidates, 3, tolerance=tol)
    check(min(len(a_distractors), len(b_distractors)) >= 1 and len(c_distractors) >= 2, "not enough distractors")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the x-coordinate of A.",
                answer_kind="number",
                answer=numeric_answer(p),
                answer_text=p_text,
                answer_latex=latex_answer(p_text),
                marks=1,
                working=[f"Equate the curves: {f_text} = {g_text}.", f"The x-coordinate of A is {p_text}."],
                distractors=a_distractors,
            ),
            dict(
                prompt="Find the x-coordinate of B.",
                answer_kind="number",
                answer=numeric_answer(q),
                answer_text=q_text,
                answer_latex=latex_answer(q_text),
                marks=1,
                working=[f"The x-coordinate of B is {q_text}."],
                distractors=b_distractors,
            ),
            dict(
                prompt="Hence find the area of the region enclosed between the two curves, in square metres, "
                "correct to 2 decimal places.",
                answer_kind="number",
                answer=numeric_answer(shown),
                answer_text=area_text,
                answer_latex=latex_answer(area_text),
                marks=3,
                sensible_range=(0, None),
                follow_through=f"{abs(scale)}*Abs(part_b-part_a)**3/6",
                working=[
                    "Area = integral of (upper curve - lower curve) from A to B.",
                    f"= {abs(scale)} x ({q_text} - ({p_text}))^3 / 6 = {area_text} (2 d.p.).",
                ],
                distractors=c_distractors,
            ),
            dict(
                prompt="Using your area from (c), find the cost of the region, in dollars to the nearest cent.",
                answer_kind="number",
                answer=numeric_answer(cost),
                answer_text=cost_text,
                answer_latex=latex_answer(cost_text),
                marks=1,
                sensible_range=(0, None),
                follow_through=f"{price}*part_c",
                working=[f"Cost = {area_text} x {price} = ${cost_text}."],
                distractors=pick_distractors(
                    cost_text,
                    [
                        Distractor(money(Fraction(price) / shown), None, round_half_up(Fraction(price) / shown, 2)),
                        Distractor(money(cost / 2), None, round_half_up(cost / 2, 2)),
                        Distractor(money(shown + price), None, round_half_up(shown + price, 2)),
                    ],
                    3,
                    tolerance=tol,
                ),
            ),
        ],
    )
