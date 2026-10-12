"""Exam source: estimate a three-factor cost by rounding to one significant figure, then judge the estimate.

The practice tiers apply one named rounding rule to the operands of a sum. The exam question asks for a whole
estimation exercise, as a short-answer paper would:

(a) estimate a cost that is the product of three measured quantities, rounding each to 1 significant figure;
(b) work out the exact cost to the nearest dollar;
(c) say how far the estimate is from the exact cost as a percentage of the exact cost.

Part (c) uses the answers to (a) and (b), so it carries follow-through. Every figure is built from exact
`Fraction` arithmetic; a draw is kept only when no operand sits on a rounding tie at its first significant
figure and the final percentage is clear of a rounding boundary at one decimal place.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, generator, phrase, round_half_up
from mathgen.generators.exd_x_n1_support import clear_of_boundary, money, number_part, show

# (stem pattern, a, b, c) -- the three factors appear in the stem in this order.
_AREA = (
    "A community hall has a rectangular floor measuring {a} m by {b} m. The floor is to be covered with timber "
    "at ${c} per square metre.",
    "A rectangular courtyard measures {a} m by {b} m. Pavers for the courtyard cost ${c} per square metre laid.",
    "A council is turfing a rectangular oval measuring {a} m by {b} m, with turf priced at ${c} per square metre.",
)
_COURIER = (
    "A courier firm makes {a} deliveries a day for {b} days, and each delivery costs ${c} to run.",
    "A bakery sends out {a} orders a day for {b} days, and each order costs ${c} to pack and post.",
    "A school canteen serves {a} lunch orders a day for {b} days; each order costs ${c} to prepare.",
)


def _sig1_exponent(value: Fraction) -> int:
    for e in range(-6, 8):
        if Fraction(10) ** e <= value < Fraction(10) ** (e + 1):
            return e
    raise ValueError(value)


def _round1(value: Fraction) -> Fraction:
    unit = Fraction(10) ** _sig1_exponent(value)
    return round_half_up(value / unit, 0) * unit


def _trunc1(value: Fraction) -> Fraction:
    unit = Fraction(10) ** _sig1_exponent(value)
    return Fraction((value / unit).numerator // (value / unit).denominator) * unit


def _ceil1(value: Fraction) -> Fraction:
    unit = Fraction(10) ** _sig1_exponent(value)
    ratio = value / unit
    return Fraction(-((-ratio.numerator) // ratio.denominator)) * unit


def _round2(value: Fraction) -> Fraction:
    unit = Fraction(10) ** (_sig1_exponent(value) - 1)
    return round_half_up(value / unit, 0) * unit


def _tie(value: Fraction) -> bool:
    unit = Fraction(10) ** _sig1_exponent(value)
    return (value / unit) % 1 == Fraction(1, 2)


def _plain(value: Fraction) -> str:
    """A quantity as written in the stem: whole numbers bare, otherwise with its decimals."""
    if value.denominator == 1:
        return f"{int(value):,}" if value >= 1000 else str(int(value))
    places = 2 if (value * 10).denominator != 1 else 1
    return show(value, places)


@generator("exd_x_n1_estimate_then_check", level=11, presets={"exam": dict(tier="exam")})
def exd_x_n1_estimate_then_check(rng, tier="exam"):
    for _ in attempts(1000, "an estimation draw clear of ties and rounding boundaries"):
        if rng.random() < 0.55:
            patterns = _AREA
            a = Fraction(rng.randint(120, 989), 10)
            b = Fraction(rng.randint(120, 989), 10)
            c = Fraction(rng.randint(2500, 9899), 100)
        else:
            patterns = _COURIER
            a = Fraction(rng.randint(12, 98))
            b = Fraction(rng.randint(12, 98))
            c = Fraction(rng.randint(250, 989), 100)
        if any(_tie(v) for v in (a, b, c)):
            continue
        ea, eb, ec = _round1(a), _round1(b), _round1(c)
        estimate = ea * eb * ec
        exact = a * b * c
        exact_whole = round_half_up(exact, 0)
        gap = abs(estimate - exact_whole) / exact_whole * 100
        # the learner's route (from the rounded exact cost) and the exact route must agree at 1 d.p.
        gap_exact = abs(estimate - exact) / exact * 100
        if not (Fraction(1, 2) <= gap <= 40):
            continue
        if not (clear_of_boundary(gap, 1) and clear_of_boundary(gap_exact, 1)):
            continue
        if show(gap, 1) != show(gap_exact, 1):
            continue
        if estimate == exact_whole:
            continue
        break

    stem = phrase(rng, patterns, a=_plain(a), b=_plain(b), c=show(c, 2))
    trunc = _trunc1(a) * _trunc1(b) * _trunc1(c)
    two_only = ea * eb * c  # the third factor left unrounded
    to_whole = round_half_up(a, 0) * round_half_up(b, 0) * round_half_up(c, 0)

    part_a = number_part(
        "Estimate the cost by rounding each of the three numbers to 1 significant figure. Give your estimate "
        "in dollars.",
        estimate,
        0,
        2,
        [
            f"Round to 1 s.f.: {_plain(a)} → {_plain(ea)}, {_plain(b)} → {_plain(eb)}, {show(c, 2)} → {_plain(ec)}.",
            f"Estimate = {_plain(ea)} × {_plain(eb)} × {_plain(ec)} = {_plain(estimate)}.",
        ],
        [
            (trunc, None),
            (to_whole, None),
            (two_only, None),
            (a * eb * ec, None),
            (ea * b * ec, None),
            (exact_whole, None),
            (_ceil1(a) * _ceil1(b) * _ceil1(c), None),
            (_round2(a) * _round2(b) * _round2(c), None),
        ],
        sensible_range=(0, None),
    )
    part_b = number_part(
        "Work out the exact cost, correct to the nearest dollar.",
        exact_whole,
        0,
        1,
        [f"{_plain(a)} × {_plain(b)} × {show(c, 2)} = {money(exact, 4)}, so about ${_plain(exact_whole)}."],
        [
            (estimate, None),
            (a * b * ec, None),
            (trunc, None),
            (to_whole, None),
            (two_only, None),
            (ea * b * c, None),
            (exact * 10, None),
            (exact / 10, None),
        ],
        sensible_range=(0, None),
    )
    part_c = number_part(
        "Using your answers to (a) and (b), find the difference between the estimate and the exact cost as a "
        "percentage of the exact cost, correct to 1 decimal place.",
        gap,
        1,
        2,
        [
            f"Difference = |{_plain(estimate)} - {_plain(exact_whole)}| = {_plain(abs(estimate - exact_whole))}.",
            f"{_plain(abs(estimate - exact_whole))} ÷ {_plain(exact_whole)} × 100 = {show(gap, 1)}%.",
        ],
        [
            (abs(estimate - exact_whole) / estimate * 100, None),
            (abs(estimate - exact_whole), "gave-change-not-percent"),
            (abs(estimate - exact_whole) / exact_whole, None),
        ],
        follow_through="Abs(part_a - part_b)/part_b*100",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c])
