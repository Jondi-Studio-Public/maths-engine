"""Exam-tier source for `significant-figures`: reporting a calculated area to the precision of its data.

The practice tiers round a number they hand over to a stated number of significant figures. In a measurement
exam the number of figures is not stated: it comes from the data. Two measurements with different numbers of
significant figures are multiplied, and the learner must pick the right count and round.

    (a) the exact product of the two measurements, every digit written down,
    (b) how many significant figures the result should have (the count of the less precise measurement),
    (c) the product rounded to that many significant figures.

Part (c) follows through from the learner's own (a) and (b): their product, rounded half-up to their count.

Each measurement is built from a digit string with no trailing zero, so its count of significant figures is not
open to doubt. The product is kept below ``10 ** k`` so the rounded answer always has a decimal point and no
trailing zeros that stand only for magnitude.
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    attempts,
    check,
    fixed_point_string,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
)

# (item, unit, integer digits of the length, its allowed leading digits, integer digits of the width, its allowed
# leading digits).
# 0 integer digits means the value is 0.d..., so a decking strip is a few metres by about a tenth of a metre
# and a steel plate is tens to hundreds of millimetres on each side.
_ITEMS = (
    ("pane of glass", "cm", (2, 3), range(1, 10), (2,), range(1, 10)),
    ("slate paver", "cm", (2,), range(1, 10), (2,), range(1, 10)),
    ("sheet of aluminium", "m", (1,), range(1, 10), (0, 1), range(1, 10)),
    ("solar panel", "m", (1,), (1, 2), (0, 1), range(1, 10)),
    ("steel plate", "mm", (2, 3), range(1, 10), (2,), range(1, 10)),
    ("strip of timber decking", "m", (1,), range(2, 6), (0,), (1, 2)),
)


def _measurement(rng, sig_figs: int, integer_digits: int, leading=range(1, 10)) -> tuple[int, int]:
    """(digits, decimal places) of a measurement with that many significant figures and no trailing zero."""
    first = rng.choice(tuple(leading))
    middle = [rng.randint(0, 9) for _ in range(sig_figs - 2)]
    last = rng.randint(1, 9)
    digits = int("".join(map(str, [first, *middle, last])))
    return digits, sig_figs - integer_digits


def _exponent(value: Fraction) -> int:
    """The e with 10**e <= value < 10**(e + 1)."""
    e = 0
    ten = Fraction(10)
    while value >= ten ** (e + 1):
        e += 1
    while value < ten**e:
        e -= 1
    return e


def _round_sig(value: Fraction, figures: int, *, truncate: bool = False) -> Fraction:
    unit = Fraction(10) ** (_exponent(value) - figures + 1)
    scaled = value / unit
    return Fraction(math.floor(scaled) if truncate else math.floor(scaled + Fraction(1, 2))) * unit


def _text(value: Fraction, places: int) -> str:
    scaled = value * 10**places
    check(scaled.denominator == 1, "value is not on the display grid")
    return fixed_point_string(scaled.numerator, places)


@generator("exd_x_n4_sig_fig_product", level=9, presets={"exam": {"tier": "exam"}})
def exd_x_n4_sig_fig_product(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "two measurements whose product needs rounding cleanly"):
        sf_u, sf_v = rng.sample((2, 3, 4), 2)
        item, unit_name, length_digits, length_leading, width_digits, width_leading = rng.choice(_ITEMS)
        du, pu = _measurement(rng, sf_u, rng.choice(length_digits), length_leading)
        dv, pv = _measurement(rng, sf_v, rng.choice(width_digits), width_leading)
        if pu < 1 or pv < 1:
            continue
        u, v = Fraction(du, 10**pu), Fraction(dv, 10**pv)
        if u < v:
            continue  # a length is never shorter than the width
        product = u * v
        k = min(sf_u, sf_v)
        e = _exponent(product)
        unit = Fraction(10) ** (e - k + 1)
        if unit > 1 or (product / unit).denominator == 1 or (product / unit) % 1 == Fraction(1, 2):
            continue  # want decimal places in the answer, digits left to drop, and no tie
        places = max(0, -(e - k + 1))
        exact_places = pu + pv
        rounded = _round_sig(product, k)
        shown_u, shown_v = _text(u, pu), _text(v, pv)
        stem = (
            f"A student measures a rectangular {item}. Its length is {shown_u} {unit_name} "
            f"and its width is {shown_v} {unit_name}."
        )
        prompts = (
            "Calculate the area, writing down every digit of the exact product.",
            "The area can be no more precise than the less precise measurement. "
            "How many significant figures should it be given to?",
            "Hence write the area to that number of significant figures.",
        )
        a_text, b_text, c_text = _text(product, exact_places), str(k), _text(rounded, places)
        shown_numbers = {Fraction(t) for t in (shown_u, shown_v, b_text, a_text, c_text)}
        if len(shown_numbers) < 5:
            continue

        def zeros_after_point(x: Fraction, places_: int) -> int:
            """Zeros between the decimal point and the first non-zero digit, counted from the displayed text."""
            return places_ - len(str(int(x * 10**places_)))

        padded = min(sf_u + max(0, zeros_after_point(u, pu)), sf_v + max(0, zeros_after_point(v, pv)))

        # A number answer's distractors must show the answer's decimal places, or the format gives the key away. A
        # slip is shown at those places only when that leaves its value alone (3.8688 as 3.86880), never rounded to fit.
        def shown(value: Fraction, places_: int, tag: str) -> list[Distractor]:
            scaled = value * 10**places_
            return [Distractor(_text(value, places_), tag, value)] if scaled.denominator == 1 else []

        cands_a = [
            *shown(product * 10, exact_places, "decimal-point-one-place-too-far-right"),
            *shown(product / 10, exact_places, "decimal-point-one-place-too-far-left"),
            *shown(u + v, exact_places, "added-instead-of-multiplied"),
            *shown(2 * (u + v), exact_places, "perimeter-for-area"),
        ]
        cands_b = [
            Distractor(str(max(sf_u, sf_v)), "used-more-precise-measurement", max(sf_u, sf_v)),
            Distractor(str(min(pu, pv)), "counts-decimal-places-not-sig-figs", min(pu, pv)),
            Distractor(str(padded), "counts-leading-zeros", padded),
        ]
        by_places = round_half_up(product, k)
        truncated = _round_sig(product, k, truncate=True)
        fewer = _round_sig(product, k - 1)
        cands_c = [
            *(shown(by_places, places, "counts-decimal-places-not-sig-figs") if by_places > 0 else []),
            *shown(truncated, places, "truncated-not-rounded"),
            *shown(fewer, places, "counts-leading-zeros" if product < 1 else "rounds-to-one-fewer-sig-fig"),
        ]
        tolerance = Fraction(1, 2 * 10**places) if places else Fraction(0)
        d_a = pick_distractors(a_text, cands_a, 3, tolerance=Fraction(1, 2 * 10**exact_places))
        d_b = pick_distractors(b_text, cands_b, 3)
        d_c = pick_distractors(c_text, cands_c, 3, tolerance=tolerance)
        if min(len(d_a), len(d_b), len(d_c)) >= 2:
            break

    # Independent route: the product from the scaled integers, the count from the digit strings.
    check(product == Fraction(du * dv, 10 ** (pu + pv)), "product disagrees with the scaled integers")
    check(k == min(len(str(du)), len(str(dv))), "significant figure count disagrees with the digit strings")
    check(len(str(int(rounded / unit))) == k, "rounded value does not have k significant figures")
    check(abs(rounded - product) <= unit / 2, "rounded value is not within half a unit")

    follow = "floor(part_a/10**(floor(log(part_a)/log(10))-part_b+1)+1/2)*10**(floor(log(part_a)/log(10))-part_b+1)"
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=prompts[0],
                answer_kind="number",
                answer=numeric_answer(product),
                answer_text=a_text,
                answer_latex=latex_answer(a_text),
                marks=1,
                sensible_range=(0, None),
                working=[f"{shown_u} × {shown_v} = {a_text} {unit_name}²."],
                distractors=d_a,
            ),
            dict(
                prompt=prompts[1],
                answer_kind="number",
                answer=k,
                answer_text=b_text,
                answer_latex=latex_answer(b_text),
                marks=1,
                sensible_range=(0, None),
                working=[f"{shown_u} has {sf_u} significant figures and {shown_v} has {sf_v}; the smaller is {k}."],
                distractors=d_b,
            ),
            dict(
                prompt=prompts[2],
                answer_kind="number",
                answer=numeric_answer(rounded),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=follow,
                working=[f"{a_text} rounded to {k} significant figures is {c_text} {unit_name}²."],
                distractors=d_c,
            ),
        ],
    )
