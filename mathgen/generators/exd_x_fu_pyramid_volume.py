"""Exam source for `senior-use-formulas-to-find-the-volume-of-pyramids`: a three-part chain on one pyramid.

The stem gives a pyramid with a square or rectangular base and the slant height of one triangular face. Part (a)
finds the perpendicular height by Pythagoras on the right triangle formed by the height, half of the base edge perpendicular to
the edge that face stands on, and the slant height. Part (b) uses that height for the volume (one third of base area times
height). Part (c) turns the volume into a capacity in litres. Parts (b) and (c) carry `follow_through`, so a learner
whose (a) is wrong still earns (b) and (c) for using their own answer correctly.

The half-edge, height and slant height are drawn as a scaled Pythagorean triple, so the height in (a) is a whole
number of centimetres. The volume and the capacity are rounded to 1 decimal place and a draw is kept only when
each rounded value is clear of a rounding boundary, so the exact route and the route through the learner's
rounded (b) agree.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    VerificationError,
    attempts,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_MARGIN = Fraction(1, 100)  # keep every rounded value this far (in units of the last place) from a boundary
_TRIPLES = ((3, 4, 5), (5, 12, 13), (8, 15, 17), (7, 24, 25), (20, 21, 29), (9, 40, 41), (12, 35, 37))
_OBJECTS = (
    "pyramid-shaped planter",
    "pyramid-shaped glass display case",
    "pyramid-shaped sandpit cover",
    "pyramid-shaped water feature",
    "pyramid-shaped storage bin",
)


def _clear(value: Fraction, places: int) -> bool:
    """True when `value` is at least `_MARGIN` (of the last place) from a rounding boundary at `places` d.p."""
    scaled = value * 10**places
    return abs(scaled % 1 - Fraction(1, 2)) > _MARGIN


def _r1(value: Fraction) -> Fraction:
    return round_half_up(value, 1)


def _fmt(value: Fraction) -> str:
    return stated_precision_string(_r1(value), 1)


def _draw(rng):
    for _ in attempts(2000, "exd_x_fu_pyramid_volume: no draw"):
        p, q, r = rng.choice(_TRIPLES)
        if rng.random() < 0.5:
            p, q = q, p
        k = rng.randint(1, 6)
        half, height, slant = p * k, q * k, r * k
        w = 2 * half  # the base edge the face stands on
        if not 20 <= w <= 120 or not 10 <= height <= 120:
            continue
        square = rng.random() < 0.5
        b = w if square else 2 * rng.randint(10, 60)
        if not square and b == w:
            continue
        volume = Fraction(b * w * height, 3)
        litres = volume / 1000
        if not (_clear(volume, 1) and _clear(litres, 1)):
            continue
        if height in (b, w, slant) or height == slant - half:
            continue
        return dict(half=half, height=height, slant=slant, w=w, b=b, square=square, volume=volume, litres=litres)
    raise VerificationError("exd_x_fu_pyramid_volume: no draw")


@generator("exd_x_fu_pyramid_volume", level=12, presets={"exam": {}})
def exd_x_fu_pyramid_volume(rng, tier: str = "exam"):
    """Height by Pythagoras, then pyramid volume, then capacity in litres."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    d = _draw(rng)
    half, h, slant, w, b, volume, litres = (d[k] for k in ("half", "height", "slant", "w", "b", "volume", "litres"))
    thing = rng.choice(_OBJECTS)
    if d["square"]:
        stem = (
            f"A {thing} has a square base of side {w} cm and its apex directly above the centre of the base. "
            f"Each triangular face has a slant height of {slant} cm."
        )
    else:
        stem = (
            f"A {thing} has a rectangular base {b} cm by {w} cm and its apex directly above the centre of the "
            f"base. The triangular face standing on a {'longer' if b > w else 'shorter'} edge has a slant height "
            f"of {slant} cm."
        )
    # (a)
    added = _sqrt_tenths(slant, half)
    h_text = str(h)
    a_cands = [
        Distractor(str(slant), "slant-height-as-height", slant),
        Distractor(str(slant - half), "pythagoras-subtract-lengths", slant - half),
        Distractor(stated_precision_string(added, 1), "pythagoras-add-squares-for-leg", added),
    ]
    # (b)
    v_text = _fmt(volume)
    base_area = b * w
    b_cands = [
        Distractor(_fmt(Fraction(base_area * h)), "pyramid-volume-no-third", _r1(Fraction(base_area * h))),
        Distractor(_fmt(Fraction(base_area * slant, 3)), "slant-height-as-height", _r1(Fraction(base_area * slant, 3))),
        Distractor(_fmt(Fraction(base_area * h, 2)), None, _r1(Fraction(base_area * h, 2))),
    ]
    # (c)
    l_text = _fmt(litres)
    c_cands = [
        Distractor(_fmt(volume / 100), "cubic-cm-to-litres-wrong-factor", _r1(volume / 100)),
        Distractor(_fmt(volume / 10), "cubic-cm-to-litres-wrong-factor", _r1(volume / 10)),
        Distractor(_fmt(Fraction(base_area * h, 1000)), "pyramid-volume-no-third", _r1(Fraction(base_area * h, 1000))),
    ]
    a_d = pick_distractors(h_text, a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(v_text, b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(l_text, c_cands, 3, tolerance=GRADER_TOLERANCE)
    if not (a_d and b_d and c_d):
        raise VerificationError("exd_x_fu_pyramid_volume: a part has no distractor")
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find the perpendicular height of the pyramid, in centimetres.",
                answer_kind="number",
                answer=numeric_answer(h),
                answer_text=h_text,
                answer_latex=latex_answer(h_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    (
                        f"Half of the {w} cm edge is {half} cm. The height, this half-edge and the slant height form a right triangle."
                        if d["square"]
                        else f"The face stands on the {b} cm edge, so the run to the apex is half of the other edge: {w} ÷ 2 = {half} cm. "
                        "The height, this run and the slant height form a right triangle."
                    ),
                    f"h² = {slant}² − {half}² = {slant * slant - half * half}, so h = {h_text} cm.",
                ],
                distractors=a_d,
            ),
            dict(
                prompt="Hence find the volume of the pyramid, in cubic centimetres, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(_r1(volume)),
                answer_text=v_text,
                answer_latex=latex_answer(v_text),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"part_a*{base_area}/3",
                working=[f"V = ⅓ × {b} × {w} × {h_text} = {v_text} cm³ (1 d.p.)."],
                distractors=b_d,
            ),
            dict(
                prompt="The pyramid is filled to the top with water. Using your volume from (b), find how many "
                "litres it holds, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(_r1(litres)),
                answer_text=l_text,
                answer_latex=latex_answer(l_text),
                marks=1,
                sensible_range=(0, None),
                follow_through="part_b/1000",
                working=[f"1 litre = 1000 cm³, so capacity = {v_text} ÷ 1000 = {l_text} litres (1 d.p.)."],
                distractors=c_d,
            ),
        ],
    )


def _sqrt_tenths(slant: int, half: int) -> Fraction:
    """sqrt(slant^2 + half^2) to 1 d.p., as the learner who adds the squares would report it."""
    n = (slant * slant + half * half) * 100
    return Fraction(isqrt(n) + (1 if (isqrt(n) + Fraction(1, 2)) ** 2 <= n else 0), 10)
