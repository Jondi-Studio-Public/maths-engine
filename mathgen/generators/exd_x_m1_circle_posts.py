"""Exam source of `circle-reverse-from-area`: from a circle's area to its radius, its circumference and a count.

The practice tiers ask for one quantity worked back from an area. An exam question uses the radius as the first
link in a chain: (a) the radius from the area (a square root and a division by pi), (b) the circumference from
that radius, (c) how many evenly spaced posts go around the edge, which needs the circumference and a decision
about rounding a count (a fence post that does not fit still has to be bought).

## Construction: draw the radius first

The radius is a whole number of tenths that is not a whole number, so (a) is a genuine decimal. The displayed area
is the only rounded input. Working back from it (square root of area over pi) lands on the drawn radius because a
tenth of radius is worth far more area than the rounding of the area. Every displayed value is kept clear of a
rounding boundary, and the post count is checked on both routes (the exact circumference and the one-decimal
circumference a learner carries from (b)), so the exact route and the learner's route always agree.
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import (
    GRADER_TOLERANCE,
    attempts,
    check,
    generator,
    latex_answer,
    numeric_answer,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

PI = Fraction(314159, 100000)  # the value of pi the prompt tells the learner to use
_MARGIN = Fraction(1, 200)  # keep a displayed value this far from a rounding boundary
_R10_LO, _R10_HI = 15, 185  # radius 1.5 .. 18.5 m, in tenths
_SPACINGS = [
    ("1.5", Fraction(3, 2)),
    ("2", Fraction(2)),
    ("2.5", Fraction(5, 2)),
    ("3", Fraction(3)),
    ("4", Fraction(4)),
]

_STEMS = [
    ("A circular pond has a surface area of {area} m².", "pond", "fence post"),
    ("A circular vegetable garden has an area of {area} m².", "garden", "stake"),
    ("A circular dam has a surface area of {area} m².", "dam", "fence post"),
    ("A circular lawn at a school has an area of {area} m².", "lawn", "edging stake"),
    ("A circular sandpit has an area of {area} m².", "sandpit", "marker peg"),
    ("A circular trampoline pad has an area of {area} m².", "pad", "safety post"),
]


def _tenths(value: Fraction) -> Fraction:
    return round_half_up(value, 1)


def _clear(value: Fraction, places: int) -> bool:
    """True when `value` is at least `_MARGIN` from a boundary of the `places`-decimal-place rounding grid."""
    scaled = value * 10**places
    return abs((scaled % 1) - Fraction(1, 2)) / 10**places > _MARGIN


def _count(circumference: Fraction, spacing: Fraction) -> int:
    return math.ceil(circumference / spacing)


@generator("exd_x_m1_circle_posts", level=11, presets={"exam": {"tier": "exam"}})
def exd_x_m1_circle_posts(rng, tier: str = "exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(1000, "a radius and spacing clear of every rounding boundary"):
        r10 = rng.randint(_R10_LO, _R10_HI)
        if r10 % 10 == 0:
            continue
        r = Fraction(r10, 10)
        spacing_text, spacing = rng.choice(_SPACINGS)
        circumference = 2 * PI * r
        c1 = _tenths(circumference)
        if not _clear(circumference, 1):
            continue
        posts = _count(circumference, spacing)
        # The learner's route rounds the circumference to 1 d.p. first; the two counts must agree, with room to spare.
        if posts != _count(c1, spacing) or abs(circumference / spacing - round(circumference / spacing)) < Fraction(
            1, 50
        ):
            continue
        shown = {stated_precision_string(round_half_up(PI * r * r, 2), 2), spacing_text, "1", "3.14159"}
        if {stated_precision_string(r, 1), stated_precision_string(c1, 1), str(posts)} & shown:
            continue  # a key must not appear as a number in the prompt
        break

    area_display = round_half_up(PI * r * r, 2)
    area_text = stated_precision_string(area_display, 2)
    # Independent route, as a learner works: from the displayed area.
    r_back = math.sqrt(float(area_display) / float(PI))
    check(round(r_back, 1) == float(r), "radius does not survive the area rounding")
    check(abs(2 * float(PI) * r_back - float(circumference)) < 0.01, "circumference drifts with the area rounding")

    r_text = stated_precision_string(r, 1)
    c_text = stated_precision_string(c1, 1)
    stem_tmpl, thing, post = rng.choice(_STEMS)
    stem = f"{stem_tmpl.format(area=area_text)} Use π ≈ 3.14159."

    def tenth(v):
        return stated_precision_string(_tenths(v), 1)

    a_cands = [
        Distractor(tenth(PI * r * r / PI), "area-without-square-root", _tenths(r * r)),
        Distractor(
            tenth(Fraction(math.sqrt(float(area_display)))),
            "area-root-without-pi",
            _tenths(Fraction(math.sqrt(float(area_display)))),
        ),
        Distractor(tenth(2 * r), "diameter-for-radius", _tenths(2 * r)),
        Distractor(tenth(area_display / (2 * PI)), "area-formula-for-circumference", _tenths(area_display / (2 * PI))),
    ]
    b_cands = [
        Distractor(tenth(PI * r * r), "area-formula-for-circumference", _tenths(PI * r * r)),
        Distractor(tenth(2 * r), "diameter-for-circumference", _tenths(2 * r)),
        tenth(PI * r),
        Distractor(tenth(2 * PI * area_display), "area-formula-for-circumference", _tenths(2 * PI * area_display)),
    ]
    c_cands = [
        Distractor(str(math.floor(circumference / spacing)), "rounded-count-down", math.floor(circumference / spacing)),
        str(round(circumference / spacing)),
        Distractor(str(math.ceil(2 * r / spacing)), "diameter-for-circumference", math.ceil(2 * r / spacing)),
        str(math.ceil(circumference)),
    ]
    a_d = pick_distractors(r_text, a_cands, 3, tolerance=GRADER_TOLERANCE)
    b_d = pick_distractors(c_text, b_cands, 3, tolerance=GRADER_TOLERANCE)
    c_d = pick_distractors(str(posts), c_cands, 3)
    check(len(a_d) >= 2 and len(b_d) >= 2 and len(c_d) >= 2, "too few distractors survive")
    # Every distractor must be far from the key under the grader's tolerance (integers are graded exactly).
    check(all(abs(Fraction(str(d)) - _tenths(r)) > GRADER_TOLERANCE for d in a_d), "an (a) distractor is accepted")
    check(all(abs(Fraction(str(d)) - c1) > GRADER_TOLERANCE for d in b_d), "a (b) distractor is accepted")
    check(all(int(str(d)) != posts for d in c_d), "a (c) distractor equals the key")

    return dict(
        text=stem,
        parts=[
            dict(
                prompt=f"Find the radius of the {thing}, in metres, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(r),
                answer_text=r_text,
                answer_latex=latex_answer(r_text),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"Area = πr², so r² = {area_text} ÷ π.",
                    f"r = √({area_text} ÷ π) = {r_text} m (1 d.p.).",
                ],
                distractors=a_d,
            ),
            dict(
                prompt=f"Hence find the circumference of the {thing}, in metres, correct to 1 decimal place.",
                answer_kind="number",
                answer=numeric_answer(c1),
                answer_text=c_text,
                answer_latex=latex_answer(c_text),
                marks=1,
                sensible_range=(0, None),
                follow_through="2*3.14159*part_a",
                working=[f"C = 2πr = 2 × π × {r_text} = {c_text} m (1 d.p.)."],
                distractors=b_d,
            ),
            dict(
                prompt=f"One {post} is placed every {spacing_text} m around the edge of the {thing}, "
                f"and the last gap may be shorter. Using your circumference from (b), "
                f"how many {post}s are needed?",
                answer_kind="number",
                answer=posts,
                answer_text=str(posts),
                answer_latex=latex_answer(str(posts)),
                marks=2,
                sensible_range=(0, None),
                follow_through=f"ceiling(part_b/({spacing.numerator}/{spacing.denominator}))",
                working=[
                    f"{c_text} ÷ {spacing_text} = {float(c1 / spacing):.2f}, which is not a whole number.",
                    f"A part gap still needs a post, so round up: {posts} {post}s.",
                ],
                distractors=c_d,
            ),
        ],
    )
