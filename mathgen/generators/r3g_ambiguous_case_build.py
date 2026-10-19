"""Generates `r3g_ambiguous_case` questions (easy, medium, hard) and the exam chain `r3g_exam_ambiguous_case`.

Skill ``ambiguous-case-triangle`` (``trigonometry``, answers ``set`` and ``number``): the ambiguous case of the sine
rule. Two sides and a non-included angle (SSA) fit two different triangles when the side opposite the given angle is
shorter than the other given side but longer than that side times the sine of the angle. The existing sine rule skill
handles the case with a single triangle.

* easy: both possible sizes of the angle B, as a set, to 1 decimal place.
* medium: both possible lengths of the third side c, as a set, to 1 decimal place.
* hard: either the two possible areas of the triangle, or the least whole-number length of the side opposite the given
  angle that allows a second triangle, or the length of that side at which the two triangles become one (a right angle).
* exam (``r3g_exam_ambiguous_case``): (a) the acute angle B, (b) the obtuse angle B (follows through from (a)),
  (c) the third side in the acute case (follows through from (a)) and (d) the third side in the obtuse case (follows
  through from (b)).

Every angle comes from the sine rule and is checked a second way: the third side of each triangle is a root of the
quadratic given by the cosine rule, ``c² - 2bc cos A + b² - a² = 0``, and the angle B follows from it.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator
from ._r3g_kit import (
    decimal_dist,
    decimal_set_dist,
    decimal_set_fields,
    value_set_dist,
    value_set_fields,
    number_fields,
    pick,
    retry_short,
)

F = Fraction


def _clear_of_ties(value: float, places: int) -> bool:
    scaled = abs(value) * 10**places
    return abs(scaled - math.floor(scaled) - 0.5) > 0.04


def _tenth(value: float) -> Fraction:
    return F(str(round(value, 9)))


def _triangle(rng, *, need_integer_gap: bool = False):
    """(A, a, b) in degrees and lengths with b sin A < a < b, the quantities all clear of rounding ties."""
    for _ in attempts(2000, "an ambiguous-case triangle clear of rounding ties"):
        A = rng.randint(22, 58)
        b = rng.randint(70, 150) / 10
        a = rng.randint(int(b * math.sin(math.radians(A)) * 10) + 3, int(b * 10) - 3) / 10
        sA = math.sin(math.radians(A))
        sB = b * sA / a
        if not 0.2 < sB < 0.96:
            continue
        B = math.degrees(math.asin(sB))
        Bo = 180 - B
        C1, C2 = 180 - A - B, 180 - A - Bo
        if C2 <= 3:
            continue
        c1, c2 = a * math.sin(math.radians(C1)) / sA, a * math.sin(math.radians(C2)) / sA
        area1, area2 = 0.5 * a * b * math.sin(math.radians(C1)), 0.5 * a * b * math.sin(math.radians(C2))
        if not all(_clear_of_ties(v, 1) for v in (B, c1, c2, area1, area2)):
            continue
        if need_integer_gap and abs(b * sA - round(b * sA)) < 0.2:
            continue
        return A, a, b
    raise AssertionError  # pragma: no cover


def solve(A: float, a: float, b: float) -> dict:
    """Both triangles: angles, third sides and areas. The third side is also found as a root of the cosine rule."""
    sA = math.sin(math.radians(A))
    B = math.degrees(math.asin(b * sA / a))
    out = {"B": [B, 180 - B]}
    out["C"] = [180 - A - Bv for Bv in out["B"]]
    out["c"] = [a * math.sin(math.radians(C)) / sA for C in out["C"]]
    out["area"] = [0.5 * a * b * math.sin(math.radians(C)) for C in out["C"]]
    # second route: the cosine rule c² - 2bc cos A + b² - a² = 0
    disc = a * a - (b * sA) ** 2
    roots = sorted([b * math.cos(math.radians(A)) - math.sqrt(disc), b * math.cos(math.radians(A)) + math.sqrt(disc)])
    check(all(r > 0 for r in roots), "a root of the cosine rule is not a length")
    check(all(abs(x - y) < 1e-9 for x, y in zip(sorted(out["c"]), roots)), "third sides differ from the cosine rule")
    for Bv, c in zip(out["B"], out["c"]):
        cos_b = (a * a + c * c - b * b) / (2 * a * c)
        check(abs(math.degrees(math.acos(cos_b)) - Bv) < 1e-7, "angle B differs from the cosine rule")
    return out


def _stem(A, a, b, ask: str) -> str:
    variant = (A + int(a * 10) + int(b * 10)) % 3
    if variant == 0:
        return (
            f"In triangle ABC, angle A = {A}°, side a = {a:.1f} cm and side b = {b:.1f} cm, where a is opposite A and b "
            f"is opposite B. There are two different triangles with these measurements. {ask}"
        )
    if variant == 1:
        return (
            f"Two different triangles ABC can be drawn with angle A = {A}°, side b = {b:.1f} cm and side a = {a:.1f} cm "
            f"(a lies opposite A, b opposite B). {ask}"
        )
    return (
        f"A surveyor measures angle A = {A}° and the sides AC = b = {b:.1f} cm and BC = a = {a:.1f} cm of triangle ABC. "
        f"These measurements fit two different triangles. {ask}"
    )


def _easy(rng):
    A, a, b = _triangle(rng)
    sol = solve(A, a, b)
    B1, B2 = sol["B"]
    fields = decimal_set_fields([_tenth(B1), _tenth(B2)], 1)
    sA = math.sin(math.radians(A))
    inverted = math.degrees(math.asin(min(1.0, a * sA / b)))
    cands = [
        decimal_set_dist([_tenth(B1)], 1, "r3g-ambiguous-second-triangle-missed"),
        decimal_set_dist([_tenth(B1), _tenth(90 - B1)], 1, "r3g-ambiguous-complement-for-supplement"),
        decimal_set_dist([_tenth(inverted), _tenth(180 - inverted)], 1, "r3g-ambiguous-sine-rule-inverted"),
        decimal_set_dist([_tenth(B2)], 1, "r3g-ambiguous-acute-case-missed"),
    ]
    return dict(
        text=_stem(A, a, b, "Find the two possible sizes of angle B, correct to 1 decimal place, separated by commas."),
        working=[
            f"By the sine rule, sin B = b sin A ÷ a = {b:.1f} × sin {A}° ÷ {a:.1f}, so B = {B1:.1f}° or B = 180° - {B1:.1f}° = {B2:.1f}°.",
            f"Both fit: A + B < 180° in each case because b > a means B > A.",
        ],
        distractors=pick(fields["answer_text"], cands, 3),
        **fields,
    )


def _medium(rng):
    A, a, b = _triangle(rng)
    sol = solve(A, a, b)
    c1, c2 = sol["c"]
    fields = decimal_set_fields([_tenth(c1), _tenth(c2)], 1)
    sA = math.sin(math.radians(A))
    B1, B2 = sol["B"]
    angle_slip = [a * math.sin(math.radians(180 - Bv)) / sA for Bv in (B1, B2)]
    cands = [
        decimal_set_dist([_tenth(c1)], 1, "r3g-ambiguous-second-triangle-missed"),
        decimal_set_dist([_tenth(c2)], 1, "r3g-ambiguous-acute-case-missed"),
        decimal_set_dist([_tenth(v) for v in angle_slip], 1, "r3g-ambiguous-angle-sum-slip"),
        decimal_set_dist(
            [_tenth(b * math.sin(math.radians(C)) / sA) for C in sol["C"]], 1, "r3g-ambiguous-wrong-side-in-sine-rule"
        ),
    ]
    return dict(
        text=_stem(
            A, a, b, "Find the two possible lengths of side c, correct to 1 decimal place, separated by commas."
        ),
        working=[
            f"sin B = {b:.1f} sin {A}° ÷ {a:.1f}, so B = {B1:.1f}° or {B2:.1f}°.",
            f"C = 180° - {A}° - B = {sol['C'][0]:.1f}° or {sol['C'][1]:.1f}°.",
            f"c = a sin C ÷ sin A = {c1:.1f} cm or {c2:.1f} cm.",
        ],
        distractors=pick(fields["answer_text"], cands, 3),
        **fields,
    )


def _hard_areas(rng):
    A, a, b = _triangle(rng)
    sol = solve(A, a, b)
    a1, a2 = sol["area"]
    fields = decimal_set_fields([_tenth(a1), _tenth(a2)], 1)
    B1, B2 = sol["B"]
    wrong_angle = [0.5 * a * b * math.sin(math.radians(Bv)) for Bv in (B1, B2)]
    cands = [
        decimal_set_dist([_tenth(a1)], 1, "r3g-ambiguous-second-triangle-missed"),
        decimal_set_dist([_tenth(a2)], 1, "r3g-ambiguous-acute-case-missed"),
        decimal_set_dist([_tenth(0.5 * a * b * math.sin(math.radians(A)))], 1, "r3g-ambiguous-included-angle-wrong"),
    ]
    if round(wrong_angle[0], 1) != round(wrong_angle[1], 1):
        cands.append(decimal_set_dist([_tenth(v) for v in wrong_angle], 1, "r3g-ambiguous-included-angle-wrong"))
    return dict(
        text=_stem(
            A, a, b, "Find the two possible areas of the triangle, correct to 1 decimal place, separated by commas."
        ).replace("cm.", "cm."),
        working=[
            f"B = {B1:.1f}° or {B2:.1f}°, so C = {sol['C'][0]:.1f}° or {sol['C'][1]:.1f}°.",
            f"Area = ½ a b sin C = {a1:.1f} cm² or {a2:.1f} cm².",
        ],
        distractors=pick(fields["answer_text"], cands, 3),
        **fields,
    )


def _hard_critical(rng):
    """The interval of side lengths that gives two triangles: both ends, b sin A and b."""
    from ._r3g_kit import TooFewDistractors

    A = rng.randint(22, 58)
    b = rng.randint(70, 150) / 10
    critical = b * math.sin(math.radians(A))
    if not _clear_of_ties(critical, 2):
        raise TooFewDistractors("critical value on a rounding tie")
    # second route: at a = b sin A the cosine rule has a repeated root c = b cos A, so the triangle is right-angled at C
    disc = (2 * b * math.cos(math.radians(A))) ** 2 - 4 * (b * b - critical**2)
    check(abs(disc) < 1e-9, "the cosine rule should have a repeated root at the critical length")
    fields = decimal_set_fields([_tenth(critical), _tenth(b)], 2)
    wrong = [
        (b * math.sin(A), "r3g-ambiguous-calculator-in-radians"),
        (b * math.cos(math.radians(A)), "r3g-ambiguous-wrong-ratio"),
        (b / math.sin(math.radians(A)), "r3g-ambiguous-wrong-ratio"),
        (b * math.tan(math.radians(A)), "r3g-ambiguous-wrong-ratio"),
    ]
    cands = [decimal_set_dist([_tenth(w), _tenth(b)], 2, t) for w, t in wrong]
    cands.append(decimal_set_dist([_tenth(critical)], 2, "r3g-ambiguous-second-triangle-missed"))
    cands.append(decimal_set_dist([_tenth(b)], 2, "r3g-ambiguous-upper-limit-given"))
    return dict(
        text=(
            f"In triangle ABC, angle A = {A}° and side b = {b:.1f} cm. The side a, opposite angle A, is varied. "
            "There are two different triangles for every a strictly between two limits. "
            "Find the two limits, correct to 2 decimal places, separated by commas."
        ),
        working=[
            f"At the lower limit B = 90°, so sin B = 1 and a = b sin A = {b:.1f} × sin {A}° = {critical:.2f} cm.",
            f"At the upper limit a = b = {b:.1f} cm, because for a at least b the angle B must be smaller than A and only one triangle fits.",
        ],
        distractors=pick(fields["answer_text"], cands, 3),
        **fields,
    )


def _hard_whole(rng):
    for _ in attempts(500, "a critical length away from a whole number, with a short list of whole numbers"):
        A = rng.randint(22, 58)
        b = rng.randint(80, 150) / 10
        critical = b * math.sin(math.radians(A))
        if abs(critical - round(critical)) > 0.15 and 2 <= math.ceil(b) - 1 - math.floor(critical) <= 5:
            break
    lo, hi = math.floor(critical) + 1, math.ceil(b) - 1
    members = list(range(lo, hi + 1))
    # second route: test every whole number against b sin A < a < b directly
    check(
        all((critical < v < b) == (v in members) for v in range(1, 20)),
        "whole-number members differ from the interval test",
    )
    fields = value_set_fields(members)
    cands = [
        value_set_dist(range(lo - 1, hi + 1), "r3g-ambiguous-boundary-value-given"),
        value_set_dist(range(lo, hi + 2), "r3g-ambiguous-upper-limit-given"),
        value_set_dist(range(lo + 1, hi + 1), "r3g-ambiguous-rounded-the-wrong-way"),
        value_set_dist(range(round(critical), hi + 1), "r3g-ambiguous-critical-value-not-rounded-up"),
    ]
    return dict(
        text=(
            f"In triangle ABC, angle A = {A}° and side b = {b:.1f} cm. The side a, opposite angle A, is a whole number "
            "of centimetres. Find every value of a for which there are two different triangles, "
            "separated by commas."
        ),
        working=[
            f"Two triangles need b sin A < a < b, that is {critical:.2f} < a < {b:.1f}.",
            f"The whole numbers in that interval are {', '.join(map(str, members))}.",
        ],
        distractors=pick(fields["answer_text"], cands, 3),
        **fields,
    )


@generator(
    "r3g_ambiguous_case",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@retry_short
def r3g_ambiguous_case(rng, tier="easy"):
    """The ambiguous case of the sine rule; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_areas, _hard_critical, _hard_whole])(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("r3g_exam_ambiguous_case", level=12, presets={"exam": dict(tier="exam")})
@retry_short
def r3g_exam_ambiguous_case(rng, tier="exam"):
    """Exam chain: acute B, obtuse B, and the third side in each case."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a triangle whose follow-through sides round to the same tenth"):
        A, a, b = _triangle(rng)
        sol = solve(A, a, b)
        B1, B2 = _tenth(sol["B"][0]), _tenth(sol["B"][1])
        sA = math.sin(math.radians(A))
        fts = [a * math.sin(math.radians(180 - A - float(Bv))) / sA for Bv in (B1, B2)]
        if all(round(ft, 1) == round(c, 1) and _clear_of_ties(ft, 1) for ft, c in zip(fts, sol["c"])):
            break
    c1, c2 = sol["c"]
    ft_c = f"{a:.1f}*sin((180 - {A} - part_{{}})*pi/180)/sin({A}*pi/180)"

    def part(prompt, value, marks, working, wrongs, follow=None, places=1):
        fields = number_fields(value, places)
        d = dict(prompt=prompt, marks=marks, working=working, sensible_range=(0, None), **fields)
        d["distractors"] = pick(
            fields["answer_text"], [decimal_dist(F(str(round(w, 8))), places, t) for w, t in wrongs], 3
        )
        if follow:
            d["follow_through"] = follow
        return d

    B_acute = sol["B"][0]
    variant = rng.randrange(3)
    return dict(
        text=(
            f"In triangle ABC, angle A = {A}°, side a = {a:.1f} cm (opposite A) and side b = {b:.1f} cm (opposite B). "
            "These measurements fit two different triangles."
            if variant == 0
            else f"Two triangles ABC fit these measurements: angle A = {A}°, side b = {b:.1f} cm (opposite B) and side a = {a:.1f} cm (opposite A)."
            if variant == 1
            else f"A surveyor records angle A = {A}°, BC = a = {a:.1f} cm and AC = b = {b:.1f} cm in triangle ABC. The data fit two different triangles."
        ),
        parts=[
            part(
                "Use the sine rule to find the acute angle B, correct to 1 decimal place.",
                B1,
                2,
                [f"sin B = {b:.1f} sin {A}° ÷ {a:.1f}, so B = {float(B1):.1f}°."],
                [
                    (180 - B_acute, "r3g-ambiguous-acute-case-missed"),
                    (math.degrees(math.asin(min(1.0, a * sA / b))), "r3g-ambiguous-sine-rule-inverted"),
                    (90 - B_acute, "r3g-ambiguous-complement-for-supplement"),
                    (math.degrees(math.sin(math.radians(B_acute))), "r3g-ambiguous-sine-value-given-as-angle"),
                ],
            ),
            part(
                "Hence find the obtuse angle B in the second triangle, correct to 1 decimal place.",
                B2,
                1,
                [f"The other angle with the same sine is 180° - {float(B1):.1f}° = {float(B2):.1f}°."],
                [
                    (B_acute, "r3g-ambiguous-second-triangle-missed"),
                    (90 - B_acute, "r3g-ambiguous-complement-for-supplement"),
                    (360 - B_acute, "r3g-ambiguous-wrong-supplement"),
                    (90 + B_acute, "r3g-ambiguous-wrong-supplement"),
                ],
                "180 - part_a",
            ),
            part(
                "In the triangle with the acute angle B, find the length of side c, correct to 1 decimal place.",
                F(str(round(c1, 8))),
                2,
                [f"C = 180° - {A}° - {float(B1):.1f}° = {sol['C'][0]:.1f}°, and c = a sin C ÷ sin A = {c1:.1f} cm."],
                [
                    (c2, "r3g-ambiguous-second-triangle-missed"),
                    (a * math.sin(math.radians(180 - float(B1))) / sA, "r3g-ambiguous-angle-sum-slip"),
                    (b * math.sin(math.radians(sol["C"][0])) / sA, "r3g-ambiguous-wrong-side-in-sine-rule"),
                    (a * math.sin(math.radians(float(B1))) / sA, "r3g-ambiguous-angle-sum-slip"),
                ],
                ft_c.format("a"),
            ),
            part(
                "In the triangle with the obtuse angle B, find the length of side c, correct to 1 decimal place.",
                F(str(round(c2, 8))),
                2,
                [f"C = 180° - {A}° - {float(B2):.1f}° = {sol['C'][1]:.1f}°, and c = a sin C ÷ sin A = {c2:.1f} cm."],
                [
                    (c1, "r3g-ambiguous-acute-case-missed"),
                    (a * math.sin(math.radians(180 - float(B2))) / sA, "r3g-ambiguous-angle-sum-slip"),
                    (b * math.sin(math.radians(sol["C"][1])) / sA, "r3g-ambiguous-wrong-side-in-sine-rule"),
                    (a * math.sin(math.radians(float(B2))) / sA, "r3g-ambiguous-angle-sum-slip"),
                ],
                ft_c.format("b"),
            ),
        ],
    )
