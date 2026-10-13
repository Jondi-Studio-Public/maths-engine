"""Exam-tier source for `senior-suitable-converses-of-some-of-the-above-results`.

The practice tiers ask which conclusion a configuration supports. An exam question instead gives angles as
expressions in an unknown, states a converse theorem as the condition, and asks for the value of the unknown that
makes the converse apply, then uses the result. Three shapes, chosen per seed:

* ``cyclic``: opposite angles of a quadrilateral supplementary (converse of the cyclic quadrilateral theorem),
* ``same_arc``: equal angles on the same side of a chord (converse of "angles in the same segment are equal"), then
  the angle at the centre,
* ``semicircle``: a right angle at the third vertex (converse of "the angle in a semicircle is a right angle"), then
  the remaining angles of the triangle.

Each has three connected parts: (a) the unknown, (b) an angle that depends on it, (c) a further angle found from (b).
(b) and (c) follow through.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor
from mathgen.core import check, generator, join_terms, latex_answer, pick_distractors, rounded_decimal_string


def _lin(a: int, b: int) -> str:
    return f"({join_terms([(a, 'x'), (b, '')])})°"


def _cand(value: Fraction | None, misconception: str | None) -> Distractor | None:
    if value is None or value <= 0 or value >= 1000:
        return None
    text = rounded_decimal_string(value, 2)
    return Distractor(text, misconception, Fraction(text))


def _tag(true: int, cands) -> list[Distractor]:
    cands = [c for c in cands if c is not None]
    picked = pick_distractors(str(true), cands, 3)
    check(len(picked) >= 2, "too few distractors")
    check(all(Fraction(d.value) != true for d in picked), "a distractor equals the key")
    return picked


def _part(prompt: str, value: int, marks: int, **extra) -> dict:
    return dict(
        prompt=prompt,
        answer_kind="number",
        answer=value,
        answer_text=str(value),
        answer_latex=latex_answer(str(value)),
        marks=marks,
        sensible_range=(0, 180),
        **extra,
    )


def _solve(a: int, b: int, target: int) -> Fraction:
    """x with a*x + b == target."""
    return Fraction(target - b, a)


def _cyclic(rng: random.Random):
    for _ in range(1000):
        x = rng.randint(5, 30)
        a, c, e = rng.randint(1, 4), rng.randint(1, 4), rng.randint(1, 4)
        angle_a = rng.randint(55, 125)
        b = angle_a - a * x
        d = 180 - angle_a - c * x
        angle_b = rng.randint(50, 130)
        f = angle_b - e * x
        if max(abs(b), abs(d), abs(f)) <= 45 and angle_b != angle_a and angle_b != 180 - angle_a:
            break
    else:
        raise AssertionError("no cyclic draw in 1000 tries")
    angle_d = 180 - angle_b
    check(a * x + b + c * x + d == 180 and e * x + f == angle_b, "construction broke")
    stem = (
        f"In quadrilateral ABCD, ∠DAB = {_lin(a, b)}, ∠ABC = {_lin(e, f)} and ∠BCD = {_lin(c, d)}, "
        "where x is a positive whole number. By the converse of the cyclic quadrilateral theorem, ABCD is a cyclic "
        "quadrilateral when its opposite angles are supplementary."
    )
    x_equal = _solve(a - c, b - d, 0) if a != c else None
    a_cands = [
        _cand(x_equal, "equates-instead-of-supplementary") if x_equal else None,
        _cand(_solve(a + c, b + d, 360), "sum-to-360-not-180"),
        _cand(_solve(a + c, b + d, 90), None),
        _cand(Fraction(180 - b - d, a + c) + 1, None),
    ]
    b_cands = [
        _cand(Fraction(angle_d), "reports-the-given-partner-angle"),
        _cand(Fraction(e * (x + 1) + f), None),
        _cand(Fraction(e * x - f), "substitution-sign-error"),
    ]
    c_cands = [
        _cand(Fraction(angle_b), "copies-the-given-angle"),
        _cand(Fraction(360 - angle_b), "sum-to-360-not-180"),
        _cand(Fraction(90 - angle_b + 90) + 1, None),
    ]
    parts = [
        _part(
            "Find the value of x for which ABCD is a cyclic quadrilateral.",
            x,
            2,
            working=[f"∠DAB + ∠BCD = 180°: {join_terms([(a + c, 'x'), (b + d, '')])} = 180, so x = {x}."],
            distractors=_tag(x, a_cands),
        ),
        _part(
            "For this value of x, find the size of ∠ABC, in degrees.",
            angle_b,
            1,
            follow_through=f"{e} * part_a + ({f})",
            working=[f"∠ABC = {join_terms([(e, 'x'), (f, '')])} at x = {x}, which is {angle_b}°."],
            distractors=_tag(angle_b, b_cands),
        ),
        _part(
            "Hence find the size of ∠ADC, in degrees.",
            angle_d,
            1,
            follow_through="180 - part_b",
            working=[
                f"Opposite angles of a cyclic quadrilateral are supplementary: ∠ADC = 180 − {angle_b} = {angle_d}°."
            ],
            distractors=_tag(angle_d, c_cands),
        ),
    ]
    return stem, parts


def _same_arc(rng: random.Random):
    for _ in range(1000):
        x = rng.randint(4, 25)
        a, c = rng.randint(1, 5), rng.randint(1, 5)
        if a == c:
            continue
        angle = rng.randint(30, 85)
        b = angle - a * x
        d = angle - c * x
        if max(abs(b), abs(d)) <= 40:
            break
    else:
        raise AssertionError("no same-arc draw in 1000 tries")
    check(a * x + b == c * x + d == angle, "construction broke")
    stem = (
        f"Points C and D lie on the same side of the line AB. ∠ACB = {_lin(a, b)} and ∠ADB = {_lin(c, d)}, where x is "
        "a positive whole number. By the converse of the theorem on angles in the same segment, A, B, C and D lie on "
        "a circle with centre O when ∠ACB = ∠ADB."
    )
    a_cands = [
        _cand(_solve(a + c, b + d, 180), "applies-supplementary-to-equal-type"),
        _cand(_solve(a + c, b + d, 90), None),
        _cand(Fraction(abs(b - d), abs(a - c)) + 1, None),
    ]
    b_cands = [
        _cand(Fraction(180 - angle), "reports-the-neighbouring-angle"),
        _cand(Fraction(a * (x + 1) + b), None),
        _cand(Fraction(a * x - b), "substitution-sign-error"),
    ]
    c_cands = [
        _cand(Fraction(angle), "copies-the-given-angle"),
        _cand(Fraction(angle, 2), "angle-at-centre-halved"),
        _cand(Fraction(180 - angle), "reports-the-neighbouring-angle"),
        _cand(Fraction(3 * angle), None),
    ]
    parts = [
        _part(
            "Find the value of x for which A, B, C and D lie on a circle.",
            x,
            2,
            working=[f"{join_terms([(a, 'x'), (b, '')])} = {join_terms([(c, 'x'), (d, '')])}, so x = {x}."],
            distractors=_tag(x, a_cands),
        ),
        _part(
            "For this value of x, find the size of ∠ACB, in degrees.",
            angle,
            1,
            follow_through=f"{a} * part_a + ({b})",
            working=[f"∠ACB = {join_terms([(a, 'x'), (b, '')])} at x = {x}, which is {angle}°."],
            distractors=_tag(angle, b_cands),
        ),
        _part(
            "Hence find the size of ∠AOB, the angle at the centre standing on chord AB, in degrees.",
            2 * angle,
            2,
            follow_through="2 * part_b",
            working=[f"The angle at the centre is twice the angle at the circumference: 2 × {angle} = {2 * angle}°."],
            distractors=_tag(2 * angle, c_cands),
        ),
    ]
    return stem, parts


def _semicircle(rng: random.Random):
    for _ in range(1000):
        x = rng.randint(5, 25)
        a = rng.randint(2, 6)
        b = 90 - a * x
        other = rng.randint(25, 65)
        e = rng.randint(1, 4)
        f = other - e * x
        if abs(b) <= 40 and abs(f) <= 40 and other != 45:
            break
    else:
        raise AssertionError("no semicircle draw in 1000 tries")
    third = 90 - other
    check(a * x + b == 90 and e * x + f == other, "construction broke")
    stem = (
        f"In triangle ABC, ∠ACB = {_lin(a, b)} and ∠CAB = {_lin(e, f)}, where x is a positive whole number. By the "
        "converse of the angle in a semicircle theorem, AB is a diameter of the circle through A, B and C when "
        "∠ACB is a right angle."
    )
    a_cands = [
        _cand(_solve(a, b, 180), "uses-180-for-semicircle"),
        _cand(_solve(a, b, 45), None),
        _cand(_solve(a - e, b - f, 0) if a != e else None, None),
        _cand(Fraction(90 - b, a) + 1, None),
    ]
    b_cands = [
        _cand(Fraction(third), "reports-the-given-partner-angle"),
        _cand(Fraction(e * (x + 1) + f), None),
        _cand(Fraction(e * x - f), "substitution-sign-error"),
    ]
    c_cands = [
        _cand(Fraction(180 - other), "ignored-the-found-angle"),
        _cand(Fraction(other), "copies-the-given-angle"),
        _cand(Fraction(90 + other), None),
    ]
    parts = [
        _part(
            "Find the value of x for which AB is a diameter of the circle through A, B and C.",
            x,
            2,
            working=[f"{join_terms([(a, 'x'), (b, '')])} = 90, so x = {x}."],
            distractors=_tag(x, a_cands),
        ),
        _part(
            "For this value of x, find the size of ∠CAB, in degrees.",
            other,
            1,
            follow_through=f"{e} * part_a + ({f})",
            working=[f"∠CAB = {join_terms([(e, 'x'), (f, '')])} at x = {x}, which is {other}°."],
            distractors=_tag(other, b_cands),
        ),
        _part(
            "Hence find the size of ∠ABC, in degrees.",
            third,
            1,
            follow_through="90 - part_b",
            working=[f"The angles of triangle ABC sum to 180°: ∠ABC = 180 − 90 − {other} = {third}°."],
            distractors=_tag(third, c_cands),
        ),
    ]
    return stem, parts


_SHAPES = [_cyclic, _same_arc, _semicircle]


@generator("exd_x_sc6_circle_converses", level=10, presets={"exam": dict(tier="exam")})
def exd_x_sc6_circle_converses(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    stem, parts = rng.choice(_SHAPES)(rng)
    return dict(text=stem, parts=parts)
