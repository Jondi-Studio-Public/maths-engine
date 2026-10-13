"""Exam source for angles on a straight line: unknown angles written as expressions in x.

Three angles share a straight line, two of them written as expressions in x. Part (a) forms and solves the equation
for x, part (b) uses x for the size of the first angle, and part (c) bisects that angle and asks for the angle
between the bisector and the next ray. Parts (b) and (c) carry `follow_through`.

Built backwards from x, the coefficients and the constants; the known middle angle is whatever makes the three sum
to 180.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator, join_terms

from ._exd_pr3_support import leaks, part

_OPENINGS = [
    "A surveyor in Ballarat marks a straight boundary AB and a point O on it. Rays OC and OD lie on the same side "
    "of AB, with C nearer A.",
    "On a map of a Brisbane street, points A, O and B lie on a straight road. Two paths, OC and OD, leave O on the "
    "same side of the road, with OC nearer A.",
    "Points A, O and B lie on a straight line. Rays OC and OD are drawn on one side of the line, with C nearer A.",
]


def _expr(coef: int, const: int) -> str:
    sign = "+" if const >= 0 else "−"
    lead = "x" if coef == 1 else f"{coef}x"
    return f"({lead} {sign} {abs(const)})°"


@generator("exd_x_pr3_straight_line_angles", level=8, presets={"exam": dict(tier="exam")})
def exd_x_pr3_straight_line_angles(rng, tier="exam"):
    check(tier == "exam", "only the exam tier exists")
    for _ in attempts(2000, "angles that are all sensible whole numbers"):
        x = rng.randint(6, 30)
        a, c = rng.randint(1, 5), rng.randint(1, 5)
        b = rng.choice([v for v in range(-15, 41) if v])
        d = rng.choice([v for v in range(-15, 41) if v])
        aoc, dob = a * x + b, c * x + d
        k = 180 - aoc - dob
        if aoc < 20 or dob < 15 or not 15 <= k <= 90 or aoc % 2 or a + c not in (2, 4, 5, 8, 10):
            continue
        stem = (
            f"{rng.choice(_OPENINGS)} ∠AOC = {_expr(a, b)}, ∠COD = {k}° and ∠DOB = {_expr(c, d)}. Ray OE bisects ∠AOC. "
            "Give every answer as a whole number."
        )
        if not leaks(stem, [x, aoc, aoc // 2 + k]):
            break
    check(aoc + k + dob == 180 and (a + c) * x + b + d + k == 180, "angles do not sum to 180")
    e_od = aoc // 2 + k
    # Kept only where the slip lands on a whole number (the answer is one), so each is spelled like the key.
    xs = [
        (Fraction(180 - b - d, a + c), "ignored-known-angle"),
        (Fraction(360 - k - b - d, a + c), "uses-360"),
        (Fraction(180 + b + d - k, a + c), "sign-error-moving-term"),
        (Fraction(180 - k - b - d, 1), "forgot-to-divide-by-coefficient"),
        (Fraction(180 - k, a + c), "ignored-constant-term"),
        (Fraction(180 - k - b - d, a), "used-one-coefficient"),
        (Fraction(180 - k - b - d, c), "used-one-coefficient"),
        (Fraction(180 - b - d, 1), "forgot-to-divide-by-coefficient"),
        (Fraction(360 - k - b - d, 1), "forgot-to-divide-by-coefficient"),
    ]
    parts = [
        part(
            "Use the angles on the straight line to find the value of x.",
            x,
            marks=2,
            sensible_range=(0, None),
            working=[
                f"Angles on a straight line add to 180°: ({join_terms([(a, 'x'), (b, '')])}) + {k} + ({join_terms([(c, 'x'), (d, '')])}) = 180.",
                f"{a + c}x = {180 - k - b - d}, so x = {x}.",
            ],
            candidates=xs,
        ),
        part(
            "Using your answer to (a), find the size of ∠AOC, in degrees.",
            aoc,
            marks=1,
            follow_through=f"{a}*part_a+({b})",
            sensible_range=(0, 180),
            working=[f"∠AOC = {a} × {x} {'+' if b > 0 else '-'} {abs(b)} = {aoc}°."],
            candidates=[
                (a * x, "dropped-constant-term"),
                (dob, "used-other-angle-expression"),
                (180 - aoc, "angle-to-wrong-arm"),
            ],
        ),
        part(
            "Using your answer to (b), find the size of ∠EOD, in degrees.",
            e_od,
            marks=2,
            follow_through=f"part_b/2+{k}",
            sensible_range=(0, 180),
            working=[f"∠EOC = {aoc} ÷ 2 = {aoc // 2}°.", f"∠EOD = {aoc // 2} + {k} = {e_od}°."],
            candidates=[
                (aoc // 2, "ignored-known-angle"),
                (aoc + k, "bisector-not-halved"),
                (180 - aoc // 2, "angle-to-wrong-arm"),
            ],
        ),
    ]
    return dict(text=stem, parts=parts)
