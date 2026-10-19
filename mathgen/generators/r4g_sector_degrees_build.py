"""Sector area and arc length from an angle in degrees: `r4g_sector_degrees` (easy, medium, hard).

``sector-area-and-arc-from-degrees`` (``measurement``, answer ``number``). A sector with centre angle theta degrees
is the fraction theta/360 of a circle. Every key uses the true value of pi, is written to the stated number of
decimal places, and is kept clear of a rounding boundary so a learner's calculator agrees with it.

* easy: the area of a sector from its radius and angle.
* medium: the arc length of a sector, or its perimeter (arc plus the two radii).
* hard: work backwards or compose: the radius from a sector's area and angle, the angle (to the nearest degree)
  from an arc length and radius, or the area of a curved path between two arcs of different radii.

Back-calculations are checked by working forward from the displayed (rounded) figure, as a learner does.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import attempts, check, generator
from ._r4g_kit import clear_of_tie, number_text, single_number, retry

PI = Fraction(math.pi)
ANGLES = [a for a in range(30, 330, 10) if a not in (90, 180, 270)]
# (opening words, unit of length, largest sensible centre angle)
SCENES = [
    ("A lawn sprinkler waters a sector", "m", 280),
    ("A slice of pizza is a sector", "cm", 120),
    ("A garden bed is a sector", "m", 270),
    ("A lighthouse beam sweeps a sector", "km", 200),
    ("A fan-shaped dance floor is a sector", "m", 200),
    ("A wedge of cake is a sector", "cm", 120),
]


def draw_sector(rng, *, radii=(4, 24), tenths=False):
    scene = rng.choice(SCENES)
    if scene[0].startswith(("A slice", "A wedge")):
        radii = (min(radii[0], 6), min(radii[1], 14))
    r = Fraction(rng.randint(radii[0] * 10, radii[1] * 10), 10) if tenths else Fraction(rng.randint(*radii))
    theta = rng.choice([a for a in ANGLES if a <= scene[2]])
    return scene, r, theta


def area(r, theta):
    return Fraction(theta, 360) * PI * r * r


def arc(r, theta):
    return Fraction(theta, 360) * 2 * PI * r


def _head(scene, r, theta):
    text, unit, _ = scene
    return f"{text} with radius {int(r)} {unit} and a centre angle of {theta}°."


def _easy(rng):
    for _ in attempts(500, "an area clear of a rounding tie"):
        scene, r, theta = draw_sector(rng)
        if clear_of_tie(area(r, theta), 1):
            break
    a = area(r, theta)
    check(abs(float(a) - theta / 360 * math.pi * float(r) ** 2) < 1e-9, "area re-derivation differs")
    unit = scene[1]
    text = f"{_head(scene, r, theta)} Find its area, correct to 1 decimal place. Use the π button on your calculator."
    d = 2 * r
    slips = [
        (Fraction(theta, 360) * PI * r, "r4g-radius-not-squared"),
        (Fraction(theta, 360) * PI * d * d, "r4g-diameter-squared-for-radius-squared"),
        (arc(r, theta), "r4g-arc-formula-for-area"),
        (PI * r * r, "r4g-whole-circle-area"),
        (Fraction(theta, 180) * PI * r * r, "r4g-angle-fraction-over-180"),
        (Fraction(1, 2) * r * r * theta, "r4g-radian-formula-with-degrees"),
    ]
    working = [
        f"The sector is {theta}/360 of a circle.",
        f"Area = {theta}/360 x π x {int(r)}² = {number_text(a, 1)} {unit}².",
    ]
    return single_number(text, a, 1, working, slips)


def _medium_arc(rng):
    for _ in attempts(500, "an arc clear of a rounding tie"):
        scene, r, theta = draw_sector(rng)
        if clear_of_tie(arc(r, theta), 1):
            break
    l_ = arc(r, theta)
    unit = scene[1]
    text = f"{_head(scene, r, theta)} Find the length of its curved edge (the arc), correct to 1 decimal place."
    slips = [
        (area(r, theta), "r4g-area-formula-for-arc"),
        (2 * PI * r, "r4g-whole-circumference"),
        (Fraction(theta, 360) * PI * r, "r4g-radius-for-diameter-in-circumference"),
        (Fraction(theta, 180) * PI * r * 2, "r4g-angle-fraction-over-180"),
        (Fraction(theta) * r / 180 * 1, "r4g-angle-used-as-radians"),
    ]
    working = [f"Arc = {theta}/360 x 2 x π x {int(r)} = {number_text(l_, 1)} {unit}."]
    return single_number(text, l_, 1, working, slips)


def _medium_perimeter(rng):
    for _ in attempts(500, "a perimeter clear of a rounding tie"):
        scene, r, theta = draw_sector(rng)
        if clear_of_tie(arc(r, theta) + 2 * r, 1):
            break
    p = arc(r, theta) + 2 * r
    unit = scene[1]
    text = f"{_head(scene, r, theta)} Find the distance all the way round its edge (its perimeter), correct to 1 decimal place."
    slips = [
        (arc(r, theta), "r4g-radii-left-out-of-perimeter"),
        (arc(r, theta) + r, "r4g-one-radius-in-perimeter"),
        (arc(r, theta) + 2 * PI * r, "r4g-whole-circumference"),
        (2 * PI * r + 2 * r, "r4g-whole-circumference"),
        (arc(r, theta) + 4 * r, "r4g-diameter-added-twice"),
    ]
    working = [
        f"Arc = {theta}/360 x 2 x π x {int(r)} = {number_text(arc(r, theta), 1)} {unit}.",
        f"Perimeter = arc + 2 radii = {number_text(arc(r, theta), 1)} + {2 * int(r)} = {number_text(p, 1)} {unit}.",
    ]
    return single_number(text, p, 1, working, slips)


def _medium(rng):
    return _medium_arc(rng) if rng.random() < 0.5 else _medium_perimeter(rng)


def _hard_radius(rng):
    for _ in attempts(500, "a radius that survives rounding the area"):
        scene, r, theta = draw_sector(rng, radii=(5, 20), tenths=True)
        a2 = number_text(area(r, theta), 2)
        back = math.sqrt(float(Fraction(a2)) * 360 / (math.pi * theta))
        if abs(back - float(r)) < 0.02 and r.denominator != 1:
            break
    unit = scene[1]
    text = (
        f"{scene[0]} with a centre angle of {theta}°. Its area is {a2} {unit}². "
        f"Find the radius, correct to 1 decimal place. Use the π button on your calculator."
    )
    k = Fraction(a2)
    slips = [
        (k * 360 / (PI * theta), "r4g-square-root-not-taken"),
        (Fraction(math.sqrt(float(k) * 360 / theta)), "r4g-pi-left-out"),
        (Fraction(math.sqrt(float(k) * 180 / (math.pi * theta))), "r4g-angle-fraction-over-180"),
        (Fraction(math.sqrt(float(k) / math.pi)), "r4g-angle-fraction-left-out"),
        (2 * r, "r4g-diameter-for-radius"),
    ]
    working = [
        f"Area = {theta}/360 x π x r², so r² = {a2} x 360 / ({theta} π).",
        f"r = √({number_text(k * 360 / (PI * theta), 1)}) = {number_text(r, 1)} {unit}.",
    ]
    check(abs(float(area(r, theta)) - float(k)) < 0.0051, "area does not match radius")
    return single_number(text, r, 1, working, slips)


def _hard_angle(rng):
    for _ in attempts(500, "an angle that survives rounding the arc"):
        scene, r, theta = draw_sector(rng, radii=(5, 20))
        if scene[2] < 280 and theta > 180 and rng.random() < 0.5:
            continue
        l2 = number_text(arc(r, theta), 2)
        back = float(Fraction(l2)) * 360 / (2 * math.pi * float(r))
        if abs(back - theta) < 0.2:
            break
    unit = scene[1]
    text = (
        f"{scene[0]} with radius {int(r)} {unit}. The curved edge (the arc) is {l2} {unit} long. "
        f"Find the centre angle of the sector, to the nearest degree."
    )
    k = Fraction(l2)
    slips = [
        (k * 360 / (PI * r), "r4g-circumference-without-the-2"),
        (k / r, "r4g-radians-given-as-degrees"),
        (k * 360 / (2 * PI * r * r), "r4g-radius-squared-for-arc"),
        (k * 180 / (PI * r), "r4g-angle-halved"),
        (k * 360 / (2 * PI), "r4g-radius-left-out"),
    ]
    working = [f"Arc = θ/360 x 2π x {int(r)}, so θ = {l2} x 360 / (2π x {int(r)}) = {theta}°."]
    return single_number(text, theta, 0, working, [(v, t) for v, t in slips if Fraction(v) <= 720])


def _hard_path(rng):
    for _ in attempts(500, "a path area clear of a rounding tie"):
        scene, r, theta = draw_sector(rng, radii=(4, 14))
        wide = Fraction(rng.randint(2, 5))
        big = r + wide
        a = Fraction(theta, 360) * PI * (big * big - r * r)
        if clear_of_tie(a, 1):
            break
    unit = scene[1]
    text = (
        f"A curved path is the region between two arcs with the same centre. The inner arc has radius {int(r)} {unit}, the "
        f"outer arc has radius {int(big)} {unit}, and the angle at the centre is {theta}°. "
        "Find the area of the path, correct to 1 decimal place."
    )
    check(abs(float(a) - float(area(big, theta) - area(r, theta))) < 1e-9, "path area re-derivation differs")
    slips = [
        (Fraction(theta, 360) * PI * (big - r) ** 2, "r4g-radii-subtracted-before-squaring"),
        (area(big, theta), "r4g-outer-sector-only"),
        (Fraction(theta, 360) * PI * wide, "r4g-width-for-area"),
        (area(big, theta) + area(r, theta), "r4g-sectors-added-not-subtracted"),
        (Fraction(theta, 360) * 2 * PI * (big - r), "r4g-arc-formula-for-area"),
    ]
    working = [
        f"Path = outer sector - inner sector = {theta}/360 x π x ({int(big)}² - {int(r)}²).",
        f"= {number_text(a, 1)} {unit}².",
    ]
    return single_number(text, a, 1, working, slips)


def _hard(rng):
    return rng.choice((_hard_radius, _hard_angle, _hard_path))(rng)


_TIER = {"easy": _easy, "medium": _medium, "hard": _hard}


@generator(
    "r4g_sector_degrees",
    level=11,
    presets={"easy": {"tier": "easy"}, "medium": {"tier": "medium"}, "hard": {"tier": "hard"}},
)
def r4g_sector_degrees(rng, tier="easy"):
    return retry(_TIER[tier], rng)
