"""Area of a triangle from its three sides: `r3j_heron` (easy, medium, hard) and the exam chain `exd_x_r3j_heron_plot`.

``triangle-area-heron`` (``measurement``, answer ``number``). The three sides of a triangular region are given and no
height is; the semi-perimeter is never handed over.

* easy: a triangle whose three sides give a whole-number area (13, 14, 15 and its relatives, scaled): the area.
* medium: any three sides that make a triangle, the area to 1 decimal place.
* hard: the perpendicular height on the longest side (to 1 decimal place), or the cost of covering the region at a
  stated price per square metre (to the nearest dollar).
* exam (``exd_x_r3j_heron_plot``): the semi-perimeter, the area, the height on the longest side and the cost of
  surfacing, each part following through from the earlier ones.

The area is rebuilt a second way, from the angle between two sides (cosine rule, then half the product of the two
sides and the sine), rather than from the same formula.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import isqrt, sqrt

from ..core import attempts, check, generator, phrase
from ._exd_distinct_parts import distinct_part_answers
from ._r3j_kit import dollars, number_part, number_text, single_number

# (the region, its smallest and largest plausible side in metres)
_PLACES = [
    ("a triangular sandpit", 3, 12),
    ("a triangular garden bed", 4, 25),
    ("a triangular roof gable", 4, 16),
    ("a triangular sail", 5, 30),
    ("a triangular courtyard", 8, 50),
    ("a triangular car park", 15, 80),
    ("a triangular paddock", 60, 300),
    ("a triangular reserve beside a creek", 60, 300),
]


@lru_cache(maxsize=1)
def _whole_area_triples():
    """Primitive and scaled-down triples (a <= b <= c, c <= 30) with a whole-number area, not right-angled."""
    out = []
    for a in range(3, 31):
        for b in range(a, 31):
            for c in range(b, min(a + b, 31)):
                n = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)  # 16 * area squared
                root = isqrt(n)
                if root * root == n and root % 4 == 0 and a * a + b * b != c * c and c < a + b:
                    out.append((a, b, c))
    return tuple(out)


def _sixteen_a2(a, b, c):
    return (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)


def area_exact_or_float(a, b, c):
    """The area from Heron's formula: a Fraction when 16 A^2 is a perfect square, else a float."""
    n = _sixteen_a2(a, b, c)
    root = isqrt(n)
    return Fraction(root, 4) if root * root == n else sqrt(n) / 4


def _area_by_angle(a, b, c) -> float:
    """The area from half ab sin C, with C found from the cosine rule (a different route from Heron's formula)."""
    cos_c = (a * a + b * b - c * c) / (2 * a * b)
    return 0.5 * a * b * sqrt(1 - cos_c * cos_c)


def _area(a, b, c):
    area = area_exact_or_float(a, b, c)
    check(abs(float(area) - _area_by_angle(a, b, c)) < 1e-6 * max(1.0, float(area)), "area routes disagree")
    return area


def _as_fraction(x) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x).limit_denominator(10**9)


def _clear_of_tie(value, places) -> bool:
    """True when `value` is not close to a rounding boundary at `places` places."""
    return abs((float(value) * 10**places) % 1 - 0.5) > 0.03


def _slips_for_area(a, b, c, area):
    s = Fraction(a + b + c, 2)
    perimeter = a + b + c
    return [
        (
            sqrt(perimeter * (perimeter - a) * (perimeter - b) * (perimeter - c)) if perimeter > max(a, b, c) else 0,
            "r3j-heron-perimeter-used-as-semi",
        ),
        (float(s * (s - a) * (s - b) * (s - c)), "r3j-heron-root-left-out"),
        (sqrt(float(s * (s - a) * (s - b) * (s - c)) / s), "r3j-heron-one-factor-dropped"),
        (float(s), "r3j-heron-semi-perimeter-given"),
        (a * b / 2, "r3j-heron-half-product-of-two-sides"),
        (float(area) * 2, "r3j-heron-halving-forgotten"),
    ]


def _sides_text(a, b, c):
    return f"{a} m, {b} m and {c} m"


def _easy(rng):
    for _ in attempts(500, "a region and a whole-area triangle that fit it"):
        place, lo, hi = rng.choice(_PLACES)
        a, b, c = rng.choice(_whole_area_triples())
        scale = rng.choice([1, 1, 2, 3, 4, 5, 10])
        a, b, c = a * scale, b * scale, c * scale
        if lo <= a and c <= hi:
            break
    sides = [a, b, c]
    rng.shuffle(sides)
    area = _area(a, b, c)
    check(Fraction(area).denominator == 1, "easy area must be whole")
    text = f"{place[0].upper()}{place[1:]} has sides of {_sides_text(*sides)}. Find its area in square metres."
    s = Fraction(a + b + c, 2)
    working = [
        f"s = ({a} + {b} + {c}) / 2 = {plain_num(s)}.",
        f"Area = sqrt({plain_num(s)} x {plain_num(s - a)} x {plain_num(s - b)} x {plain_num(s - c)}) = {int(area)} m².",
    ]
    return single_number(text, Fraction(area), 0, working, _slips_for_area(a, b, c, area))


def plain_num(x):
    x = Fraction(x)
    return str(int(x)) if x.denominator == 1 else f"{float(x):g}"


def _draw_sides(rng, lo, hi, obtuse=None):
    lo = max(lo, 3)
    for _ in attempts(2000, "three sides with a non-whole area clear of a rounding tie"):
        a, b, c = sorted(rng.randint(lo, hi) for _ in range(3))
        if a + b <= c + 1 or len({a, b, c}) < 2:
            continue
        n = _sixteen_a2(a, b, c)
        if isqrt(n) ** 2 == n:
            continue
        is_obtuse = a * a + b * b < c * c
        if obtuse is not None and is_obtuse != obtuse:
            continue
        area = float(area_exact_or_float(a, b, c))
        if not (_clear_of_tie(area, 1) and _clear_of_tie(2 * area / c, 1)):
            continue
        sides = [a, b, c]
        rng.shuffle(sides)
        return a, b, c, sides
    raise AssertionError  # pragma: no cover


def _medium_units(rng):
    """Sides written in two units: one in metres, two in centimetres. The area is asked in square metres."""
    things = [
        "a triangular banner",
        "a triangular tablecloth",
        "a triangular shade sail",
        "a triangular garden trellis",
    ]
    for _ in attempts(2000, "a triangle with a non-whole area clear of a rounding tie"):
        a, b, c = sorted(rng.randint(80, 260) for _ in range(3))
        if a + b <= c + 5 or isqrt(_sixteen_a2(a, b, c)) ** 2 == _sixteen_a2(a, b, c):
            continue
        area_cm2 = float(_area(a, b, c))
        if _clear_of_tie(area_cm2 / 10**4, 2):
            break
    area_m2 = area_cm2 / 10**4
    shown = [a, b, c]
    rng.shuffle(shown)
    in_metres = shown[0]
    # one side is written in metres (a whole number of centimetres is a decimal number of metres)
    text_sides = f"{in_metres / 100:g} m, {shown[1]} cm and {shown[2]} cm"
    thing = rng.choice(things)
    text = (
        f"{thing[0].upper()}{thing[1:]} has sides of {text_sides}. Find its area in square metres, correct to "
        "2 decimal places."
    )
    s_half = Fraction(a + b + c, 2)
    slips = [
        (area_cm2, "r3j-heron-cm2-given"),
        (area_cm2 / 100, "r3j-heron-area-converted-with-linear-factor"),
        (area_cm2 / 10**3, "r3j-heron-area-converted-with-cube-factor"),
        (float(s_half * (s_half - a) * (s_half - b) * (s_half - c)) / 10**4, "r3j-heron-root-left-out"),
        (area_m2 * 2, "r3j-heron-halving-forgotten"),
    ]
    working = [
        f"In centimetres the sides are {a} cm, {b} cm and {c} cm, so s = {plain_num(s_half)} cm.",
        f"Area = sqrt({plain_num(s_half)} x {plain_num(s_half - a)} x {plain_num(s_half - b)} x {plain_num(s_half - c)}) = "
        f"{area_cm2:.1f} cm².",
        f"1 m² = 10 000 cm², so the area is {number_text(_as_fraction(area_m2), 2)} m².",
    ]
    return single_number(text, _as_fraction(area_m2), 2, working, slips)


def _medium(rng):
    if rng.random() < 0.4:
        return _medium_units(rng)
    place, lo, hi = rng.choice(_PLACES)
    a, b, c, sides = _draw_sides(rng, lo, hi, obtuse=rng.choice([True, False, False]))
    area = _area(a, b, c)
    text = (
        f"{place[0].upper()}{place[1:]} has sides of {_sides_text(*sides)}. "
        "Find its area in square metres, correct to 1 decimal place."
    )
    s = Fraction(a + b + c, 2)
    working = [
        f"s = ({a} + {b} + {c}) / 2 = {plain_num(s)}.",
        f"Area = sqrt({plain_num(s)} x {plain_num(s - a)} x {plain_num(s - b)} x {plain_num(s - c)}) = "
        f"{number_text(_as_fraction(area), 1)} m².",
    ]
    return single_number(text, _as_fraction(area), 1, working, _slips_for_area(a, b, c, area))


def _hard_height(rng):
    place, lo, hi = rng.choice(_PLACES)
    a, b, c, sides = _draw_sides(rng, lo, hi)
    area = float(_area(a, b, c))
    h = 2 * area / c
    text = (
        f"{place[0].upper()}{place[1:]} has sides of {_sides_text(*sides)}. A straight line is marked from the "
        "corner opposite the longest side to that side, meeting it at right angles. How long is the line, in "
        "metres, correct to 1 decimal place?"
    )
    slips = [
        (area / c, "r3j-heron-height-halving-forgotten"),
        (2 * area / a, "r3j-heron-height-on-shortest-side"),
        (2 * area / b, "r3j-heron-height-on-middle-side"),
        (area, "r3j-heron-area-given-as-height"),
        (sqrt(max(b * b - (c / 2) ** 2, 1)), "r3j-heron-assumed-isosceles"),
    ]
    working = [
        f"Area = {number_text(_as_fraction(area), 1)} m² (Heron's formula).",
        f"Area = (1/2) x {c} x height, so height = 2 x {number_text(_as_fraction(area), 3)} / {c} = "
        f"{number_text(_as_fraction(h), 1)} m.",
    ]
    return single_number(text, _as_fraction(h), 1, working, slips)


def _hard_cost(rng):
    for _ in attempts(500, "a cost clear of a rounding tie"):
        use = rng.choice(
            [
                ("a triangular courtyard", "paved", "per square metre", 8, 40),
                ("a triangular lawn", "turfed", "per square metre", 8, 40),
                ("a triangular garden bed", "mulched", "per square metre", 4, 25),
                ("a triangular car park", "sealed", "per square metre", 15, 80),
            ]
        )
        a, b, c, sides = _draw_sides(rng, use[3], use[4])
        area = float(_area(a, b, c))
        rate = rng.choice([8, 12, 15, 18, 22, 25, 30])
        cost = area * rate
        if _clear_of_tie(cost, 0):
            break
    text = (
        f"{use[0][0].upper()}{use[0][1:]} has sides of {_sides_text(*sides)}. It is to be {use[1]} at "
        f"{dollars(rate)} {use[2]}. What is the cost, to the nearest dollar?"
    )
    s = Fraction(a + b + c, 2)
    slips = [
        (float(s * (s - a) * (s - b) * (s - c)) * rate, "r3j-heron-root-left-out"),
        (area * rate * 2, "r3j-heron-halving-forgotten"),
        (a * b / 2 * rate, "r3j-heron-half-product-of-two-sides"),
        (float(s) * rate, "r3j-heron-semi-perimeter-given"),
        ((a + b + c) * rate, "r3j-heron-perimeter-priced"),
    ]
    working = [
        f"Area = sqrt({plain_num(s)} x {plain_num(s - a)} x {plain_num(s - b)} x {plain_num(s - c)}) = {area:.3f} m².",
        f"Cost = {area:.3f} x {rate} = {dollars(Fraction(cost).limit_denominator(10**6), 2)}, which is "
        f"{dollars(round(cost))} to the nearest dollar.",
    ]
    return single_number(text, _as_fraction(cost), 0, working, slips)


@generator(
    "r3j_heron",
    level=10,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3j_heron(rng, tier="easy"):
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return rng.choice([_hard_height, _hard_cost])(rng)


# -- exam ---------------------------------------------------------------------------------------------------------


@generator("exd_x_r3j_heron_plot", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3j_heron_plot(rng, tier="exam"):
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a plot whose area, height and cost are clear of rounding ties"):
        place, lo, hi = rng.choice(_PLACES[4:6])
        a, b, c, sides = _draw_sides(rng, lo, min(hi, 60))
        area = float(_area(a, b, c))
        rate = rng.choice([8, 12, 15, 18])
        cost = area * rate
        if _clear_of_tie(cost, 0):
            break
    s = Fraction(a + b + c, 2)
    h = 2 * area / c
    text = phrase(
        rng,
        [
            "{Place} has sides of {sides}. It is to be surfaced at {rate} per square metre.",
            "The sides of {place} measure {sides}. Surfacing it costs {rate} per square metre.",
            "A council plans to resurface {place} with sides of {sides}, at a cost of {rate} per square metre.",
        ],
        Place=f"{place[0].upper()}{place[1:]}",
        place=place,
        sides=_sides_text(*sides),
        rate=dollars(rate),
    )
    parts = [
        number_part(
            "Find the semi-perimeter of the plot, in metres.",
            s,
            1,
            1,
            [f"s = ({a} + {b} + {c}) / 2 = {plain_num(s)} m."],
            [
                (a + b + c, "r3j-heron-perimeter-used-as-semi"),
                (Fraction(a + b, 2), "r3j-heron-semi-of-two-sides"),
                (Fraction(a + b + c, 3), "r3j-heron-divided-by-three"),
            ],
            follow_through=None,
        ),
        number_part(
            "Find the area of the plot, in square metres, correct to 1 decimal place.",
            _as_fraction(area),
            1,
            2,
            [
                f"Area = sqrt({plain_num(s)} x {plain_num(s - a)} x {plain_num(s - b)} x {plain_num(s - c)}) = "
                f"{number_text(_as_fraction(area), 1)} m²."
            ],
            _slips_for_area(a, b, c, area),
            follow_through=f"sqrt(part_a*(part_a-{a})*(part_a-{b})*(part_a-{c}))",
        ),
        number_part(
            "A drain is to run from the corner opposite the longest side to that side, at right angles to it. "
            "How long is the drain, in metres, correct to 1 decimal place?",
            _as_fraction(h),
            1,
            2,
            [f"Height = 2 x {number_text(_as_fraction(area), 3)} / {c} = {number_text(_as_fraction(h), 1)} m."],
            [
                (area / c, "r3j-heron-height-halving-forgotten"),
                (2 * area / a, "r3j-heron-height-on-shortest-side"),
                (2 * area / b, "r3j-heron-height-on-middle-side"),
                (area, "r3j-heron-area-given-as-height"),
            ],
            follow_through=f"2*part_b/{c}",
        ),
        number_part(
            "Find the cost of surfacing the plot, to the nearest dollar.",
            _as_fraction(cost),
            0,
            2,
            [f"Cost = {number_text(_as_fraction(area), 3)} x {rate} = {dollars(round(cost))}."],
            [
                (float(s * (s - a) * (s - b) * (s - c)) * rate, "r3j-heron-root-left-out"),
                (area * rate * 2, "r3j-heron-halving-forgotten"),
                ((a + b + c) * rate, "r3j-heron-perimeter-priced"),
                (a * b / 2 * rate, "r3j-heron-half-product-of-two-sides"),
            ],
            follow_through=f"{rate}*part_b",
        ),
    ]
    return dict(text=text, parts=parts)
