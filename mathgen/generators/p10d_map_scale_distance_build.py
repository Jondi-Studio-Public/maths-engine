"""Map-scale distance problems (senior-use-scales-to-find-distances-such-as-on / ACMEM083).

Topic: ``map_scale_distance``. Three tiers, distinct reasoning shapes:

- easy   (``verbal_forward``): "1 cm represents {d} km/m" -- a verbal scale
  applied FORWARD, one multiplication step. The real-world unit is already
  named by the scale statement, so no unit conversion is needed.
- medium (``ratio_forward``): a unitless ratio scale "1 : n" applied
  FORWARD. Because the ratio names no unit, the learner must compute the
  raw centimetre figure via the ratio and THEN convert that figure into a
  sensible real-world unit (m or km, chosen by magnitude) -- the genuine
  escalation over easy.
- hard   (``ratio_reverse``): the same unitless ratio scale, REVERSE
  direction: the real-world distance is given and the map distance (always
  a plain whole number of centimetres) is asked for -- division by the
  ratio, the inverse of medium's multiplication, plus converting the given
  real-world unit down to centimetres first.

Every value drawn here is an exact rational multiple of 0.1 by construction
(curated pools, never a free-form random range) -- see the plausibility
test in ``tests/test_p10d_map_scale_distance_build.py``, which proves this rather than
merely asserting it. Every check
below is exact ``Fraction`` equality; there is no float rounding anywhere
in this generator.
"""

from __future__ import annotations

import math
from fractions import Fraction

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    exact_decimal_string,
    fixed_point_string,
    generator,
    numeric_answer,
    pick_distractors,
    stated_precision_tolerance,
)

MAP_CONTEXTS = [
    "road map",
    "street map",
    "bushwalking map",
    "online map",
    "cadastral map",
]

# Verbal-scale values for the easy tier: "1 cm on the map represents {d} km"
# / "{d} m" -- real bushwalking/road-map and street-map scale statements.
KM_SCALES = [1, 2, 3, 4, 5, 10, 15, 20, 25]
M_SCALES = [10, 20, 25, 50, 100, 200, 250, 500]

# A map distance measurable with a ruler: 1.0 .. 30.0 cm in 0.5 cm steps
# (integer or one-decimal-place, per the skill's easy-tier contract).
_MAP_DISTANCES_HALF = [Fraction(h, 2) for h in range(2, 61)]

# Ratio scales "1 : n" for the medium/hard tiers. Every value is a multiple
# of 1000 whose own quotient by 1000 is a multiple of 10 -- chosen so every
# forward computation (an integer map distance 1..30 cm, times n) lands on
# a real-world figure that terminates within 1 decimal place in EITHER unit
# (m or km), proved by the plausibility test, never 2+. Every value sits
# inside the [5000, 250000] range; 5000 itself is deliberately
# excluded -- pairing it with an odd map distance would need 2 decimal
# places in km, which the plausibility test's RED case demonstrates.
RATIO_SCALES = [10000, 20000, 50000, 100000, 150000, 200000, 250000]

# Map distances for medium/hard: plain integers, 1..30 cm (the stated
# bound), so the hard tier's answer (a map distance) is always a whole
# number of centimetres -- no decimal-place instruction ever needed there.
MAP_DISTANCES_INT = list(range(1, 31))


def _easy_options(scales: list[int], max_product: int = 999) -> list[tuple[int, Fraction]]:
    """(scale, map_distance) pairs whose real-world figure, in the SAME
    unit the verbal scale already names, stays inside [1, max_product] --
    the plausibility bound, applied here as a fixed, auditable
    pool rather than trusted to a runtime resample."""
    return [(scale, md) for scale in scales for md in _MAP_DISTANCES_HALF if 1 <= scale * md <= max_product]


EASY_KM_OPTIONS = _easy_options(KM_SCALES)
EASY_M_OPTIONS = _easy_options(M_SCALES)


def _trim(value: Fraction) -> str:
    """Natural display of a GIVEN fact (never the graded answer): no forced
    decimal point on a whole number."""
    return str(int(value)) if value.denominator == 1 else exact_decimal_string(value, 1)


def _forced_decimal_string(value: Fraction, dp: int) -> str:
    """The GRADED answer's own text, forced to exactly `dp` decimal places
    (e.g. dp=1 renders a whole number as "75.0", never trimmed to "75") --
    the question explicitly asks for that many decimal places whenever
    `dp > 0` here, so this is the requested precision, not a spurious
    trailing zero. `exact_decimal_string` is deliberately NOT used for this:
    it returns the SHORTEST terminating form, which would silently drop the
    requested decimal place for a seed that happens to land on a whole
    number, exactly the mismatch the plausibility/oracle tests caught."""
    scaled = value * (10**dp)
    check(scaled.denominator == 1, f"{value} does not terminate at {dp} dp")
    return fixed_point_string(scaled.numerator, dp)


def _forced_distractor_string(value: Fraction, dp: int) -> str:
    """A DISTRACTOR's text, forced to the SAME `dp` the tier's own answer is
    forced to (never trimmed) -- rounds half-up if `value` does not
    terminate that finely, since a misconception's raw value has no reason
    to land exactly on the answer's own precision. Kept deliberately
    separate from `rounded_decimal_string` (which returns the SHORTEST
    terminating/rounded form): that shortest-form behaviour is exactly what
    made a whole-number distractor render without the decimal point the
    tier's own answer is forced to carry."""
    scaled = value * (10**dp)
    if scaled.denominator != 1:
        scaled = Fraction(math.floor(scaled + Fraction(1, 2)))
    return fixed_point_string(int(scaled), dp)


def _fmt_map_distance(md: Fraction) -> str:
    return _trim(md)


def _real_distance_and_unit(n: int, map_distance: int) -> tuple[Fraction, str]:
    """The real-world distance for ratio 1:n and a map distance of
    `map_distance` cm, in whichever of m/km keeps the figure sensible
    (never requiring more than 3 significant figures to state honestly)."""
    real_m = Fraction(n * map_distance, 100)
    if real_m < 1000:
        return real_m, "m"
    return real_m / 1000, "km"


def _dp_for_unit(unit: str) -> int:
    return 0 if unit == "m" else 1


def _article_for(context: str) -> str:
    return "an" if context[0] in "aeiou" else "a"


def _avoid_collision(value: Fraction, answer: Fraction, nudge: Fraction = Fraction(1)) -> Fraction:
    """A handful of RATIO_SCALES values (n=100000 in particular, where 1 cm
    of map = 1 km of reality) make the raw map-distance NUMBER coincide
    exactly with the real-world figure's number for a specific misconception
    formula, even though the two are conceptually different quantities in
    different units -- e.g. n=100000, map_distance=6 cm gives a real-world
    distance of exactly 6 km. That is a genuine coincidence of this domain,
    not a formula bug, so a one-unit nudge (still a value nobody would
    mistake for a *correct* derivation) keeps the distractor a distinct
    option rather than raising a false verification failure."""
    return value + nudge if value == answer else value


def _verify_forward(n: int, map_distance: int, real_value: Fraction, unit: str) -> None:
    """Independent route for the FORWARD direction: convert the map
    distance to MILLIMETRES first (a pivot unit construction never uses),
    scale by n to real millimetres, then divide down to the target unit --
    re-derives the same figure via a different unit chain from
    construction's own centimetre-then-metre one."""
    real_mm = Fraction(map_distance) * 10 * n
    real_m = real_mm / 1000
    expected = real_m if unit == "m" else real_m / 1000
    check_equivalent(expected, real_value, "map-scale forward conversion mismatch")


def _verify_reverse(n: int, map_distance: int, real_value: Fraction, unit: str) -> None:
    """Independent route for the REVERSE direction: convert the DISPLAYED
    real-world figure back to centimetres via the unit's own factor, divide
    by n, and confirm the result recovers `map_distance` exactly -- the
    direction a learner actually performs, independent of how construction
    derived `real_value` forward from `map_distance`."""
    real_cm = real_value * (100 if unit == "m" else 100_000)
    check(real_cm.denominator == 1, f"{real_cm} is not a whole number of centimetres")
    recomputed = real_cm / n
    check_equivalent(recomputed, map_distance, "map-scale reverse conversion mismatch")


# ---------------------------------------------------------------------------
# Easy: verbal scale, forward direction
# ---------------------------------------------------------------------------


def _verbal_forward(rng) -> dict:
    unit = rng.choice(["km", "m"])
    options = EASY_KM_OPTIONS if unit == "km" else EASY_M_OPTIONS
    scale, md = rng.choice(options)
    context = rng.choice(MAP_CONTEXTS)

    real_value = Fraction(scale) * md
    check_equivalent(md * scale, real_value, "verbal-scale forward mismatch")

    answer_text = _forced_decimal_string(real_value, 1)
    answer = numeric_answer(real_value)
    md_str = _fmt_map_distance(md)

    text = (
        f"On {_article_for(context)} {context}, a scale of 1 cm to {scale} {unit} "
        f"is used. Two landmarks are {md_str} cm apart on the map. What is the "
        f"actual distance between the landmarks, in {unit}, to 1 decimal place?"
    )

    ignores_the_scale = _avoid_collision(Fraction(md), real_value)
    shifts_decimal_point = _avoid_collision(real_value * 10, real_value)
    halves_the_real_world_distance = _avoid_collision(real_value / 2, real_value)

    check_distinct(real_value, ignores_the_scale, shifts_decimal_point, halves_the_real_world_distance)
    candidates = [
        _forced_distractor_string(ignores_the_scale, 1),
        _forced_distractor_string(shifts_decimal_point, 1),
        _forced_distractor_string(halves_the_real_world_distance, 1),
    ]
    distractors = pick_distractors(
        answer_text,
        candidates,
        tolerance=stated_precision_tolerance(1),
    )

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# Medium: ratio scale, forward direction, forced unit conversion
# ---------------------------------------------------------------------------


def _ratio_forward(rng) -> dict:
    n = rng.choice(RATIO_SCALES)
    map_distance = rng.choice(MAP_DISTANCES_INT)
    context = rng.choice(MAP_CONTEXTS)

    real_value, unit = _real_distance_and_unit(n, map_distance)
    _verify_forward(n, map_distance, real_value, unit)
    dp = _dp_for_unit(unit)

    answer_text = _forced_decimal_string(real_value, dp)
    answer = numeric_answer(real_value)

    dp_phrase = ", to 1 decimal place" if dp == 1 else ""
    text = (
        f"{_article_for(context).capitalize()} {context} uses a scale of "
        f"1 : {n}. The distance between two points on the map is "
        f"{map_distance} cm. What is the actual distance between the "
        f"points, in {unit}{dp_phrase}?"
    )

    real_cm = n * map_distance
    # forgets-unit-conversion: reports the intermediate, less-converted figure -- the raw
    # ratio-scaled centimetre figure when the branch needed the METRE conversion (off by a factor of
    # 100), or the metre figure when the branch needed the further METRE->KM conversion (off by a
    # factor of 1000) -- never a value that collapses to a degenerate near-zero, which a raw "divide
    # by n" mistake at this scale always would.
    if unit == "km":
        forgets_unit_conversion = Fraction(n * map_distance, 100)  # the metres figure
    else:
        forgets_unit_conversion = Fraction(real_cm)  # the raw centimetres figure
    forgets_unit_conversion = _avoid_collision(forgets_unit_conversion, real_value)
    # ignores the scale entirely: reports the raw map distance as if it
    # were already the real-world distance, unconverted.
    ignores_the_scale = _avoid_collision(Fraction(map_distance), real_value)
    # mixes up which power-of-ten conversion applies (a factor of 10 off
    # from the correct answer).
    wrong_conversion_factor = _avoid_collision(real_value * 10, real_value)

    check_distinct(real_value, forgets_unit_conversion, ignores_the_scale, wrong_conversion_factor)
    candidates = [
        _forced_distractor_string(forgets_unit_conversion, dp),
        _forced_distractor_string(ignores_the_scale, dp),
        _forced_distractor_string(wrong_conversion_factor, dp),
    ]
    tolerance = stated_precision_tolerance(dp) if dp else 0
    distractors = pick_distractors(answer_text, candidates, tolerance=tolerance)

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# Hard: ratio scale, REVERSE direction
# ---------------------------------------------------------------------------


def _ratio_reverse(rng) -> dict:
    n = rng.choice(RATIO_SCALES)
    map_distance = rng.choice(MAP_DISTANCES_INT)  # this tier's ANSWER
    context = rng.choice(MAP_CONTEXTS)

    real_value, unit = _real_distance_and_unit(n, map_distance)
    _verify_reverse(n, map_distance, real_value, unit)
    real_display = _trim(real_value)

    answer_text = str(map_distance)
    answer = map_distance

    text = (
        f"{_article_for(context).capitalize()} {context} uses a scale of "
        f"1 : {n}. The real-world distance between two towns is "
        f"{real_display} {unit}. What is the distance between the towns "
        f"on the map, in cm?"
    )

    real_cm = real_value * (100 if unit == "m" else 100_000)
    check(real_cm.denominator == 1, f"{real_cm} is not a whole number of centimetres")
    answer_frac = Fraction(map_distance)
    # multiplies-instead-of-divides.
    multiplies_instead = _avoid_collision(real_cm * n, answer_frac)
    # forgets-unit-conversion: reports the given real-world number directly as if it were already
    # the map distance in cm -- forgetting BOTH the unit conversion and the ratio division. Dividing
    # by n at this ratio's scale (10,000-250,000) instead collapses any partial-conversion mistake
    # to a degenerate near-zero value that rounds to 0, so this is the non-degenerate substitute.
    forgets_unit_conversion = _avoid_collision(real_value, answer_frac)
    # mixes up which power-of-ten conversion applies: divides by n/10
    # instead of n (a factor of 10 off from the correct ratio).
    wrong_conversion_factor = _avoid_collision(real_cm / (Fraction(n, 10)), answer_frac)

    check_distinct(
        answer_frac,
        multiplies_instead,
        forgets_unit_conversion,
        wrong_conversion_factor,
    )
    candidates = [
        _forced_distractor_string(multiplies_instead, 0),
        _forced_distractor_string(forgets_unit_conversion, 0),
        _forced_distractor_string(wrong_conversion_factor, 0),
    ]
    distractors = pick_distractors(answer_text, candidates, tolerance=0)

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        distractors=distractors,
    )


@generator(
    "map_scale_distance",
    level=11,
    presets={
        "easy": dict(mode="verbal_forward"),
        "medium": dict(mode="ratio_forward"),
        "hard": dict(mode="ratio_reverse"),
    },
)
def map_scale_distance(rng, mode: str = "verbal_forward") -> dict:
    if mode == "verbal_forward":
        return _verbal_forward(rng)
    if mode == "ratio_forward":
        return _ratio_forward(rng)
    if mode == "ratio_reverse":
        return _ratio_reverse(rng)
    raise ValueError(f"unknown mode {mode!r}")
