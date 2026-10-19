"""- ``annulus_cross_section_area`` (``annulus-cross-section-area``, domain
  measurement, answer: number) -- area of a hollow shaft/pipe/washer
  cross-section: ``pi * (R**2 - r**2)``, ``R`` outer radius, ``r`` inner
  radius.
- ``frustum_volume`` (``frustum-volume``, domain measurement, answer: number)
  -- volume of a frustum of a cone: ``(1/3) * pi * h * (R**2 + R*r + r**2)``,
  ``R`` base radius, ``r`` top radius, ``h`` height.

The grader's tolerance is derived from what ``answer_text`` itself states
(``mathgen/grading/typed.py:_typed_number_tolerance``), so stating 1dp gives a 0.05 tolerance -- comfortable
headroom against the ~2.65e-6 relative error between 3.14159 and true pi, which is too close for comfort at
2dp.

Constructibility. Every
seed, every tier:
  - annulus: ``r`` strictly less than ``R``, AND ``r`` in ``[0.2R, 0.9R]`` --
    a wall thin enough to look like a crack, or thick enough to be a
    pinhole, is technically an annulus but not what a machinist would spec.
  - frustum: ``r`` strictly less than ``R``, ``h`` strictly positive, AND
    ``r`` in ``[0.2R, 0.9R]`` (near-cylinder / near-cone are technically
    frustums but not what this skill teaches) AND ``h >= 0.3R`` (a near-flat
    disc is not a solid a learner could picture).
Both bounds are swept in ``tests/test_t5_annulus_frustum_build.py`` over 60
seeds x all 3 tiers, and both were proven able to fail against a
deliberately-loosened throwaway copy of the draw logic.

Every generated value is also kept away from the 1dp rounding tie (the
x.x5 boundary where half-up rounding flips) by more than ``_tie_margin_for`` --
``_tenths_tie_distance`` measures that distance directly in the exact
``Fraction`` value, resampling on a near-tie exactly like
``p8e_percentage_review_build.py``'s own ``_pct_medium``.
"""

from __future__ import annotations

import math
from fractions import Fraction

from sympy import sqrt as sympy_sqrt

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    pick_distractors,
    round_half_up,
)

PI = Fraction(314159, 100000)

#: Tolerance for both the real-grader-facing distractor gap (half a unit in
#: the last stated decimal place, 1dp here -- not the generic
#: GRADER_TOLERANCE=0.051 constant, which is a slightly different number) and
#: for `pick_distractors`.
ANSWER_TOLERANCE = Fraction(1, 20)

# Ceiling on the raw (unrounded) answer value every draw must clear, derived, not chosen freely. A
# learner who re-derives the answer with a calculator's true-pi button gets a value differing from
# this module's `PI`-based one by a RELATIVE error of ~8.45e-7 (`abs(float(PI) - math.pi) /
# math.pi`) -- fixed regardless of magnitude, so its ABSOLUTE size grows with the answer. Measured
# directly (not assumed): re-deriving every rendered question with `math.pi` and feeding the result
# to `mathgen.grading.GRADERS["number"]` rejected 114/360 (31.7%) before this cap existed --
# frustum_volume/medium seed 2 (296301.2) differs from its own math.pi recomputation by ~0.25, five
# times the 0.05 tolerance the 1dp convention promises. Capping the raw value at 15000 keeps the
# worst-case discrepancy at ~15000*8.45e-7 =~ 0.013, comfortably inside the 0.05 tolerance.
_MAX_RAW_VALUE = Fraction(15000)


# How far (in the exact, unrounded Fraction answer) a draw must sit from the
# x.x5 half-up rounding tie before it is accepted -- MAGNITUDE-AWARE, not a
# flat constant. A flat `Fraction(1, 1000)` margin was not enough on its
# own: even under the `_MAX_RAW_VALUE` cap above, a value could sit just
# outside that flat margin from OUR tie yet still be pushed across it by
# the true-pi learner's ~8.45e-7 relative discrepancy (worst case ~0.0127 at
# the 15000 cap) -- measured directly, this flat-margin-only version still
# rejected 4/360 real math.pi recomputations. The margin below scales with
# the draw's own magnitude at 3x that relative error (2.5e-6), so at the
# 15000 cap it demands >=0.0375 clearance -- comfortably above the 0.0127
# worst case -- while staying at the original flat 1/1000 floor for small
# values where the relative term is negligible. Re-measured after this
# change: 0/360 real math.pi recomputations rejected.
def _tie_margin_for(raw: Fraction) -> Fraction:
    return max(Fraction(1, 1000), abs(raw) * Fraction(1, 400000))


_RESAMPLE_ATTEMPTS = 1000

# Pythagorean triples for the frustum "hard" tier's slant-height derivation
# (R - r, h, l). Copied verbatim from
# ``p8d_quadrilateral_theorems_build.py``'s own ``_TRIPLES``, not reinvented.
_TRIPLES = [
    (3, 4, 5),
    (6, 8, 10),
    (5, 12, 13),
    (8, 15, 17),
    (7, 24, 25),
    (9, 12, 15),
    (12, 16, 20),
    (20, 21, 29),
    (9, 40, 41),
]


def _tenths_tie_distance(frac: Fraction) -> Fraction:
    """Distance from `frac` to the nearest x.x5 half-up rounding tie point
    (the boundary between two adjacent tenths where the 2nd decimal digit is
    exactly 5) -- NOT the distance from `frac` to its own rounded tenth.

    `frac*10 + 1/2` is an exact integer precisely when `frac` sits exactly on
    a tie (see `core.round_half_up`); this returns how far, in `frac`'s
    own units, that quantity's fractional part is from the two integers either
    side of it, i.e. how far `frac` is from the nearest such tie."""
    scaled = frac * 10 + Fraction(1, 2)
    whole = scaled.numerator // scaled.denominator
    frac_part = scaled - whole
    return min(frac_part, 1 - frac_part) / 10


def _fmt1(frac: Fraction) -> str:
    """Format a Fraction rounded to 1dp the way every answer/distractor in
    this module is displayed: `f"{v:.1f}"`, never a bare-integer string
    sitting among decimals."""
    return f"{float(frac):.1f}"


def _band(outer: int) -> tuple[int, int]:
    """Inclusive integer `[0.2*outer, 0.9*outer]` band -- the plausibility
    bound shared by both skills' inner/top radius."""
    lo = math.ceil(Fraction(outer, 5))
    hi = math.floor(Fraction(9 * outer, 10))
    return lo, hi


def _in_band(outer: int, inner: int) -> bool:
    return Fraction(outer, 5) <= inner <= Fraction(9 * outer, 10)


# ============================================================================
# Skill 1: annulus_cross_section_area
# ============================================================================


def _annulus_raw_and_checks(R: int, r: int) -> Fraction:
    """The annulus area as a Fraction, term-by-term (`PI*R**2 - PI*r**2`),
    independently re-checked via the difference-of-squares factoring
    `PI*(R-r)*(R+r)` -- a distinct algebraic route from the construction."""
    raw = PI * R**2 - PI * r**2
    check_equivalent(
        raw,
        PI * (R - r) * (R + r),
        "annulus_cross_section_area: difference-of-squares check disagrees with term-by-term construction",
    )
    return raw


def _annulus_distractors(R: int, r: int, raw: Fraction, answer_text: str) -> list[str]:
    ignores_inner = PI * R**2
    subtracts_first = PI * (R - r) ** 2
    adds_instead = PI * (R**2 + r**2)

    check_distinct(raw, ignores_inner, msg="ignores-inner-circle distractor equals the answer")
    check_distinct(raw, subtracts_first, msg="subtracts-radii-first distractor equals the answer")
    check_distinct(raw, adds_instead, msg="adds-instead-of-subtracts distractor equals the answer")

    check(ignores_inner >= 0, "annulus area must not be negative")
    check(subtracts_first >= 0, "annulus area must not be negative")
    check(adds_instead >= 0, "annulus area must not be negative")
    candidates = [
        _fmt1(round_half_up(ignores_inner, 1)),
        _fmt1(round_half_up(subtracts_first, 1)),
        _fmt1(round_half_up(adds_instead, 1)),
    ]
    return pick_distractors(answer_text, candidates, tolerance=ANSWER_TOLERANCE)


@generator(
    "annulus_cross_section_area",
    level=12,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="mixed_units"),
        "hard": dict(variant="thickness_units"),
    },
)
def annulus_cross_section_area(rng, variant="direct"):
    """Area of a hollow-shaft cross-section: pi * (R^2 - r^2)."""

    if variant == "direct":
        # Easy: both R and r given directly as radii, same unit (cm).
        found = False
        for _ in range(_RESAMPLE_ATTEMPTS):
            R = rng.randint(10, 50)
            lo, hi = _band(R)
            if lo > hi:
                continue
            r = rng.randint(lo, hi)
            if r >= R:
                continue
            raw = PI * R**2 - PI * r**2
            if raw > _MAX_RAW_VALUE:
                continue
            if _tenths_tie_distance(raw) <= _tie_margin_for(raw):
                continue
            found = True
            break
        check(found, "annulus_cross_section_area/direct: no valid (R, r) combo found")

        check(r < R, "annulus_cross_section_area/direct: r not strictly less than R")
        check(_in_band(R, r), "annulus_cross_section_area/direct: r outside plausibility band")

        raw = _annulus_raw_and_checks(R, r)
        check(raw >= 0, "annulus area must not be negative")
        rounded = round_half_up(raw, 1)
        answer_text = _fmt1(rounded)

        question_text = (
            f"A hollow cylindrical shaft has an outer radius of {R} cm and an "
            f"inner radius of {r} cm. Taking π as 3.14159, what is the "
            f"area of its annular cross-section, in square centimetres, to 1 "
            f"decimal place?"
        )
        working = [
            "Area of an annulus = π × (R² − r²), where R is the outer radius and r is the inner radius.",
            f"Area = 3.14159 × ({R}² − {r}²) = 3.14159 × {R * R - r * r} = {float(raw):.5f} cm².",
            f"To 1 decimal place: {answer_text} cm².",
        ]
        distractors = _annulus_distractors(R, r, raw, answer_text)

    elif variant == "mixed_units":
        # Medium: R given as a DIAMETER, r given as a RADIUS, or vice versa
        # (randomised) -- forces noticing which is which before applying the
        # formula. Still one unit (cm) throughout.
        found = False
        for _ in range(_RESAMPLE_ATTEMPTS):
            R = rng.randint(10, 50)
            lo, hi = _band(R)
            if lo > hi:
                continue
            r = rng.randint(lo, hi)
            if r >= R:
                continue
            raw = PI * R**2 - PI * r**2
            if raw > _MAX_RAW_VALUE:
                continue
            if _tenths_tie_distance(raw) <= _tie_margin_for(raw):
                continue
            found = True
            break
        check(found, "annulus_cross_section_area/mixed_units: no valid (R, r) combo found")

        check(r < R, "annulus_cross_section_area/mixed_units: r not strictly less than R")
        check(_in_band(R, r), "annulus_cross_section_area/mixed_units: r outside plausibility band")

        raw = _annulus_raw_and_checks(R, r)
        check(raw >= 0, "annulus area must not be negative")
        rounded = round_half_up(raw, 1)
        answer_text = _fmt1(rounded)

        outer_as_diameter = rng.random() < 0.5
        if outer_as_diameter:
            question_text = (
                f"A hollow cylindrical shaft has an outer diameter of {2 * R} cm "
                f"and an inner radius of {r} cm. Taking π as 3.14159, "
                f"what is the area of its annular cross-section, in square "
                f"centimetres, to 1 decimal place?"
            )
            working = [
                f"Outer radius R = {2 * R} / 2 = {R} cm. Inner radius r = {r} cm (given directly).",
                "Area of an annulus = π × (R² − r²).",
                f"Area = 3.14159 × ({R}² − {r}²) = 3.14159 × {R * R - r * r} = {float(raw):.5f} cm².",
                f"To 1 decimal place: {answer_text} cm².",
            ]
        else:
            question_text = (
                f"A hollow cylindrical shaft has an outer radius of {R} cm and an "
                f"inner diameter of {2 * r} cm. Taking π as 3.14159, what "
                f"is the area of its annular cross-section, in square "
                f"centimetres, to 1 decimal place?"
            )
            working = [
                f"Outer radius R = {R} cm (given directly). Inner radius r = {2 * r} / 2 = {r} cm.",
                "Area of an annulus = π × (R² − r²).",
                f"Area = 3.14159 × ({R}² − {r}²) = 3.14159 × {R * R - r * r} = {float(raw):.5f} cm².",
                f"To 1 decimal place: {answer_text} cm².",
            ]
        distractors = _annulus_distractors(R, r, raw, answer_text)

    elif variant == "thickness_units":
        # Hard: outer DIAMETER given (mm), wall THICKNESS t = R - r given
        # (cm, a DIFFERENT unit) -- the learner derives r = R - t as an
        # unstated intermediate, reconciling units first.
        found = False
        for _ in range(_RESAMPLE_ATTEMPTS):
            R = rng.randint(15, 70)  # outer radius, mm -- see _MAX_RAW_VALUE
            lo, hi = _band(R)
            if lo > hi:
                continue
            r = rng.randint(lo, hi)
            t_mm = R - r
            if t_mm <= 0 or t_mm % 5 != 0:
                continue
            raw = PI * R**2 - PI * r**2
            if raw > _MAX_RAW_VALUE:
                continue
            if _tenths_tie_distance(raw) <= _tie_margin_for(raw):
                continue
            found = True
            break
        check(found, "annulus_cross_section_area/thickness_units: no valid (R, r) combo found")

        check(r < R, "annulus_cross_section_area/thickness_units: r not strictly less than R")
        check(_in_band(R, r), "annulus_cross_section_area/thickness_units: r outside plausibility band")

        raw = _annulus_raw_and_checks(R, r)
        check(raw >= 0, "annulus area must not be negative")
        rounded = round_half_up(raw, 1)
        answer_text = _fmt1(rounded)

        outer_diameter_mm = 2 * R
        t_cm = t_mm / 10  # exact: t_mm is a multiple of 5
        t_display = f"{t_cm:g}"

        question_text = (
            f"A hollow shaft has an outer diameter of {outer_diameter_mm} mm and "
            f"a wall thickness of {t_display} cm. Taking π as 3.14159, "
            f"what is the area of its annular cross-section, in square "
            f"millimetres, to 1 decimal place?"
        )
        working = [
            f"Outer radius R = {outer_diameter_mm} / 2 = {R} mm. Wall thickness t = {t_display} cm = {t_mm} mm.",
            f"Inner radius r = R - t = {R} - {t_mm} = {r} mm.",
            "Area of an annulus = π × (R² − r²).",
            f"Area = 3.14159 × ({R}² − {r}²) = 3.14159 × {R * R - r * r} = {float(raw):.5f} mm².",
            f"To 1 decimal place: {answer_text} mm².",
        ]
        distractors = _annulus_distractors(R, r, raw, answer_text)

    else:
        raise ValueError(f"unknown variant {variant!r}")

    return {
        "text": question_text,
        "latex": question_text,
        "answer": float(rounded),
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": working,
    }


# ============================================================================
# Skill 2: frustum_volume
# ============================================================================


def _frustum_raw_and_checks(R: int, r: int, h) -> Fraction:
    """The frustum volume as a Fraction, via the closed-form polynomial
    `(1/3)*pi*h*(R^2+R*r+r^2)`, independently re-checked via the
    "big cone minus small cone" similar-triangles route -- a distinct
    construction from the polynomial."""
    h = Fraction(h)
    raw = Fraction(1, 3) * PI * h * (R**2 + R * r + r**2)

    total_height = Fraction(h * R, R - r)
    small_height = total_height - h
    vol_big = Fraction(1, 3) * PI * R**2 * total_height
    vol_small = Fraction(1, 3) * PI * r**2 * small_height
    check_equivalent(
        raw,
        vol_big - vol_small,
        "frustum_volume: big-cone-minus-small-cone check disagrees with polynomial construction",
    )
    return raw


def _frustum_distractors(R: int, r: int, h, raw: Fraction, answer_text: str) -> list[str]:
    h = Fraction(h)
    full_cone = Fraction(1, 3) * PI * R**2 * h
    mean_cylinder = PI * h * (Fraction(R + r, 2)) ** 2
    omits_third = PI * h * (R**2 + R * r + r**2)

    check_distinct(raw, full_cone, msg="treats-as-full-cone distractor equals the answer")
    check_distinct(raw, mean_cylinder, msg="uses-mean-radius-as-cylinder distractor equals the answer")
    check_distinct(raw, omits_third, msg="omits-one-third-factor distractor equals the answer")

    check(full_cone >= 0, "frustum volume must not be negative")
    check(mean_cylinder >= 0, "frustum volume must not be negative")
    check(omits_third >= 0, "frustum volume must not be negative")
    candidates = [
        _fmt1(round_half_up(full_cone, 1)),
        _fmt1(round_half_up(mean_cylinder, 1)),
        _fmt1(round_half_up(omits_third, 1)),
    ]
    return pick_distractors(answer_text, candidates, tolerance=ANSWER_TOLERANCE)


@generator(
    "frustum_volume",
    level=12,
    presets={
        "easy": dict(variant="direct"),
        "medium": dict(variant="diameters"),
        "hard": dict(variant="slant_height"),
    },
)
def frustum_volume(rng, variant="direct"):
    """Volume of a frustum of a cone: (1/3) * π * h * (R² + R*r + r²)."""

    if variant == "direct":
        # Easy: R, r, h all given directly, same unit (cm).
        found = False
        for _ in range(_RESAMPLE_ATTEMPTS):
            R = rng.randint(8, 22)  # see _MAX_RAW_VALUE
            lo, hi = _band(R)
            if lo > hi:
                continue
            r = rng.randint(lo, hi)
            if r >= R:
                continue
            h_lo = math.ceil(Fraction(3 * R, 10))
            h_hi = R
            if h_lo > h_hi:
                continue
            h = rng.randint(h_lo, h_hi)
            raw = Fraction(1, 3) * PI * h * (R**2 + R * r + r**2)
            if raw > _MAX_RAW_VALUE:
                continue
            if _tenths_tie_distance(raw) <= _tie_margin_for(raw):
                continue
            found = True
            break
        check(found, "frustum_volume/direct: no valid (R, r, h) combo found")

        check(r < R, "frustum_volume/direct: r not strictly less than R")
        check(h > 0, "frustum_volume/direct: h not strictly positive")
        check(_in_band(R, r), "frustum_volume/direct: r outside plausibility band")
        check(Fraction(h) >= Fraction(3 * R, 10), "frustum_volume/direct: h below the 0.3R plausibility floor")

        raw = _frustum_raw_and_checks(R, r, h)
        check(raw >= 0, "frustum volume must not be negative")
        rounded = round_half_up(raw, 1)
        answer_text = _fmt1(rounded)

        question_text = (
            f"A frustum of a cone has a base radius of {R} cm, a top radius of "
            f"{r} cm and a height of {h} cm. Taking π as 3.14159, what "
            f"is its volume, in cubic centimetres, to 1 decimal place?"
        )
        working = [
            "Volume of a frustum = (1/3) × π × h × (R² + R × r + r²), where R "
            "is the base radius, r is the top radius and h is the height.",
            f"Volume = (1/3) × 3.14159 × {h} × ({R}² + {R} × {r} + {r}²) "
            f"= (1/3) × 3.14159 × {h} × {R * R + R * r + r * r} = {float(raw):.5f} cm³.",
            f"To 1 decimal place: {answer_text} cm³.",
        ]
        distractors = _frustum_distractors(R, r, h, raw, answer_text)

    elif variant == "diameters":
        # Medium: R and r given as DIAMETERS, h given directly -- forces
        # halving both diameters to radii before applying the formula.
        found = False
        for _ in range(_RESAMPLE_ATTEMPTS):
            R = rng.randint(8, 22)  # see _MAX_RAW_VALUE
            lo, hi = _band(R)
            if lo > hi:
                continue
            r = rng.randint(lo, hi)
            if r >= R:
                continue
            h_lo = math.ceil(Fraction(3 * R, 10))
            h_hi = R
            if h_lo > h_hi:
                continue
            h = rng.randint(h_lo, h_hi)
            raw = Fraction(1, 3) * PI * h * (R**2 + R * r + r**2)
            if raw > _MAX_RAW_VALUE:
                continue
            if _tenths_tie_distance(raw) <= _tie_margin_for(raw):
                continue
            found = True
            break
        check(found, "frustum_volume/diameters: no valid (R, r, h) combo found")

        check(r < R, "frustum_volume/diameters: r not strictly less than R")
        check(h > 0, "frustum_volume/diameters: h not strictly positive")
        check(_in_band(R, r), "frustum_volume/diameters: r outside plausibility band")
        check(Fraction(h) >= Fraction(3 * R, 10), "frustum_volume/diameters: h below the 0.3R plausibility floor")

        raw = _frustum_raw_and_checks(R, r, h)
        check(raw >= 0, "frustum volume must not be negative")
        rounded = round_half_up(raw, 1)
        answer_text = _fmt1(rounded)

        question_text = (
            f"A frustum of a cone has a base diameter of {2 * R} cm, a top "
            f"diameter of {2 * r} cm and a height of {h} cm. Taking π "
            f"as 3.14159, what is its volume, in cubic centimetres, to "
            f"1 decimal place?"
        )
        working = [
            f"Base radius R = {2 * R} / 2 = {R} cm. Top radius r = {2 * r} / 2 = {r} cm.",
            "Volume of a frustum = (1/3) × π × h × (R² + R × r + r²).",
            f"Volume = (1/3) × 3.14159 × {h} × ({R}² + {R} × {r} + {r}²) "
            f"= (1/3) × 3.14159 × {h} × {R * R + R * r + r * r} = {float(raw):.5f} cm³.",
            f"To 1 decimal place: {answer_text} cm³.",
        ]
        distractors = _frustum_distractors(R, r, h, raw, answer_text)

    elif variant == "slant_height":
        # Hard: h is NOT given. The slant height l is given, and h must be
        # derived via Pythagoras: l^2 = h^2 + (R - r)^2. (R - r, h, l) is
        # drawn as a scaled Pythagorean triple so the sqrt always lands on an
        # exact integer.
        # R_CAP keeps this tier's magnitudes comparable to easy/medium's own
        # (8, 22) range -- see _MAX_RAW_VALUE; also raises the loop's hit
        # rate, since scale=3/4 draws of the larger _TRIPLES entries would
        # otherwise burn most of the 1000 attempts on combos the raw-value
        # cap rejects anyway.
        R_CAP = 22
        found = False
        for _ in range(_RESAMPLE_ATTEMPTS):
            leg1, leg2, hyp = rng.choice(_TRIPLES)
            scale = rng.randint(1, 2)
            a, b = leg1 * scale, leg2 * scale
            if rng.random() < 0.5:
                a, b = b, a
            diff, h = a, b  # diff = R - r (never the hypotenuse); h the other leg
            l = hyp * scale

            r_lo = math.ceil(Fraction(diff, 4))  # r >= 0.25*diff, from r >= 0.2R
            r_hi_band = math.floor(Fraction(9 * diff, 1))  # r <= 9*diff, from r <= 0.9R
            r_hi_h = math.floor(Fraction(10 * h, 3) - diff)  # r <= 10h/3 - diff, from h >= 0.3R
            r_hi_cap = R_CAP - diff  # r <= R_CAP - diff, from R <= R_CAP
            r_hi = min(r_hi_band, r_hi_h, r_hi_cap)
            if r_lo < 1:
                r_lo = 1
            if r_lo > r_hi:
                continue
            r = rng.randint(r_lo, r_hi)
            R = r + diff
            if not _in_band(R, r):
                continue
            if not (Fraction(h) >= Fraction(3 * R, 10)):
                continue

            raw = Fraction(1, 3) * PI * h * (R**2 + R * r + r**2)
            if raw > _MAX_RAW_VALUE:
                continue
            if _tenths_tie_distance(raw) <= _tie_margin_for(raw):
                continue
            found = True
            break
        check(found, "frustum_volume/slant_height: no valid (R, r, h, l) combo found")

        check(r < R, "frustum_volume/slant_height: r not strictly less than R")
        check(h > 0, "frustum_volume/slant_height: h not strictly positive")
        check(_in_band(R, r), "frustum_volume/slant_height: r outside plausibility band")
        check(Fraction(h) >= Fraction(3 * R, 10), "frustum_volume/slant_height: h below the 0.3R plausibility floor")
        check(l**2 == h**2 + diff**2, "frustum_volume/slant_height: triple does not satisfy Pythagoras")

        # Independent re-derivation of h from l and (R - r) via sympy, a
        # different route from the triple-drawing above.
        h_check = sympy_sqrt(l**2 - (R - r) ** 2)
        check_equivalent(h_check, h, "frustum_volume/slant_height: sympy sqrt(l^2-(R-r)^2) disagrees with drawn h")

        raw = _frustum_raw_and_checks(R, r, h)
        check(raw >= 0, "frustum volume must not be negative")
        rounded = round_half_up(raw, 1)
        answer_text = _fmt1(rounded)

        question_text = (
            f"A frustum of a cone has a base radius of {R} cm, a top radius of "
            f"{r} cm and a slant height of {l} cm. Taking π as 3.14159, "
            f"what is its volume, in cubic centimetres, to 1 decimal place?"
        )
        working = [
            f"Height h: l² = h² + (R − r)², so h = √({l}² − ({R} − {r})²) = √({l * l} − {diff * diff}) = {h} cm.",
            "Volume of a frustum = (1/3) × π × h × (R² + R × r + r²).",
            f"Volume = (1/3) × 3.14159 × {h} × ({R}² + {R} × {r} + {r}²) "
            f"= (1/3) × 3.14159 × {h} × {R * R + R * r + r * r} = {float(raw):.5f} cm³.",
            f"To 1 decimal place: {answer_text} cm³.",
        ]
        distractors = _frustum_distractors(R, r, h, raw, answer_text)

    else:
        raise ValueError(f"unknown variant {variant!r}")

    return {
        "text": question_text,
        "latex": question_text,
        "answer": float(rounded),
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": distractors,
        "working": working,
    }
