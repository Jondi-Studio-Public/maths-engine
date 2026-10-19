"""Generates `exd_e5_circle_area_to_diameter_and_circumference` questions: from a circle's area, find its
diameter and circumference.

## The shape

Given a circle's AREA (a rounded, DISPLAYED value), find the diameter and
the circumference -- BACKWARDS through a square root
(`r = sqrt(area / pi)`). `answer: pair`.

## Construction: draw the radius FIRST, never the area

brief: the radius `r` is drawn directly, from a small "nice" set of
integers (`_R_LO=2` .. `_R_HI=20`) -- never derived by inverting a
already-rounded area. Every canonical quantity is computed directly from
this internally-known EXACT `r`, using TRUE pi (`Fraction(math.pi)`, the
exact rational value of the double-precision float -- accurate to ~1e-16
relative error, orders of magnitude finer than anything this module ever
rounds to; NOT a truncated decimal literal like `3.14159` or a coarse
rational like `22/7`, which is exactly the T5-annulus/frustum failure mode
carried as an exact `Fraction` through every intermediate step and rounded
ONLY ONCE, at the point each quantity is displayed:

- `area_true = PI * r**2` (exact Fraction) -- rounded to 2 decimal places
  ONLY for display in the question prompt (`area_display`). Never used
  again internally after that: the diameter and circumference below are
  computed from `r` directly, never by re-deriving from `area_display`.
- `diameter_true = 2 * r` (exact integer -- no pi involved at all) --
  rounded to 1 decimal place for display (a no-op on an already-exact
  integer: `round_half_up(Fraction(2 * r), 1)` always lands back on
  `2 * r` exactly, so there is nothing for this rounding step to lose).
- `circumference_true = 2 * PI * r` (exact Fraction) -- rounded to 1
  decimal place for display.

## The round-trip proof -- this is what the independent oracle proves, not
what this generator computes

A learner (and the independent test-file oracle) can only see the
ROUNDED, DISPLAYED area in the prompt text, so they compute
`r' = sqrt(area_display / PI)` (full precision, no intermediate rounding),
then `d' = 2 * r'` and `C' = PI * d'` (also full precision), and round
THOSE to the stated 1 decimal place ONLY at the very end -- never
re-rounding an already-rounded intermediate, the same "round once, at the
end" discipline this generator itself follows.

**Chosen combination: radius domain = integers 2..20; area stated to 2 decimal places; diameter/circumference
stated to 1 decimal place.** Proved clean for EVERY one of the 19 possible radii (not a seed sample -- the
domain is small and fully enumerable) by direct computation in this module's own test file
(`test_round_trip_recovers_canonical_values_for_every_possible_radius`): rounding `area_true` to 2dp and then
recomputing `r'`/`d'`/`C'` from that rounded value alone reproduces `diameter_true` and `circumference_true`
at their own 1dp precision, exactly, for all 19 radii, zero mismatches.

## The 0.051 hazard

`_PAIR_TOLERANCE = 0.051` (`mathgen/grading/typed.py:421`) is a FIXED absolute tolerance for every `answer:
pair` label. Radius `>= 2` keeps `diameter >= 4` and `circumference >= 12.5`, both orders of magnitude clear
of `0.051`. Every misconception distractor's gap from the true value is checked, unconditionally, against a `>
0.5` floor -- every one of this module's three misconceptions clears that floor by a wide margin (the
smallest, "reports the radius as the diameter", has a minimum gap of `r >= 2`), proved directly rather than
assumed.

## Genuine tier escalation -- template wording only

`scripts/measure_tiers.py`/`prompt_shape` masks every digit run (including
sign), so a tier lever built from the radius range or the stated
precision's digit count is invisible to the instrument. Every tier draws
`r` from the SAME range and escalates entirely through explicit added
TEMPLATE WORDING:

- **easy**: a plain single-sentence template, no rounding instruction
  stated at all.
- **medium**: adds an explicit clause -- "...each correct to 1 decimal
  place." -- the `requested_rounding`/`_DECIMAL_PLACES_RE` trigger.
- **hard**: a further explicit clause instructing the learner to show the
  intermediate radius as a working step -- which also strengthens the
  independent-oracle round-trip proof, since the intermediate radius is now
  checkable against the rendered `working` too.
- **exam**: has the shape of a written exam question
  -- lettered (a)/(b) parts and "showing your working" -- its own distinct
  clause.

Every part is always rounded/displayed the same way (2dp area, 1dp
diameter/circumference) regardless of tier wording -- only the SENTENCE
differs, matching `rectangle-area-perimeter-three-units`'s own precedent
for this masking-safe tier design.

## Misconceptions

1. `reports-radius-as-diameter` -- reports the radius (not doubled) as the
   diameter answer. Always exercisable; gap from the true diameter is
   exactly `r`, `>= 2` at this skill's radius floor.
2. `circumference-direct-from-area` -- computes circumference directly
   from the DISPLAYED area with the wrong formula (`C = 2 * pi * area`)
   instead of going through the radius/diameter at all. Always
   exercisable and never coincidentally close -- this misconception is
   dimensionally wrong (multiplying an area-scale quantity by `2*pi`
   rather than a length-scale one), so its magnitude is always far larger
   than the true circumference at this skill's radius range.
3. `diameter-formula-on-radius` -- uses the diameter-based circumference
   formula (`C = pi * r`, i.e. treats the radius as if it were the
   diameter) instead of `C = pi * d = 2 * pi * r`. Always exercisable; this
   is exactly HALF the true circumference, so the gap is
   `circumference_true / 2 >= pi >= 3.14` at this skill's radius floor.
"""

from __future__ import annotations

import functools
import math
from fractions import Fraction

from ..core import (
    GRADER_TOLERANCE,
    check,
    check_equivalent,
    compose,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
    round_half_up,
    stated_precision_string,
)

_TIERS = ("easy", "medium", "hard", "exam")

# True pi, carried as the exact Fraction value of the double-precision
# float: true pi, never a rounded value. NOT a
# truncated decimal literal or a coarse rational approximation.
PI = Fraction(math.pi)

# "Nice" integer radii, comfortably clear of the 0.051 pair tolerance
# (diameter >= 4, circumference >= 12.5) -- see the module docstring's
# round-trip proof for why this range, combined with 2dp area / 1dp
# diameter+circumference, is proven clean for every possible draw.
_R_LO = 2
_R_HI = 20


def _boundary_margin(value: Fraction, places: int) -> Fraction:
    """Distance from `value` to the nearest `places`-decimal-place rounding
    boundary (an `x.x5`-style midpoint). Used only as a defensive
    assertion below -- with an irrational true-pi factor in `area_true`/
    `circumference_true` and a small, fully-enumerable integer radius
    domain, an exact tie is not mathematically possible here, and this
    module's test file confirms that by direct enumeration over all 19
    radii; this `check()` exists so a future change to the radius domain
    that DID introduce a tie would fail loudly rather than silently."""
    scale = 10**places
    scaled = value * scale
    frac = scaled - (scaled.numerator // scaled.denominator)
    return abs(frac - Fraction(1, 2))


_NOT_ON_BOUNDARY = Fraction(1, 10**6)

_EASY_TEMPLATES = (
    "A circle has an area of {A} cm². Find its diameter and its circumference. Give each answer to 1 decimal place.",
    "A circular plate has an area of {A} cm². Work out its diameter and its circumference. Round each answer to 1 decimal place.",
    "The area of a circle is {A} cm². Calculate the diameter and the circumference. Write each to 1 decimal place.",
)
_MEDIUM_TEMPLATES = (
    "A circle has an area of {A} cm². Find its diameter and its circumference, each correct to 1 decimal place.",
    "A circular plate has an area of {A} cm². Work out its diameter and its "
    "circumference, giving each answer correct to 1 decimal place.",
    "The area of a circle is {A} cm². Calculate the diameter and the circumference, each rounded to 1 decimal place.",
)
_HARD_TEMPLATES = (
    "A circle has an area of {A} cm². Find its diameter and its circumference, each "
    "correct to 1 decimal place. Show the radius you used as a working step.",
    "A circular plate has an area of {A} cm². Work out its diameter and its "
    "circumference, each correct to 1 decimal place, showing the radius you "
    "calculate along the way.",
    "The area of a circle is {A} cm². Calculate the diameter and the circumference, "
    "each rounded to 1 decimal place, and show the radius you find as part of your "
    "working.",
)
_EXAM_TEMPLATES = (
    "A circle has an area of {A} cm². Calculate (a) the "
    "diameter and (b) the circumference of the circle, each correct to 1 decimal "
    "place, showing your working.",
    "This exam question gives a circle of area {A} cm². Find, with full working, "
    "(a) the diameter and (b) the circumference, each correct to 1 decimal place.",
    "Exam question: a circle has an area of {A} cm². Show your working to find "
    "(a) the diameter and (b) the circumference, each correct to 1 decimal place.",
)
_TEMPLATE_BANKS = {
    "easy": _EASY_TEMPLATES,
    "medium": _MEDIUM_TEMPLATES,
    "hard": _HARD_TEMPLATES,
    "exam": _EXAM_TEMPLATES,
}


def _draw_radius(rng) -> int:
    """Draw the radius directly -- never the area. Defensive boundary
    checks (see `_boundary_margin`'s own docstring for why these are not
    expected to ever fire for this integer domain)."""
    for _ in range(200):
        r = rng.randint(_R_LO, _R_HI)
        area_true = PI * r * r
        circumference_true = 2 * PI * r
        if _boundary_margin(area_true, 2) <= _NOT_ON_BOUNDARY:
            continue
        if _boundary_margin(circumference_true, 1) <= _NOT_ON_BOUNDARY:
            continue
        return r
    check(False, "circle_area_to_diameter_and_circumference: could not draw a non-boundary radius in 200 attempts")


# --- the two parts (module-level, so the test file can monkeypatch one by
# name -- the established pattern for a compose() part that needs shared
# per-question state while staying patchable) --------------------------


def _part_diameter(rng, values, *, r, diameter_rounded):
    diameter_text = stated_precision_string(diameter_rounded, 1)
    working = f"Diameter = 2 × radius = 2 × {r} = {diameter_text} cm."
    return diameter_rounded, working


def _part_circumference(rng, values, *, r, circumference_rounded):
    circumference_text = stated_precision_string(circumference_rounded, 1)
    working = (
        f"Circumference = 2 × π × radius = 2 × π × {r} = "
        f"{circumference_text} cm (using true π, rounded once at the end)."
    )
    return circumference_rounded, working


def _final(values, rng, *, r, area_true, area_display, tier):
    diameter_v, circumference_v = values

    # Independent in-generator re-verification: recompute both parts a SECOND way, straight from `r`
    # -- never from a part's own already-returned value, and never by re-deriving from the rounded
    # displayed area.
    true_diameter = round_half_up(Fraction(2 * r), 1)
    check_equivalent(
        diameter_v,
        true_diameter,
        "circle_area_to_diameter_and_circumference: diameter part disagrees with independent recomputation from r",
    )
    true_circumference = round_half_up(2 * PI * r, 1)
    check_equivalent(
        circumference_v,
        true_circumference,
        "circle_area_to_diameter_and_circumference: circumference part disagrees with independent recomputation from r",
    )
    # The magnitude constraint: both parts comfortably clear the fixed
    # 0.051 pair tolerance at this skill's radius floor.
    check(
        diameter_v >= 4,
        "circle_area_to_diameter_and_circumference: diameter below the >= 4 floor",
    )
    check(
        circumference_v >= 4 * PI,
        "circle_area_to_diameter_and_circumference: circumference below its floor",
    )

    diameter_text = stated_precision_string(diameter_v, 1)
    circumference_text = stated_precision_string(circumference_v, 1)
    area_text = stated_precision_string(area_display, 2)

    answer_text = f"diameter = {diameter_text}, circumference = {circumference_text}"
    answer = {
        "diameter": numeric_answer(diameter_v),
        "circumference": numeric_answer(circumference_v),
    }

    text = phrase(rng, _TEMPLATE_BANKS[tier], A=area_text)

    # Misconceptions, one per part, held as a FULL labelled candidate
    # string (the other part correct) -- `answer: pair` compares by LABEL,
    # so a candidate must be a complete, differently-labelled pair.
    candidates = []

    # 1. reports the radius (not doubled) as the diameter.
    wrong_diameter = round_half_up(Fraction(r), 1)
    gap_diameter = abs(true_diameter - wrong_diameter)
    check(
        gap_diameter > Fraction(1, 2),
        "circle_area_to_diameter_and_circumference: reports-radius-as-diameter gap too small",
    )
    if wrong_diameter != diameter_v:
        candidates.append(
            f"diameter = {stated_precision_string(wrong_diameter, 1)}, circumference = {circumference_text}"
        )

    # 2. computes circumference directly from the DISPLAYED area with the
    #    wrong formula (C = 2 * pi * area) instead of going through r/d.
    wrong_circumference_from_area = round_half_up(2 * PI * area_display, 1)
    gap_area_formula = abs(true_circumference - wrong_circumference_from_area)
    check(
        gap_area_formula > Fraction(1, 2),
        "circle_area_to_diameter_and_circumference: circumference-direct-from-area gap too small",
    )
    if wrong_circumference_from_area != circumference_v:
        candidates.append(
            f"diameter = {diameter_text}, circumference = {stated_precision_string(wrong_circumference_from_area, 1)}"
        )

    # 3. uses the diameter-based formula on the radius (C = pi * r instead
    #    of C = pi * d) -- exactly half the true circumference.
    wrong_circumference_half = round_half_up(PI * r, 1)
    gap_half_formula = abs(true_circumference - wrong_circumference_half)
    check(
        gap_half_formula > Fraction(1, 2),
        "circle_area_to_diameter_and_circumference: diameter-formula-on-radius gap too small",
    )
    if wrong_circumference_half != circumference_v:
        candidates.append(
            f"diameter = {diameter_text}, circumference = {stated_precision_string(wrong_circumference_half, 1)}"
        )

    for cand in candidates:
        check(
            cand != answer_text,
            f"circle_area_to_diameter_and_circumference: a distractor matches the answer text: {cand!r}",
        )

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=pick_distractors(answer_text, candidates, tolerance=GRADER_TOLERANCE),
    )


def _build_circle_area_to_diameter_and_circumference(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    r = _draw_radius(rng)
    area_true = PI * r * r
    area_display = round_half_up(area_true, 2)
    diameter_rounded = round_half_up(Fraction(2 * r), 1)
    circumference_rounded = round_half_up(2 * PI * r, 1)

    part_diameter = functools.partial(
        _part_diameter,
        r=r,
        diameter_rounded=diameter_rounded,
    )
    part_circumference = functools.partial(
        _part_circumference,
        r=r,
        circumference_rounded=circumference_rounded,
    )
    final = functools.partial(
        _final,
        r=r,
        area_true=area_true,
        area_display=area_display,
        tier=tier,
    )

    return compose(rng, [part_diameter, part_circumference], final=final)


@generator(
    "exd_e5_circle_area_to_diameter_and_circumference",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def exd_e5_circle_area_to_diameter_and_circumference(rng, tier: str = "easy"):
    return _build_circle_area_to_diameter_and_circumference(rng, tier=tier)
