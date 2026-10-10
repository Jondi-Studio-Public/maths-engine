"""A NEW, multi-part skill: ONE rectangular prism, its three dimensions given
in MIXED units (length in m, width in cm, height in mm -- fixed roles, so
the independent oracle in this module's test file can parse a fixed
position for each), three independently-graded answers: the capacity in
cubic metres (m3), cubic centimetres (cm3) and litres (L). The point of the
mixed units is that the learner must convert every dimension to a common
unit before multiplying -- that conversion IS the reasoning content of this
skill, not incidental.

Uses `mathgen.core.compose` in its MULTI-PART shape: the prism is drawn ONCE, before `compose` runs, and
threaded into all three parts via `functools.partial` keyword arguments. Each part computes its own capacity
independently from the shared raw dimensions (never from another part's own value) and `final` re-verifies all
three a SECOND, independent way -- via the mm-pivot route, never the cm-pivot route construction used --
straight from the raw dimensions, never from a part's own already-returned value.

## The `>= 1 m3` magnitude constraint (`_PAIR_TOLERANCE = 0.051`, FIXED,
absolute -- mathgen/grading/typed.py:421)

Dimensions are drawn so `volume_m3 >= 1` for EVERY seed at EVERY tier, with
a comfortable margin, not a near-miss: length is drawn in whole METRES
(2-7), width in whole CENTIMETRES that are always a multiple of 10
(150-350, i.e. 1.5-3.5 m), height in whole MILLIMETRES that are always a
multiple of 100 (1000-2500, i.e. 1.0-2.5 m). That specific multiple-of-10/
multiple-of-100 choice is what keeps `volume_cm3` an exact multiple of
10,000 by construction (never by rounding), which in turn makes `volume_m3`
exact to AT MOST 2 decimal places and `volume_L` always a whole number --
eliminating the rounding-boundary hazard entirely (there is never a rounding
step on the canonical answer: `exact_decimal_string` displays the exact
value, it does not round). The measured floor (worst-case draw: length=2,
width=150cm/w=15, height=1000mm/h=10) is `volume_m3 = 2*15*10/100 = 3.0`,
comfortably clear of the >= 1 requirement -- proved as an unconditional test
over every seed/tier in the test file, not a spot check.

## Conversion factors (each literal, each tested)

cm3 -> m3 is / 1,000,000. cm3 -> L is / 1,000 (1 L = 1000 cm3 exactly). Construction pivots
through cm (`_LINEAR_TO_CM`); `_final`'s independent re-check pivots through
mm (`_LINEAR_TO_MM`, 1 cm3 = 1000 mm3), a DIFFERENT arithmetic route.

## Formatting hazard: no space-grouped thousands

`volume_cm3` runs into the millions (10,000 * length_m * width_tens_cm *
height_hundreds_mm, with the largest-magnitude seed reaching ~61,250,000).
Every value in `answer_text` -- m3, cm3, L -- is rendered as PLAIN digits, no
separators at all (never space-grouped, never even comma-grouped, so there
is no possible ambiguity for `mathgen.grading`'s labelled-pair parser). Confirmed
directly against the real `mathgen.grading.grade_pair` on the largest-magnitude
seed in the test file, not by inspection alone.

## Misconceptions (each distractor's gap from the true value, every part,
every seed/tier)

Because `volume_cm3` is always in the millions, every misconception below --
each a wrong CONVERSION FACTOR or a wrong CONSTRUCTION, never a rounding
slip -- differs from its true value by thousands at minimum, astronomically
clear of the `> 0.5` bar (itself a comfortable multiple of the
0.051 tolerance floor):

1. `m3` via / 1,000 instead of / 1,000,000 (`_wrong_m3_divisor`) -- lands
   exactly on the true `L` value numerically (a real, plausible mix-up), but
   under the `m3` label it is always wrong by thousands.
2. `cm3` via the raw, UNCONVERTED mixed-unit numbers multiplied directly
   (`_unconverted_raw_product`) -- the length's bare metre-number, the
   width's bare centimetre-number and the height's bare millimetre-number
   multiplied together with no unit reconciliation at all, offered under the
   `cm3` label. Always `volume_cm3 / 10` by this module's own fixed
   width/height construction (never assumed -- re-derived and asserted in
   the test file too).
3. `L` mistaken for `cm3` numerically -- the `cm3` figure re-offered under
   the `L` label (`_confuses_L_and_cm3`). Always wrong by ~0.999 *
   `volume_cm3` (the L value is always 1000x smaller), the safest of the
   three misconceptions by construction, since the
   real risk sits with the `m3` misconception, not this one.

## Genuine tier escalation (masking-safe): template wording, not units/digits

`scripts/measure_tiers.py`/`mathgen.text_quality.prompt_shape` masks every digit run (including its leading
sign) and every `mathgen.slots` `UNITS` bank word -- `m`, `cm`, `mm`, `km` all share one masked placeholder
(`mathgen/slots.py:61`), so "two dimensions share a unit vs three different units" is invisible to the
instrument regardless of how real the difficulty difference feels to a human. This module therefore draws the
SAME parameter family (dimension ranges, roles, unit choices) at every tier and escalates purely through each
tier's own added TEMPLATE WORDING (`_TEMPLATE_BANKS` below): easy is a single plain sentence; medium adds an
explicit "give each answer as an exact decimal" clause; hard adds a differently-named container AND an
explicit "convert every dimension to the same unit before multiplying" worked-method instruction; exam matches
the exam-item shape with "Exam-style question" framing and a "show full working" clause.
"""

from __future__ import annotations

import functools
from fractions import Fraction

from ..core import (
    GRADER_TOLERANCE,
    check,
    check_equivalent,
    compose,
    exact_decimal_string,
    generator,
    latex_answer,
    numeric_answer,
    phrase,
    pick_distractors,
)

_TIERS = ("easy", "medium", "hard", "exam")

# Length in whole metres, width in whole centimetres (always a multiple of
# 10), height in whole millimetres (always a multiple of 100) -- see the
# module docstring for why this specific multiple-of-10/multiple-of-100
# choice is what keeps volume_cm3 an exact multiple of 10,000 by
# construction, eliminating any rounding-boundary hazard.
_LENGTH_M_RANGE = (2, 7)
_WIDTH_TENS_CM_RANGE = (15, 35)  # width_cm = 10 * this, i.e. 150-350 cm
_HEIGHT_HUNDREDS_MM_RANGE = (10, 25)  # height_mm = 100 * this, i.e. 1000-2500 mm

_LINEAR_TO_CM = {"m": Fraction(100), "cm": Fraction(1), "mm": Fraction(1, 10)}
_LINEAR_TO_MM = {"m": Fraction(1000), "cm": Fraction(10), "mm": Fraction(1)}


def _volume_cm3(dims) -> Fraction:
    """Construction's own pivot: convert every (value, unit) dimension to
    cm, then multiply."""
    total = Fraction(1)
    for value, unit in dims:
        total *= Fraction(value) * _LINEAR_TO_CM[unit]
    return total


def _volume_mm3_independent(dims) -> Fraction:
    """`_final`'s independent re-check pivot: mm, a DIFFERENT base unit from
    construction, and a fresh lookup of each raw factor -- never reuses
    construction's cm-equivalent values."""
    total = Fraction(1)
    for value, unit in dims:
        total *= Fraction(value) * _LINEAR_TO_MM[unit]
    return total


def _draw_dims(rng):
    """`(length_m, width_cm, height_mm)`, each a plain int. See the module
    docstring's magnitude-constraint section for the ranges and why they
    guarantee `volume_m3 >= 1` (measured floor 3.0) and an exact-cm3-
    multiple-of-10,000 construction."""
    length_m = rng.randint(*_LENGTH_M_RANGE)
    width_cm = 10 * rng.randint(*_WIDTH_TENS_CM_RANGE)
    height_mm = 100 * rng.randint(*_HEIGHT_HUNDREDS_MM_RANGE)
    return length_m, width_cm, height_mm


_EASY_TEMPLATES = (
    "A rectangular tank measures {l} m by {w} cm by {h} mm. Find its "
    "capacity in cubic metres, cubic centimetres and litres.",
    "A rectangular container is {l} m long, {w} cm wide and {h} mm high. "
    "Find its capacity in cubic metres, cubic centimetres and litres.",
    "A rectangular tank is {l} m long, {w} cm wide and {h} mm high. Work "
    "out its capacity in cubic metres, cubic centimetres and litres.",
)
_MEDIUM_TEMPLATES = (
    "A rectangular tank measures {l} m by {w} cm by {h} mm. Find its "
    "capacity in cubic metres, cubic centimetres and litres. Give each "
    "answer as an exact decimal, using as many decimal places as needed.",
    "A rectangular storage bin is {l} m long, {w} cm wide and {h} mm high. "
    "Find its capacity in cubic metres, cubic centimetres and litres, "
    "giving each answer as an exact decimal, using as many decimal places "
    "as needed.",
    "A rectangular container measures {l} m by {w} cm by {h} mm. Calculate "
    "its capacity in cubic metres, cubic centimetres and litres, with each "
    "answer given as an exact decimal (as many decimal places as needed).",
)
_HARD_TEMPLATES = (
    "An industrial water tank measures {l} m by {w} cm by {h} mm. Convert "
    "every dimension to the same unit before multiplying, then find its "
    "capacity in cubic metres, cubic centimetres and litres.",
    "A rectangular fuel tank is {l} m long, {w} cm wide and {h} mm high. "
    "Before calculating, convert every dimension to the same unit; then "
    "find its capacity in cubic metres, cubic centimetres and litres.",
    "A rectangular holding tank measures {l} m by {w} cm by {h} mm. First "
    "convert every dimension to the same unit, then find its capacity in "
    "cubic metres, cubic centimetres and litres.",
)
_EXAM_TEMPLATES = (
    "Exam-style question. A rectangular tank measures {l} m by {w} cm by "
    "{h} mm. Show your working to find its capacity in cubic metres, "
    "cubic centimetres and litres.",
    "This exam question gives a rectangular container {l} m long, {w} cm "
    "wide and {h} mm high. Show full working to find its capacity in "
    "cubic metres, cubic centimetres and litres.",
    "A rectangular tank for this exam question measures {l} m by {w} cm "
    "by {h} mm. Show your full working as you find its capacity in cubic "
    "metres, cubic centimetres and litres.",
)
_TEMPLATE_BANKS = {
    "easy": _EASY_TEMPLATES,
    "medium": _MEDIUM_TEMPLATES,
    "hard": _HARD_TEMPLATES,
    "exam": _EXAM_TEMPLATES,
}


# the three parts ------------------


def _part_m3(rng, values, *, m3_val, cm3_val):
    m3_text = exact_decimal_string(m3_val, 2)
    working = f"Capacity in m3 = {int(cm3_val)} / 1000000 = {m3_text} m3."
    return m3_val, working


def _part_cm3(rng, values, *, cm3_val, length_m, width_cm, height_mm):
    length_cm = 100 * length_m
    height_cm_text = exact_decimal_string(Fraction(height_mm, 10), 1)
    working = (
        f"Convert every dimension to cm: length = 100 x {length_m} = "
        f"{length_cm} cm, width = {width_cm} cm, height = {height_mm} / 10 "
        f"= {height_cm_text} cm. Volume = {length_cm} x {width_cm} x "
        f"{height_cm_text} = {int(cm3_val)} cm3."
    )
    return cm3_val, working


def _part_L(rng, values, *, L_val, cm3_val):
    working = f"Capacity in L = {int(cm3_val)} / 1000 = {int(L_val)} L."
    return L_val, working


def _final(values, rng, *, length_m, width_cm, height_mm, tier):
    cm3_val, m3_val, L_val = values

    dims = [(length_m, "m"), (width_cm, "cm"), (height_mm, "mm")]

    volume_mm3 = _volume_mm3_independent(dims)
    cm3_check = volume_mm3 / 1000
    check_equivalent(
        cm3_val,
        cm3_check,
        "exd_e5_prism_capacity_three_units: cm3 disagrees with the independent mm-pivot recomputation",
    )
    check_equivalent(
        m3_val,
        volume_mm3 / Fraction(1_000_000_000),
        "exd_e5_prism_capacity_three_units: m3 disagrees with the independent mm-pivot recomputation",
    )
    check_equivalent(
        L_val,
        volume_mm3 / Fraction(1_000_000),
        "exd_e5_prism_capacity_three_units: L disagrees with the independent mm-pivot recomputation",
    )
    check(m3_val >= 1, "exd_e5_prism_capacity_three_units: volume_m3 < 1")

    m3_text = exact_decimal_string(m3_val, 2)
    cm3_text = str(int(cm3_val))
    L_text = str(int(L_val))

    answer_text = f"m3 = {m3_text}, cm3 = {cm3_text}, L = {L_text}"
    answer = {
        "m3": numeric_answer(m3_val),
        "cm3": int(cm3_val),
        "L": int(L_val),
    }

    text = phrase(rng, _TEMPLATE_BANKS[tier], l=length_m, w=width_cm, h=height_mm)

    # Misconceptions -- see the module docstring's numbered list for the
    # construction and the gap each one guarantees. Every candidate is a
    # FULL labelled answer string with exactly one part's value swapped
    # (`answer: pair` compares by label, so a candidate must be a complete,
    # differently-labelled triple, not a single mis-valued number).
    wrong_m3_val = cm3_val / 1000  # the named hazard: /1000 instead of /1e6
    check(
        abs(wrong_m3_val - m3_val) > Fraction(1, 2),
        "exd_e5_prism_capacity_three_units: wrong-divisor m3 distractor gap <= 0.5",
    )
    wrong_m3_text = str(int(wrong_m3_val))

    raw_product = Fraction(length_m) * Fraction(width_cm) * Fraction(height_mm)
    check(
        abs(raw_product - cm3_val) > Fraction(1, 2),
        "exd_e5_prism_capacity_three_units: unconverted-raw-product cm3 distractor gap <= 0.5",
    )
    raw_text = str(int(raw_product))

    check(
        abs(cm3_val - L_val) > Fraction(1, 2),
        "exd_e5_prism_capacity_three_units: confuses-L-and-cm3 distractor gap <= 0.5",
    )

    candidates = [
        f"m3 = {wrong_m3_text}, cm3 = {cm3_text}, L = {L_text}",
        f"m3 = {m3_text}, cm3 = {raw_text}, L = {L_text}",
        f"m3 = {m3_text}, cm3 = {cm3_text}, L = {cm3_text}",
    ]
    for cand in candidates:
        check(cand != answer_text, f"exd_e5_prism_capacity_three_units: a distractor matches the answer text: {cand!r}")

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=pick_distractors(answer_text, candidates, tolerance=GRADER_TOLERANCE),
    )


def _build_prism_capacity(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    length_m, width_cm, height_mm = _draw_dims(rng)
    dims = [(length_m, "m"), (width_cm, "cm"), (height_mm, "mm")]
    cm3_val = _volume_cm3(dims)
    m3_val = cm3_val / Fraction(1_000_000)
    L_val = cm3_val / Fraction(1_000)

    part_m3 = functools.partial(_part_m3, m3_val=m3_val, cm3_val=cm3_val)
    part_cm3 = functools.partial(
        _part_cm3,
        cm3_val=cm3_val,
        length_m=length_m,
        width_cm=width_cm,
        height_mm=height_mm,
    )
    part_L = functools.partial(_part_L, L_val=L_val, cm3_val=cm3_val)
    final = functools.partial(
        _final,
        length_m=length_m,
        width_cm=width_cm,
        height_mm=height_mm,
        tier=tier,
    )

    # cm3 first, then m3, then L: matches the learner-facing method (convert
    # every dimension to a common unit and multiply to get cm3, THEN convert
    # that to m3/L) -- so `working[0]` is never a forward reference to a
    # value `working[1]` has not derived yet.
    return compose(rng, [part_cm3, part_m3, part_L], final=final)


@generator(
    "exd_e5_prism_capacity_three_units",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def exd_e5_prism_capacity_three_units(rng, tier: str = "easy"):
    return _build_prism_capacity(rng, tier=tier)
