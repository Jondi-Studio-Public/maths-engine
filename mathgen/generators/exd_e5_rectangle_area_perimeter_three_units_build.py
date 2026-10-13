"""Generates `exd_e5_rectangle_area_perimeter_three_units` questions: one rectangle's perimeter in mm and its
area in cm^2 and m^2.

## The shape, pinned

ONE rectangle whose two side lengths are drawn in mm, THREE independently
labelled parts, `answer: pair`:

- `perimeter_mm` -- the perimeter, in mm, native unit, exact integer
  arithmetic (`2 * (L + W)`), no conversion at all -- deliberately the
  "easy" part of this skill's three answers, contrasted against the two
  area conversions below.
- `area_cm2` -- the area, mm^2 -> cm^2 (divide by 100).
- `area_m2` -- the area, mm^2 -> m^2 (divide by 1,000,000, equivalently
  divide `area_cm2` by 10,000).

## The magnitude constraint

`area_m2` is the dangerous part: an mm-scale rectangle's area in m^2 is
naturally tiny (a 250mm x 180mm plate is 0.045 m^2 -- inside the tolerance
danger zone). Side lengths are therefore drawn from `_SIDE_LO=1200` to
`_SIDE_HI=9000` mm -- genuinely metre-scale panel/room dimensions given in
mm for a technical drawing, NOT a shrunk "friendly" prose figure -- so the
RAW (pre-rounding) `area_m2 = L * W / 1_000_000` is `>= 1.44` for every
possible draw (minimum at `L = W = 1200`: `1200 * 1200 / 1_000_000 = 1.44`),
comfortably clear of both `0.051` and the `>= 1` floor, proven
unconditionally in `tests/test_exd_e5_rectangle_area_perimeter_three_units_
build.py` over every seed/tier tested, not by construction alone.

## The formatting hazard (space-grouped thousands)

`perimeter_mm` can run into the tens of thousands at this side-length range
(max `2 * (9000 + 9000) = 36000`). Every numeric token in `answer_text`
(`perimeter_mm`, `area_cm2`, `area_m2`) is rendered as PLAIN digits, via
`mathgen.core.stated_precision_string`/a bare `int`/`str` -- never
space-grouped, which `mathgen.grading`'s labelled-pair parser
(`_split_labelled_segments`) would misread as several separate values (the
same hazard `prism-capacity-three-units` notes). Confirmed by a
direct `mathgen.grading.grade_pair` test on this module's own largest-magnitude
seed.

## Conversion factors (each literal, each tested)

mm^2 -> cm^2 is DIVIDE BY 100.
mm^2 -> m^2 is DIVIDE BY 1,000,000. mm -> mm (perimeter) needs no
conversion at all.

## Precision: "nearest whole number", every tier, both areas

Both `area_cm2` and `area_m2` are rounded to the nearest whole number
(`mathgen.core.round_half_up(..., 0)`) and displayed with
`stated_precision_string(..., 0)` (a bare integer, no decimal point) --
`perimeter_mm` is already an exact integer with nothing to round. Medium,
hard and exam tiers state this explicitly ("...to the nearest whole
number..."), which is `mathgen.text_quality.requested_rounding`'s
`_NEAREST_RE` trigger (`"nearest whole number"` -> `places = 0`) --
the stated-precision check then
requires EVERY number token in `answer_text` to show 0 decimal places,
which all three already do by construction (unlike a `pair` skill that mixes precisions,
this one states only ONE precision -- 0dp -- across every part, so there is
no multi-precision conflict to route around). The easy tier does not state
the phrase at all (see tier design below); the check then declines to run
it (`requested_rounding` returns `None`), and the construction is unaffected
either way since every part is always rounded/displayed the same way
regardless of tier wording.

A raw area sitting EXACTLY on the 0-decimal-place rounding tie is rejected and the sides redrawn
(`_draw_clean_sides`'s retry loop, `_is_rounding_tie`), by construction rather than by chance.

## Genuine tier escalation (wording only -- the masking-safe design this
skill uses)

`scripts/measure_tiers.py`/`mathgen.text_quality.prompt_shape` masks every
digit run (including its leading sign) and every `mathgen.slots` bank word,
so a tier that differs only in the drawn side lengths' magnitude/roundness
is invisible to the instrument -- and unlike `mean-median-mode-one-set`, this
skill has no analogous structural knob: the same three labelled parts exist
at every tier. Every tier therefore draws sides from the SAME range
(`_SIDE_LO`..`_SIDE_HI`) and escalation is entirely in the TEMPLATE WORDING
(the design fixed for this skill):

- **easy**: a plain single-sentence template naming only the perimeter and
  the two area conversions in one breath, no rounding instruction stated.
- **medium**: adds an explicit clause -- "Give each area to the nearest
  whole number in its stated unit" (or an equivalent phrasing) -- the
  `_NEAREST_RE` trigger described above.
- **hard**: a further explicit clause, an engineering-context framing
  naming a specific use ("...as they would appear on a materials order").
- **exam**: has the shape of a written exam question
  -- lettered (a)/(b)/(c) parts and "showing your working for each part" --
  its own distinct clause.

Every one of the 12 templates (3 per tier) embeds the literal substring
"`{L} mm by {W} mm`" so this module's own test file's independent oracle can
recover the two drawn side lengths from `question.text` with one robust
regex, regardless of which tier/variant rendered the question.

## Misconceptions (at minimum)

1. `mm2-to-cm2-uses-ten` -- divides the mm^2 area by 10 instead of 100.
2. `perimeter-area-formula-swap` -- computes `perimeter_mm` with the AREA
   formula (`L * W`) instead of the perimeter formula (`2 * (L + W)`).
3. `mm2-to-m2-linear-factor` -- forgets the squared-unit conversion factor
   for `area_m2` entirely: divides the mm^2 area by 1,000 (the LINEAR
   mm->m factor) instead of 1,000,000 (the correct SQUARED factor),
   treating the area conversion as if it were a linear one.

Every candidate is checked, unconditionally, both for exact non-collision
with the true answer (`check`) and for the `> 0.5` magnitude-gap floor the
skill requires (this module's own generator-level
`check`/`check_distinct` calls plus the test file's explicit per-candidate
gap assertions) -- every candidate here differs from the truth by at least
one full order of magnitude, so the floor is cleared with very wide margin,
proven rather than assumed.
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
    round_half_up,
    stated_precision_string,
)

_TIERS = ("easy", "medium", "hard", "exam")

# Genuinely metre-scale panel/room dimensions given in mm -- see the module
# docstring's magnitude-constraint section. Minimum raw area_m2 at
# L=W=_SIDE_LO is 1.44, comfortably clear of both 0.051 and the
# ">= 1" floor.
_SIDE_LO = 1200
_SIDE_HI = 9000

_EASY_TEMPLATES = (
    "A rectangular sheet measures {L} mm by {W} mm. Find its perimeter in mm, its area in cm², and its area in m².",
    "A rectangular panel measures {L} mm by {W} mm. Work out the perimeter in mm and the area in both cm² and m².",
    "A rectangle measures {L} mm by {W} mm. Calculate the perimeter in mm, the area in cm², and the area in m².",
)
_MEDIUM_TEMPLATES = (
    "A rectangular sheet measures {L} mm by {W} mm. Find its perimeter in mm, its area in "
    "cm², and its area in m². Give each area to the nearest whole number in its "
    "stated unit.",
    "A rectangular panel measures {L} mm by {W} mm. Work out the perimeter in mm and the area "
    "in both cm² and m², rounding each area to the nearest whole number.",
    "A rectangle measures {L} mm by {W} mm. Calculate the perimeter in mm, the area in cm², "
    "and the area in m², giving each area correct to the nearest whole number.",
)
_HARD_TEMPLATES = (
    "A technical drawing specifies a rectangular panel {L} mm by {W} mm. Calculate the "
    "perimeter in mm and the area in cm² and m², each to the nearest whole number. "
    "State your answers as they would appear on a materials order.",
    "An engineering drawing shows a rectangular sheet {L} mm by {W} mm. Determine the perimeter "
    "in mm and the area in cm² and m², each area to the nearest whole number, ready to "
    "record on a materials order.",
    "A workshop drawing gives a rectangular panel as {L} mm by {W} mm. Work out the perimeter "
    "in mm and the area in cm² and m², each rounded to the nearest whole number, in "
    "the form used on a materials order.",
)
_EXAM_TEMPLATES = (
    "A rectangular sheet metal panel measures {L} mm by {W} mm. Determine "
    "(a) the perimeter in mm, (b) the area in cm², and (c) the area in m², each area "
    "to the nearest whole number, showing your working for each part.",
    "This exam question gives a rectangular panel {L} mm by {W} mm. Find, with full working, "
    "(a) the perimeter in mm, (b) the area in cm², and (c) the area in m², each area "
    "to the nearest whole number.",
    "Exam question: a rectangular sheet is {L} mm by {W} mm. Show your working to find the "
    "perimeter in mm and the area in cm² and m², each area rounded to the nearest "
    "whole number.",
)
_TEMPLATE_BANKS = {
    "easy": _EASY_TEMPLATES,
    "medium": _MEDIUM_TEMPLATES,
    "hard": _HARD_TEMPLATES,
    "exam": _EXAM_TEMPLATES,
}


def _is_rounding_tie(value: Fraction) -> bool:
    """Whether `value` sits exactly on the nearest-whole-number rounding
    boundary (an `x.5`), worked for `dp=0`."""
    scaled = value * 2
    return scaled.denominator == 1 and scaled.numerator % 2 == 1


def _draw_clean_sides(rng):
    """Draw `(L, W)` mm side lengths whose area sits cleanly off the
    nearest-whole-number rounding tie for BOTH `area_cm2` and `area_m2` --
    by construction, retried rather than assumed. 200 attempts is generous:
    an exact `x.5` tie is a rare rational coincidence against this range."""
    for _ in range(200):
        L = rng.randint(_SIDE_LO, _SIDE_HI)
        W = rng.randint(_SIDE_LO, _SIDE_HI)
        area_mm2 = L * W
        if _is_rounding_tie(Fraction(area_mm2, 100)):
            continue
        if _is_rounding_tie(Fraction(area_mm2, 1_000_000)):
            continue
        return L, W
    check(False, "rectangle_area_perimeter_three_units: could not draw clean sides in 200 attempts")


# the three parts ------------------


def _part_perimeter(rng, values, *, L, W, perimeter_mm):
    working = f"Perimeter: 2 × (length + width) = 2 × ({L} + {W}) mm = {perimeter_mm} mm."
    return perimeter_mm, working


def _part_area_cm2(rng, values, *, L, W, area_mm2, cm2_rounded):
    cm2_text = stated_precision_string(cm2_rounded, 0)
    # Fraction(area_mm2, 100) always terminates within 2 decimal places
    # (its reduced denominator divides 100 = 2^2 x 5^2), so
    # `exact_decimal_string` never raises here -- displayed as a real
    # decimal, never a raw p/q Fraction repr, which a learner dividing by
    # 100 on a calculator would never produce.
    exact_cm2_text = exact_decimal_string(Fraction(area_mm2, 100), 2)
    working = (
        f"Area: {L} × {W} = {area_mm2} mm². Convert mm² to cm² by dividing "
        f"by 100 (mm² to cm² is divide by 100, not 10): {area_mm2} / 100 = "
        f"{exact_cm2_text} which rounds to {cm2_text} cm² (nearest whole number)."
    )
    return cm2_rounded, working


def _part_area_m2(rng, values, *, L, W, area_mm2, m2_rounded):
    m2_text = stated_precision_string(m2_rounded, 0)
    # Fraction(area_mm2, 1_000_000) always terminates within 6 decimal
    # places (its reduced denominator divides 10^6 = 2^6 x 5^6) -- see
    # _part_area_cm2's own comment for why a real decimal, not a raw p/q
    # Fraction repr, is required here.
    exact_m2_text = exact_decimal_string(Fraction(area_mm2, 1_000_000), 6)
    working = (
        f"Area: {L} × {W} = {area_mm2} mm². Convert mm² to m² by dividing by "
        f"1000000 (the SQUARED factor, 1000 × 1000, not the linear mm-to-m factor of 1000): "
        f"{area_mm2} / 1000000 = {exact_m2_text} which rounds to {m2_text} "
        f"m² (nearest whole number)."
    )
    return m2_rounded, working


def _final(values, rng, *, L, W, area_mm2, tier):
    perimeter_v, cm2_v, m2_v = values

    # Independent in-generator re-verification: recompute every part a SECOND way, straight from
    # L/W/area_mm2 -- never from a part's own already-returned value.
    check_equivalent(
        perimeter_v,
        2 * (L + W),
        "rectangle_area_perimeter_three_units: perimeter part disagrees with independent recomputation",
    )
    true_cm2 = round_half_up(Fraction(area_mm2, 100), 0)
    check_equivalent(
        cm2_v,
        true_cm2,
        "rectangle_area_perimeter_three_units: area_cm2 part disagrees with independent recomputation",
    )
    true_m2 = round_half_up(Fraction(area_mm2, 1_000_000), 0)
    check_equivalent(
        m2_v,
        true_m2,
        "rectangle_area_perimeter_three_units: area_m2 part disagrees with independent recomputation",
    )
    # The magnitude constraint, checked every call, not only at test time:
    # the RAW (pre-rounding) area_m2 must clear 1 m^2, well above the
    # _PAIR_TOLERANCE=0.051 danger zone.
    check(
        Fraction(area_mm2, 1_000_000) >= 1,
        "rectangle_area_perimeter_three_units: raw area_m2 below the 1 m^2 magnitude floor",
    )

    perimeter_text = str(perimeter_v)
    cm2_text = stated_precision_string(cm2_v, 0)
    m2_text = stated_precision_string(m2_v, 0)

    answer_text = f"perimeter_mm = {perimeter_text}, area_cm2 = {cm2_text}, area_m2 = {m2_text}"
    answer = {
        "perimeter_mm": perimeter_v,
        "area_cm2": numeric_answer(cm2_v),
        "area_m2": numeric_answer(m2_v),
    }

    text = phrase(rng, _TEMPLATE_BANKS[tier], L=L, W=W)

    # Misconceptions, one per part, held as a FULL labelled candidate string
    # (the other two parts correct) -- `answer: pair` compares by LABEL, so
    # a candidate must be a complete, differently-labelled triple.
    candidates = []

    wrong_cm2 = round_half_up(Fraction(area_mm2, 10), 0)  # divides by 10, not 100
    if wrong_cm2 != cm2_v:
        candidates.append(
            f"perimeter_mm = {perimeter_text}, area_cm2 = {stated_precision_string(wrong_cm2, 0)}, area_m2 = {m2_text}"
        )

    wrong_perimeter = area_mm2  # uses the AREA formula (L * W) for perimeter
    if wrong_perimeter != perimeter_v:
        candidates.append(f"perimeter_mm = {wrong_perimeter}, area_cm2 = {cm2_text}, area_m2 = {m2_text}")

    wrong_m2 = round_half_up(Fraction(area_mm2, 1000), 0)  # linear factor, not squared
    if wrong_m2 != m2_v:
        candidates.append(
            f"perimeter_mm = {perimeter_text}, area_cm2 = {cm2_text}, area_m2 = {stated_precision_string(wrong_m2, 0)}"
        )

    for cand in candidates:
        check(
            cand != answer_text,
            f"rectangle_area_perimeter_three_units: a distractor matches the answer text: {cand!r}",
        )

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=pick_distractors(answer_text, candidates, tolerance=GRADER_TOLERANCE),
    )


def _build_rectangle_area_perimeter(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    L, W = _draw_clean_sides(rng)
    area_mm2 = L * W
    perimeter_mm = 2 * (L + W)
    cm2_rounded = round_half_up(Fraction(area_mm2, 100), 0)
    m2_rounded = round_half_up(Fraction(area_mm2, 1_000_000), 0)

    part_perimeter = functools.partial(
        _part_perimeter,
        L=L,
        W=W,
        perimeter_mm=perimeter_mm,
    )
    part_area_cm2 = functools.partial(
        _part_area_cm2,
        L=L,
        W=W,
        area_mm2=area_mm2,
        cm2_rounded=cm2_rounded,
    )
    part_area_m2 = functools.partial(
        _part_area_m2,
        L=L,
        W=W,
        area_mm2=area_mm2,
        m2_rounded=m2_rounded,
    )
    final = functools.partial(_final, L=L, W=W, area_mm2=area_mm2, tier=tier)

    return compose(rng, [part_perimeter, part_area_cm2, part_area_m2], final=final)


@generator(
    "exd_e5_rectangle_area_perimeter_three_units",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def exd_e5_rectangle_area_perimeter_three_units(rng, tier: str = "easy"):
    return _build_rectangle_area_perimeter(rng, tier=tier)
