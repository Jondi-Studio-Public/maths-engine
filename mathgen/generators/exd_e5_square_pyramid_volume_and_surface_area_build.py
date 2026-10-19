"""Given a square pyramid's base side `s` and PERPENDICULAR height `h` (never
the slant height), find BOTH the volume and the TOTAL surface area (the base
square plus four congruent lateral triangular faces). The slant height
`l = sqrt(h**2 + (s/2)**2)` must be derived internally, shown in `working`
as its own step, and is never given to the learner directly.

**Distinct (`skills/measurement/senior-find-the-surface-area-of-pyramids-such-as.yaml`, topic
`find_the_surface_area_of_pyramids_such_as`): that skill is SURFACE-AREA-ONLY and covers general base shapes
(square/rectangular/ right-triangle); this is a NEW, separate, COMBINED (volume + surface area) skill
restricted specifically to a square base. Neither this module nor its test file imports anything from
`p6f_measurement_faces_build.py` -- it was read only, for house style.

## Construct so the slant height is EXACT -- removes `_PAIR_TOLERANCE =
0.051` (`mathgen/grading/typed.py:421`) entirely for this skill

`s/2` and `h` are drawn as the two LEGS of a curated, hand-checked
Pythagorean triple `(leg_a, leg_b, hyp)` -- `s/2 = leg_a` (so `s =
2*leg_a`), `h = leg_b`, `l = hyp` exactly, with ZERO rounding anywhere.
`volume = (1/3) * s**2 * h` and `surface_area = s**2 + 2*s*l` are then both
exact values for every draw.

`_TRIPLES` below is additionally curated so EVERY entry has `3 | leg_a` or
`3 | leg_b`. Since `volume = (1/3) * s**2 * h = (1/3) * (2*leg_a)**2 *
leg_b = (4/3) * leg_a**2 * leg_b`, this guarantees the `/3` always divides
out EXACTLY -- `volume` is an exact INTEGER on every draw, not merely an
exact rational that might carry a non-terminating decimal (a `Fraction`
with a reduced denominator of 3 has no terminating decimal at all, which
`mathgen.core.exact_decimal_string` would raise on). So every value this
skill ever displays or grades -- `volume`, `surface_area`, and the internal
`l` -- is a plain integer, proved in this module's own test file
(`test_every_triple_is_genuinely_pythagorean`,
`test_volume_is_always_an_exact_integer`), not merely asserted here.
`surface_area = s**2 + 2*s*l` is automatically an integer whenever `s` and
`l` are (both always true here by construction), so it needs no analogous
divisibility constraint.

## Tier escalation is wording-only (`hard` also changes the GIVEN quantity)

`scripts/measure_tiers.py`/`mathgen.text_quality.prompt_shape` masks every
digit run (including its leading sign) to a single placeholder, so "which
triple family was drawn" is INVISIBLE to the instrument regardless of which
triple filled in the numbers -- tiers escalate through explicit added
TEMPLATE WORDING, and at `hard` through a genuinely different GIVEN
quantity:

- **easy**: one plain sentence giving `s` (base side) and `h` (perpendicular
  height) directly.
- **medium**: the same givens, plus an explicit added clause asking for the
  slant height to be shown as its own separate working step.
- **hard**: gives the base as its AREA (`s**2`) instead of its side `s`,
  forcing an extra square-root step (recovering `s` from the stated area)
  BEFORE the Pythagoras step even begins -- a genuine structural escalation,
  not a bigger number. The changed NOUN ("base area" vs "base side") is not
  a `mathgen.slots` bank word, so it survives masking. The independent
  oracle in the test file derives `s` from the stated area itself
  (`math.isqrt`), the same clean-by-construction discipline as the rest of
  this module -- `s` always recovers exactly, since the area is always
  `s**2` for the very `s` drawn from `_TRIPLES`.
- **exam**: has the shape of a written exam question
  -- lettered (a)/(b) parts and an explicit "show your working" clause, its
  own distinct wording.
"""

from __future__ import annotations

import functools
import math
from fractions import Fraction

from ..core import (
    check,
    check_equivalent,
    compose,
    generator,
    latex_answer,
    phrase,
    pick_distractors,
)

_TIERS = ("easy", "medium", "hard", "exam")

# Curated, hand-checked (leg_a, leg_b, hyp) Pythagorean triples: leg_a = s/2,
# leg_b = h, hyp = l -- see module docstring for the exact-construction
# rationale. EVERY entry has `3 | leg_a` or `3 | leg_b`, guaranteeing
# `volume = (4/3) * leg_a**2 * leg_b` is always an exact integer -- proved
# for every entry (not merely the ones a seed sweep happens to draw) in this
# module's own test file.
_TRIPLES = (
    (4, 3, 5),  # 3 | leg_b
    (12, 5, 13),  # 3 | leg_a
    (6, 8, 10),  # 3 | leg_a
    (8, 15, 17),  # 3 | leg_b
    (9, 12, 15),  # 3 | leg_a and leg_b
    (7, 24, 25),  # 3 | leg_b
    (12, 16, 20),  # 3 | leg_a
    (15, 20, 25),  # 3 | leg_a
    (10, 24, 26),  # 3 | leg_b
    (20, 21, 29),  # 3 | leg_b
    (9, 40, 41),  # 3 | leg_a
    (18, 24, 30),  # 3 | leg_a and leg_b
)


def _draw_square_pyramid(rng):
    """Draw one (s, h, l, volume, surface_area) tuple from `_TRIPLES` --
    every value exact by construction, checked again here (not merely
    trusted from the table) so a live draw can never silently carry a
    transcription error past this point."""
    leg_a, leg_b, hyp = rng.choice(_TRIPLES)
    s = 2 * leg_a
    h = leg_b
    l = hyp
    check(
        l * l == h * h + (s // 2) * (s // 2), "square_pyramid_volume_and_surface_area: drawn triple is not Pythagorean"
    )
    volume_frac = Fraction(s * s * h, 3)
    check(volume_frac.denominator == 1, "square_pyramid_volume_and_surface_area: volume is not an exact integer")
    volume = int(volume_frac)
    surface_area = s * s + 2 * s * l
    return s, h, l, volume, surface_area


_EASY_TEMPLATES = (
    "A square pyramid has a base side length of {s} cm and a perpendicular height of "
    "{h} cm. Find the pyramid's volume, in cm³, and its total surface area, in cm².",
    "A pyramid with a square base has a base side length of {s} cm and a perpendicular "
    "height of {h} cm. Work out the pyramid's volume, in cm³, and its total surface "
    "area, in cm².",
    "A square-based pyramid has a base side length of {s} cm and a perpendicular height "
    "of {h} cm. Calculate the pyramid's volume, in cm³, and its total surface area, "
    "in cm².",
)
_MEDIUM_TEMPLATES = (
    "A square pyramid has a base side length of {s} cm and a perpendicular height of "
    "{h} cm. Find the pyramid's volume, in cm³, and its total surface area, in "
    "cm², showing the slant height as a separate step in your working.",
    "A pyramid with a square base has a base side length of {s} cm and a perpendicular "
    "height of {h} cm. Work out the pyramid's volume, in cm³, and its total surface "
    "area, in cm² -- show the slant height as a separate working step.",
    "A square-based pyramid has a base side length of {s} cm and a perpendicular height "
    "of {h} cm. Calculate the pyramid's volume, in cm³, and its total surface area, "
    "in cm², showing the slant height as its own separate step.",
)
_HARD_TEMPLATES = (
    "A square pyramid has a base area of {area} cm² and a perpendicular height of "
    "{h} cm. Find the pyramid's volume, in cm³, and its total surface area, in "
    "cm², showing the base side length and the slant height as separate working "
    "steps.",
    "A pyramid with a square base has a base area of {area} cm² and a perpendicular "
    "height of {h} cm. Work out the pyramid's volume, in cm³, and its total surface "
    "area, in cm² -- first find the base side length, then the slant height, showing "
    "both as separate steps.",
    "A square-based pyramid has a base area of {area} cm² and a perpendicular "
    "height of {h} cm. Calculate the pyramid's volume, in cm³, and its total surface "
    "area, in cm², deriving the base side length and the slant height as separate "
    "steps.",
)
_EXAM_TEMPLATES = (
    "A square pyramid has a base side length of {s} cm and a "
    "perpendicular height of {h} cm. Find (a) the pyramid's volume, in cm³, and (b) "
    "its total surface area, in cm², showing your working for each part.",
    "This exam question gives a square pyramid a base side length of {s} cm and a "
    "perpendicular height of {h} cm. With full working, find (a) the volume, in "
    "cm³, and (b) the total surface area, in cm².",
    "Exam question: a square-based pyramid has a base side length of {s} cm and a "
    "perpendicular height of {h} cm. Show your working to find (a) the volume, in "
    "cm³, and (b) the total surface area, in cm².",
)
_TEMPLATE_BANKS = {
    "easy": _EASY_TEMPLATES,
    "medium": _MEDIUM_TEMPLATES,
    "hard": _HARD_TEMPLATES,
    "exam": _EXAM_TEMPLATES,
}


# --- the three parts (module-level, so the test file can monkeypatch one by
# name -- the established pattern for a compose() part that needs shared
# per-question state without becoming un-patchable). The slant-height part
# is not one of the two GRADED labels but must appear in `working`
# as its own shown step. ------------------------


def _part_slant(rng, values, *, s, h, l):
    half_s = s // 2
    working = (
        f"Slant height: l = √(h² + (s/2)²) = √({h}² + "
        f"{half_s}²) = √({h * h} + {half_s * half_s}) = "
        f"√{h * h + half_s * half_s} = {l} cm."
    )
    return l, working


def _part_volume(rng, values, *, s, h, volume):
    working = (
        f"Volume: (1/3) × s² × h = (1/3) × {s}² × {h} = (1/3) × {s * s} × {h} = (1/3) × {s * s * h} = {volume} cm³."
    )
    return volume, working


def _part_surface_area(rng, values, *, s, l, surface_area):
    lateral = 2 * s * l
    working = f"Surface area: s² + 2 × s × l = {s}² + 2 × {s} × {l} = {s * s} + {lateral} = {surface_area} cm²."
    return surface_area, working


def _final(values, rng, *, s, h, l, tier):
    l_v, volume_v, sa_v = values

    # Independent in-generator re-verification: recompute every part a SECOND way, straight from s/h
    # -- never from a part's own already-returned value.
    half_s = s // 2
    recomputed_l_squared = h * h + half_s * half_s
    recomputed_l = math.isqrt(recomputed_l_squared)
    check(
        recomputed_l * recomputed_l == recomputed_l_squared,
        "square_pyramid_volume_and_surface_area: h/s do not imply an exact slant height",
    )
    check_equivalent(
        l_v,
        recomputed_l,
        "square_pyramid_volume_and_surface_area: slant-height part disagrees with independent recomputation",
    )
    true_volume = Fraction(s * s * h, 3)
    check(
        true_volume.denominator == 1,
        "square_pyramid_volume_and_surface_area: recomputed volume is not an exact integer",
    )
    check_equivalent(
        volume_v,
        int(true_volume),
        "square_pyramid_volume_and_surface_area: volume part disagrees with independent recomputation",
    )
    true_sa = s * s + 2 * s * recomputed_l
    check_equivalent(
        sa_v,
        true_sa,
        "square_pyramid_volume_and_surface_area: surface_area part disagrees with independent recomputation",
    )

    answer_text = f"volume = {volume_v}, surface area = {sa_v}"
    answer = {"volume": volume_v, "surface_area": sa_v}

    area = s * s
    text = phrase(rng, _TEMPLATE_BANKS[tier], s=s, h=h, area=area)

    # Misconceptions, each a FULL relabelled pair (`answer: pair` compares
    # by label, so a candidate must be a complete, differently-labelled
    # pair, not a lone number) -- drawn from `p6f_measurement_faces_build.py`'s list for the
    # genuinely-shared error modes, written for THIS skill's two-part
    # answer.
    candidates = []

    # uses_apex_height_as_slant_height: plugs h straight into the
    # surface-area formula instead of deriving l via Pythagoras first.
    # l > h always (l**2 = h**2 + (s/2)**2 with s/2 >= 4 > 0), so this
    # differs from the true surface area on every draw, by construction.
    wrong_sa_apex = s * s + 2 * s * h
    if wrong_sa_apex != sa_v:
        candidates.append(f"volume = {volume_v}, surface area = {wrong_sa_apex}")

    # omits_base_square: sums only the four triangular faces (2*s*l), no
    # base square -- always short by exactly s**2 > 0.
    wrong_sa_omits_base = 2 * s * l_v
    if wrong_sa_omits_base != sa_v:
        candidates.append(f"volume = {volume_v}, surface area = {wrong_sa_omits_base}")

    # no_half_factor_on_laterals: uses s*l per face instead of (1/2)*s*l --
    # always high by exactly 2*s*l > 0.
    wrong_sa_no_half = s * s + 4 * s * l_v
    if wrong_sa_no_half != sa_v:
        candidates.append(f"volume = {volume_v}, surface area = {wrong_sa_no_half}")

    # no_third_factor_on_volume: uses s**2*h instead of (1/3)*s**2*h --
    # always exactly 3x the true volume (which is always > 0).
    wrong_volume_no_third = s * s * h
    if wrong_volume_no_third != volume_v:
        candidates.append(f"volume = {wrong_volume_no_third}, surface area = {sa_v}")

    for cand in candidates:
        check(
            cand != answer_text,
            f"square_pyramid_volume_and_surface_area: a distractor matches the answer text: {cand!r}",
        )

    return dict(
        text=text,
        latex=text,
        answer=answer,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_text, plain=True),
        distractors=pick_distractors(answer_text, candidates),
    )


def _build_square_pyramid_volume_and_surface_area(rng, tier: str = "easy"):
    if tier not in _TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    s, h, l, volume, surface_area = _draw_square_pyramid(rng)

    part_slant = functools.partial(_part_slant, s=s, h=h, l=l)
    part_volume = functools.partial(_part_volume, s=s, h=h, volume=volume)
    part_surface_area = functools.partial(
        _part_surface_area,
        s=s,
        l=l,
        surface_area=surface_area,
    )
    final = functools.partial(_final, s=s, h=h, l=l, tier=tier)

    return compose(rng, [part_slant, part_volume, part_surface_area], final=final)


@generator(
    "exd_e5_square_pyramid_volume_and_surface_area",
    level=11,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
        "exam": dict(tier="exam"),
    },
)
def exd_e5_square_pyramid_volume_and_surface_area(rng, tier: str = "easy"):
    return _build_square_pyramid_volume_and_surface_area(rng, tier=tier)
