"""Estimate the areas of different shapes (ACMEM022) and estimate volume/capacity of various objects (ACMEM029,
capacity-only -- see below).

A learner whose own reasonable estimate differs from the canonical value by anything less than a full order of
magnitude still lands on the option closest to it, so no defensible estimate is marked wrong.

``estimate_area_reference`` (ACMEM022) uses two units only, m2 and cm2 --
area estimation never needs a third rung the way length used mm/cm/m.
Difficulty is the asked unit, not merely which half of the catalogue is
drawn from:

- easy: a MACRO object (natural unit m2, >= 1 m2) asked in square metres --
  one-step reasoning.
- medium: a MICRO object (natural unit cm2, < 1 m2) asked in square
  centimetres -- one-step reasoning over a different natural scale.
- hard: either half (50/50), asked in the OTHER half's unit -- a macro
  object asked in cm2 (a x10 000 jump) or a micro object asked in m2 (a
  /10 000 jump). The genuine escalation is recognising that an AREA unit
  scales by the SQUARE of the length factor (100**2 = 10 000, not 100) --
  the exact class of magnitude mistake the module docstring of
  ``p10b_estimate_lengths_build`` warns about: get the squared factor right BY CONSTRUCTION (the direct
  factor table below), never by eyeballing it, and re-derive it a second,
  independent way (``check_conversion`` against ``_CM2_PER_UNIT``).

``estimate_volume_capacity_reference`` (ACMEM029) is DELIBERATELY NARROWED to CAPACITY only
(litres/millilitres, for containers and vessels). The ACMEM029 descriptor names both "volume" and "capacity"
together, but a bathtub's volume (the material of its shell) and its capacity (how much it holds) are
different quantities, and a single deterministic mcq generator cannot coherently assess both on the same
object without conflating them. Units are L and mL, difficulty is the asked unit exactly as for area:

- easy: a MACRO container (natural unit L, >= 1 L) asked in litres.
- medium: a MICRO container (natural unit mL, < 1000 mL) asked in
  millilitres.
- hard: either half (50/50), asked in the OTHER half's unit -- a macro
  object asked in mL (a x1000 jump) or a micro object asked in L (a /1000
  jump), the ordinary linear L<->mL factor (no squaring; capacity is a
  volume-of-liquid quantity carried directly in litre-scaled units, not a
  length-derived one).

Every option in a question is formatted to the SAME number of decimal places (``_format_options``, written
once at module scope and shared by both topics) -- the shortest exact decimal that displays the smallest of
the four values (always the /10 distractor) exactly -- so a learner cannot pick the correct option by counting
decimal digits rather than reasoning about scale.
"""

from __future__ import annotations

from fractions import Fraction

from .r2s_scale_options import scaled_options
from ..core import (
    exact_decimal_text,
    check,
    check_conversion,
    check_pairwise_distinct,
    fixed_point_string,
    generator,
    latex_answer,
    pick_distractors,
)

# ---------------------------------------------------------------------------
# Area catalogues. Canonical values are stored in each half's OWN natural
# unit (macro: m2; micro: cm2) -- unlike the length skill, area's two halves
# do not share one common base unit, since re-expressing every micro object
# in m2 up front would make the catalogue harder to eyeball/verify.

_MACRO_AREA: list[tuple[str, Fraction]] = [
    ("a classroom floor", Fraction(60)),
    ("a tennis court", Fraction(200)),
    ("a basketball court", Fraction(420)),
    ("a football (soccer) pitch", Fraction(7000)),
    ("a badminton court", Fraction(80)),
    ("a squash court", Fraction(62)),
    ("an Olympic swimming pool surface", Fraction(1250)),
    ("a suburban house block", Fraction(600)),
    ("a car parking space", Fraction(12)),
    ("a shipping container floor", Fraction(28)),
    ("a volleyball court", Fraction(162)),
    ("a netball court", Fraction(465)),
    ("a queen bed surface", Fraction(3)),
    ("a double garage floor", Fraction(36)),
    ("a school hall floor", Fraction(400)),
    ("a cricket pitch", Fraction(60)),
    ("a standard door surface", Fraction(8, 5)),
    ("a car roof", Fraction(2)),
    ("a table tennis table", Fraction(41, 10)),
    ("a round trampoline (3.6 m diameter)", Fraction(10)),
    ("an average bedroom floor", Fraction(12)),
    ("a kitchen floor", Fraction(15)),
    ("a residential solar panel", Fraction(17, 10)),
    ("an A0 poster sheet", Fraction(1)),
]

_MICRO_AREA: list[tuple[str, Fraction]] = [
    ("a smartphone screen", Fraction(100)),
    ("a credit card", Fraction(46)),
    ("a postage stamp", Fraction(4)),
    ("a sheet of A4 paper", Fraction(624)),
    ("a drink coaster", Fraction(100)),
    ("a dinner plate surface", Fraction(570)),
    ("a slice of bread", Fraction(100)),
    ("a laptop screen", Fraction(500)),
    ("a business card", Fraction(45)),
    ("a coffee mug base", Fraction(30)),
    ("a paperback book cover", Fraction(350)),
    ("a chess board", Fraction(2000)),
    ("a tablet screen", Fraction(300)),
    ("a birthday card", Fraction(150)),
    ("a woven placemat", Fraction(1000)),
    ("a bathroom tile", Fraction(400)),
    ("a passport photo", Fraction(15)),
    ("a light switch plate", Fraction(80)),
    ("a round birthday cake (20 cm diameter)", Fraction(314)),
    ("a large pizza (30 cm diameter)", Fraction(707)),
    ("a picture frame", Fraction(500)),
    ("a laptop keyboard", Fraction(420)),
    ("an open wallet", Fraction(100)),
    ("a face towel", Fraction(900)),
    ("a DL envelope", Fraction(240)),
]

# ---------------------------------------------------------------------------
# Capacity catalogues (litres / millilitres).

_MACRO_CAPACITY: list[tuple[str, Fraction]] = [
    ("a bathtub (full)", Fraction(150)),
    ("a bucket", Fraction(10)),
    ("a household wheelie bin", Fraction(240)),
    ("a car fuel tank", Fraction(55)),
    ("a washing machine drum", Fraction(60)),
    ("a cooler/esky box", Fraction(40)),
    ("a kitchen sink", Fraction(20)),
    ("a household hot water system", Fraction(250)),
    ("a watering can", Fraction(9)),
    ("a small fish tank", Fraction(40)),
    ("a large saucepan", Fraction(5)),
    ("a school backpack (packed capacity)", Fraction(20)),
    ("a jerry can", Fraction(20)),
    ("a wine barrel", Fraction(225)),
    ("a beer keg", Fraction(50)),
    ("a garden compost bin", Fraction(200)),
    ("a mini rubbish skip", Fraction(3000)),
    ("a backyard spa", Fraction(1500)),
    ("a small rainwater tank", Fraction(1000)),
    ("a shopping trolley", Fraction(150)),
    ("a large stockpot", Fraction(12)),
    ("a car boot", Fraction(400)),
    ("a household fridge", Fraction(400)),
    ("a small above-ground pool", Fraction(20000)),
]

_MICRO_CAPACITY: list[tuple[str, Fraction]] = [
    ("a teaspoon", Fraction(5)),
    ("a tablespoon", Fraction(15)),
    ("a shot glass", Fraction(30)),
    ("a can of soft drink", Fraction(375)),
    ("a coffee mug", Fraction(300)),
    ("a standard wine glass pour", Fraction(150)),
    ("a small water bottle", Fraction(500)),
    ("a small carton of milk", Fraction(250)),
    ("a chicken egg", Fraction(60)),
    ("a tennis ball", Fraction(150)),
    ("a beer pot/middy glass", Fraction(285)),
    ("an espresso cup", Fraction(60)),
    ("a soup bowl", Fraction(400)),
    ("a juice box", Fraction(200)),
    ("a small perfume bottle", Fraction(50)),
    ("a travel hand sanitiser bottle", Fraction(60)),
    ("a small yoghurt tub", Fraction(150)),
    ("a squeeze honey bottle", Fraction(340)),
    ("a table tennis ball", Fraction(33, 1)),
    ("a lemon", Fraction(120)),
    # Kept the object and canonical value; dropped the parenthetical.
    ("a matchbox", Fraction(10)),
    # 21 objects remain, still clearing the >=20 floor.
]

# ---------------------------------------------------------------------------
# Unit tables. Each ``_CM2_PER_UNIT``/``_ML_PER_UNIT`` pivot table is used
# ONLY by the independent `check_conversion` call, deliberately kept
# separate from the direct construction factor table so construction and
# verification never share the same arithmetic path.

_AREA_DIRECT_FACTOR: dict[tuple[str, str], Fraction] = {
    ("m2", "m2"): Fraction(1),
    ("cm2", "cm2"): Fraction(1),
    ("m2", "cm2"): Fraction(10000),
    ("cm2", "m2"): Fraction(1, 10000),
}
_CM2_PER_UNIT = {"cm2": Fraction(1), "m2": Fraction(10000)}
_AREA_UNIT_WORD = {"m2": "square metres", "cm2": "square centimetres"}
_AREA_UNIT_ABBR = {"m2": "m²", "cm2": "cm²"}

_CAPACITY_DIRECT_FACTOR: dict[tuple[str, str], Fraction] = {
    ("L", "L"): Fraction(1),
    ("mL", "mL"): Fraction(1),
    ("L", "mL"): Fraction(1000),
    ("mL", "L"): Fraction(1, 1000),
}
_ML_PER_UNIT = {"mL": Fraction(1), "L": Fraction(1000)}
_CAPACITY_UNIT_WORD = {"L": "litres", "mL": "millilitres"}
_CAPACITY_UNIT_ABBR = {"L": "L", "mL": "mL"}


def _exact_places_needed(value: Fraction) -> int:
    """The fewest decimal places that display `value` exactly."""
    places = 0
    while (value * (10**places)).denominator != 1:
        places += 1
    return places


def _format_options(correct: Fraction, x10: Fraction, tenth: Fraction, x100: Fraction, unit_abbr: str):
    """Format all four option values to the SAME number of decimal places --
    the count `tenth` (always the smallest, and the one most likely to need
    an extra digit) needs to display exactly -- so decimal-digit count never
    tells a learner which option is correct (module docstring). Shared, at
    module scope, by both topics -- do not fork a per-topic copy."""
    places = _exact_places_needed(tenth)

    def fmt(value: Fraction) -> str:
        scaled = value * (10**places)
        check(scaled.denominator == 1, f"{value} does not terminate exactly within {places} places")
        return f"{fixed_point_string(int(scaled), places)} {unit_abbr}"

    return fmt(correct), fmt(x10), fmt(tenth), fmt(x100)


def _build_area(rng, tier: str) -> dict:
    if tier == "easy":
        name, canonical = rng.choice(_MACRO_AREA)
        source_unit = target_unit = "m2"
    elif tier == "medium":
        name, canonical = rng.choice(_MICRO_AREA)
        source_unit = target_unit = "cm2"
    elif tier == "hard":
        is_macro = rng.choice([True, False])
        if is_macro:
            name, canonical = rng.choice(_MACRO_AREA)
            source_unit, target_unit = "m2", "cm2"
        else:
            name, canonical = rng.choice(_MICRO_AREA)
            source_unit, target_unit = "cm2", "m2"
    else:
        raise ValueError(f"unknown tier {tier!r}")

    correct = canonical * _AREA_DIRECT_FACTOR[(source_unit, target_unit)]
    check_conversion(canonical, source_unit, target_unit, correct, _CM2_PER_UNIT, label="estimate-area unit conversion")

    correct_text, distractors = scaled_options(rng, correct, f" {_AREA_UNIT_ABBR[target_unit]}")

    unit_word = _AREA_UNIT_WORD[target_unit]
    text = f"What is the best estimate for the area of {name}? Give your answer in {unit_word}."
    latex = f"plain:{text}"
    working = [
        f"{name} has an area of about {exact_decimal_text(canonical)} {_AREA_UNIT_ABBR[source_unit]}.",
        f"{exact_decimal_text(canonical)} {_AREA_UNIT_ABBR[source_unit]} expressed in {unit_word} is {correct_text}.",
    ][: 2 if source_unit != target_unit else 1]
    return dict(
        text=text,
        latex=latex,
        answer=correct_text,
        answer_text=correct_text,
        answer_latex=latex_answer(correct_text, plain=True),
        working=working,
        distractors=distractors,
    )


def _build_capacity(rng, tier: str) -> dict:
    if tier == "easy":
        name, canonical = rng.choice(_MACRO_CAPACITY)
        source_unit = target_unit = "L"
    elif tier == "medium":
        name, canonical = rng.choice(_MICRO_CAPACITY)
        source_unit = target_unit = "mL"
    elif tier == "hard":
        is_macro = rng.choice([True, False])
        if is_macro:
            name, canonical = rng.choice(_MACRO_CAPACITY)
            source_unit, target_unit = "L", "mL"
        else:
            name, canonical = rng.choice(_MICRO_CAPACITY)
            source_unit, target_unit = "mL", "L"
    else:
        raise ValueError(f"unknown tier {tier!r}")

    correct = canonical * _CAPACITY_DIRECT_FACTOR[(source_unit, target_unit)]
    check_conversion(
        canonical, source_unit, target_unit, correct, _ML_PER_UNIT, label="estimate-capacity unit conversion"
    )

    correct_text, distractors = scaled_options(rng, correct, f" {_CAPACITY_UNIT_ABBR[target_unit]}")

    unit_word = _CAPACITY_UNIT_WORD[target_unit]
    text = f"What is the best estimate for the capacity of {name}? Give your answer in {unit_word}."
    latex = f"plain:{text}"
    working = [
        f"{name} holds about {exact_decimal_text(canonical)} {_CAPACITY_UNIT_ABBR[source_unit]}.",
        f"{exact_decimal_text(canonical)} {_CAPACITY_UNIT_ABBR[source_unit]} expressed in {unit_word} is {correct_text}.",
    ][: 2 if source_unit != target_unit else 1]
    return dict(
        text=text,
        latex=latex,
        answer=correct_text,
        answer_text=correct_text,
        answer_latex=latex_answer(correct_text, plain=True),
        working=working,
        distractors=distractors,
    )


@generator(
    "estimate_area_reference",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def estimate_area_reference(rng, tier="easy"):
    """Best mcq estimate of a real-world object's area, among the
    correctly-scaled numeral and its x10/x0.1/x100 order-of-magnitude
    distractors. `tier` selects the object pool and the asked unit -- see
    the module docstring."""
    return _build_area(rng, tier)


@generator(
    "estimate_volume_capacity_reference",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def estimate_volume_capacity_reference(rng, tier="easy"):
    """Best mcq estimate of a real-world container's CAPACITY (litres or
    millilitres -- never solid-object volume), among the correctly-scaled
    numeral and its x10/x0.1/x100 order-of-magnitude distractors. `tier`
    selects the object pool and the asked unit -- see the module
    docstring."""
    return _build_capacity(rng, tier)
