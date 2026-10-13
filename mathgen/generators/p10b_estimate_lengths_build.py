"""Generates `estimate_length_reference` questions: choose the best estimate of a real-world object's length from
options that differ by orders of magnitude.

Estimate lengths (ACMEM018).

``estimate_length_reference`` -- given a real-world reference object, choose
the best mcq estimate of one of its length-type attributes (length, height,
width or diameter). The answer is ``mcq``: the correctly-scaled numeral among
three deliberately WRONG-ORDER-OF-MAGNITUDE options (x10, /10 and x100 of the
same numeral, in the SAME asked unit) -- never a same-order "plausible but
wrong" option. A learner whose own reasonable estimate differs from the
canonical value by anything less than a full order of magnitude still lands
on the one option closest to it, so no defensible estimate is marked wrong.
This is why the skill needs no graded band: "reasonable" is decided by which
OPTION is closest, not by comparing a typed number to a tolerance.

Difficulty is the asked UNIT, not merely which half of the catalogue is
drawn from -- a genuine reasoning difference (how much conversion the
learner must do from their own metre-scale benchmark), reusing the stated
prerequisite ``convert-metric-length`` directly:

- easy: a MACRO object (natural scale already metres) asked in METRES, the
  object's own natural unit -- one-step reasoning.
- medium: a MICRO object (natural scale smaller than a metre) asked in
  CENTIMETRES, again the object's own natural unit.
- hard: any object (macro or micro), asked in the unit that is NOT its own
  natural scale -- millimetres for a macro object, metres for a micro one --
  forcing a genuine order-of-magnitude conversion a memorised same-unit
  benchmark cannot shortcut.

Every option in a question is formatted to the SAME number of decimal places (``_format_options``) -- the
shortest exact decimal that displays the smallest of the four values (always the /10 distractor) exactly -- so
a learner cannot pick the correct option by counting decimal digits rather than reasoning about scale.
"""

from __future__ import annotations

from fractions import Fraction

from .r2s_scale_options import scaled_options
from ..core import (
    check,
    check_conversion,
    check_distinct,
    fixed_point_string,
    generator,
    latex_answer,
    pick_distractors,
)

# Object catalogue. Every canonical value is stored ONCE, in metres. ``_MACRO``'s natural unit is
# metres (used directly by easy, and as the non-natural "mm" target for hard); ``_MICRO``'s natural
# unit is centimetres (used directly by medium, and as the non-natural "m" target for hard). Each
# entry is (name including its own article, attribute word, canonical metres as an exact Fraction).

_MACRO: list[tuple[str, str, Fraction]] = [
    ("a family car", "length", Fraction("4.5")),
    ("a city bus", "length", Fraction("12")),
    ("an adult door", "height", Fraction("2")),
    ("a canoe", "length", Fraction("4")),
    ("a queen-size bed", "length", Fraction("2")),
    ("an adult person", "height", Fraction("1.7")),
    ("a bathtub", "length", Fraction("1.5")),
    ("a soccer pitch", "length", Fraction("100")),
    ("a tennis court", "length", Fraction("24")),
    ("a telegraph pole", "height", Fraction("9")),
    ("a shipping container", "length", Fraction("12")),
    ("an Olympic swimming pool", "length", Fraction("50")),
    ("an adult giraffe", "height", Fraction("5")),
    ("a classroom", "length", Fraction("9")),
    ("a double-decker bus", "height", Fraction("4.4")),
    ("a small yacht", "length", Fraction("8")),
    ("a cricket pitch", "length", Fraction("20")),
    ("a basketball court", "length", Fraction("28")),
    ("a killer whale", "length", Fraction("7")),
    ("a fire engine", "length", Fraction("10")),
    ("a train carriage", "length", Fraction("22")),
    ("a semi-trailer truck", "length", Fraction("19")),
    ("a flagpole", "height", Fraction("6")),
    ("a large gum tree", "height", Fraction("25")),
    ("a rowing eight", "length", Fraction("18")),
    ("a highway lane", "width", Fraction("3.5")),
    ("a park bench", "length", Fraction("1.8")),
    ("a stretch limousine", "length", Fraction("9")),
]

_MICRO: list[tuple[str, str, Fraction]] = [
    ("a pencil", "length", Fraction("0.18")),
    ("a smartphone", "height", Fraction("0.16")),
    ("a dinner plate", "diameter", Fraction("0.27")),
    ("an adult shoe", "length", Fraction("0.28")),
    ("a paperback book", "height", Fraction("0.20")),
    ("a laptop", "width", Fraction("0.33")),
    ("a tennis ball", "diameter", Fraction("0.07")),
    ("a soft drink can", "height", Fraction("0.12")),
    ("a bar of soap", "length", Fraction("0.10")),
    ("a wallet", "length", Fraction("0.11")),
    ("a house key", "length", Fraction("0.06")),
    ("a dinner fork", "length", Fraction("0.19")),
    ("a computer mouse", "length", Fraction("0.10")),
    ("a slice of bread", "width", Fraction("0.12")),
    ("a golf ball", "diameter", Fraction("0.04")),
    ("a standard brick", "length", Fraction("0.23")),
    ("an A4 exercise book", "height", Fraction("0.30")),
    ("a table tennis paddle", "length", Fraction("0.26")),
    ("a banana", "length", Fraction("0.18")),
    ("a wine bottle", "height", Fraction("0.30")),
    ("a teaspoon", "length", Fraction("0.14")),
    ("a hardcover textbook", "width", Fraction("0.21")),
    ("a USB flash drive", "length", Fraction("0.05")),
    ("a matchbox", "length", Fraction("0.05")),
    ("a chess piece", "height", Fraction("0.09")),
    ("a hairbrush", "length", Fraction("0.20")),
    ("a DL envelope", "length", Fraction("0.22")),
    ("a paperclip", "length", Fraction("0.03")),
]

# Amount of "mm" equal to one of each unit -- an independent table from
# `_DIRECT_FACTOR_FROM_M` below, deliberately not shared with construction,
# so the in-generator `check_conversion` re-derives every option via a
# different pivot (metres -> mm -> the asked unit) than construction's own
# direct metres -> unit factor.
_MM_PER_UNIT = {"mm": Fraction(1), "cm": Fraction(10), "m": Fraction(1000)}

# Construction's own direct metres -> unit factor.
_DIRECT_FACTOR_FROM_M = {"m": Fraction(1), "cm": Fraction(100), "mm": Fraction(1000)}

_UNIT_WORD = {"m": "metres", "cm": "centimetres", "mm": "millimetres"}
_UNIT_ABBR = {"m": "m", "cm": "cm", "mm": "mm"}


def _exact_places_needed(value: Fraction) -> int:
    """The fewest decimal places that display `value` exactly."""
    places = 0
    while (value * (10**places)).denominator != 1:
        places += 1
    return places


def _format_options(correct: Fraction, x10: Fraction, tenth: Fraction, x100: Fraction, unit: str):
    """Format all four option values to the SAME number of decimal places --
    the count `tenth` (always the smallest, and the one most likely to need
    an extra digit) needs to display exactly -- so decimal-digit count never
    tells a learner which option is correct (module docstring)."""
    places = _exact_places_needed(tenth)
    unit_abbr = _UNIT_ABBR[unit]

    def fmt(value: Fraction) -> str:
        scaled = value * (10**places)
        check(scaled.denominator == 1, f"{value} does not terminate exactly within {places} places")
        return f"{fixed_point_string(int(scaled), places)} {unit_abbr}"

    return fmt(correct), fmt(x10), fmt(tenth), fmt(x100)


def _build(rng, tier: str) -> dict:
    if tier == "easy":
        name, attribute, canonical = rng.choice(_MACRO)
        unit = "m"
    elif tier == "medium":
        name, attribute, canonical = rng.choice(_MICRO)
        unit = "cm"
    elif tier == "hard":
        is_macro = rng.choice([True, False])
        name, attribute, canonical = rng.choice(_MACRO if is_macro else _MICRO)
        unit = "mm" if is_macro else "m"
    else:
        raise ValueError(f"unknown tier {tier!r}")

    correct = canonical * _DIRECT_FACTOR_FROM_M[unit]
    check_conversion(canonical, "m", unit, correct, _MM_PER_UNIT, label="estimate-length unit conversion")

    correct_text, distractors = scaled_options(rng, correct, f" {_UNIT_ABBR[unit]}")

    text = f"What is the best estimate for the {attribute} of {name}? Give your answer in {_UNIT_WORD[unit]}."
    latex = f"plain:{text}"
    working = [
        f"{name} is naturally around {canonical} m.",
        f"{canonical} m expressed in {_UNIT_WORD[unit]} is {correct_text}.",
    ]
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
    "estimate_length_reference",
    level=12,
    presets={
        "easy": dict(tier="easy"),
        "medium": dict(tier="medium"),
        "hard": dict(tier="hard"),
    },
)
def estimate_length_reference(rng, tier="easy"):
    """Best mcq estimate of a real-world object's length/height/width/diameter,
    among the correctly-scaled numeral and its x10/x0.1/x100 order-of-
    magnitude distractors. `tier` selects the object pool and the asked
    unit -- see the module docstring."""
    return _build(rng, tier)
