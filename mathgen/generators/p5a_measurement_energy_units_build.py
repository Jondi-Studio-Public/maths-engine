"""Energy-unit conversions, senior ``acara-senior-v8-4``, domain ``measurement``, ``answer: number``.

One generator:

- ``convert_energy_units`` (``senior-convert-from-one-unit-of-energy-to-another``,
  ACMEM034) -- convert an energy measurement between joules, kilojoules, megajoules,
  and kilowatt-hours. The hard tier uses an irregular non-decimal-prefix factor
  (1 kWh = 3600 kJ) as a genuine reasoning escalation.

The generator formats ``answer_text`` from an exact ``Fraction``/``int`` value with
explicit, bounded fixed-point decimal places (never ``str(float(...))``, which can
silently switch to scientific notation below ``1e-4``) -- see
``core.exact_decimal_string``. Easy and hard tiers always produce exact integers by construction.
Medium tier allows genuine decimal results.
"""

from __future__ import annotations

from fractions import Fraction

from ..core import (
    check_conversion,
    exact_decimal_string,
    generator,
    numeric_answer,
    pick_distractors,
    stated_precision_string,
)

# Base factor table, energy expressed in joules. All conversions pivot through
# this table via check_conversion.
_FACTORS_J = {"J": Fraction(1), "kJ": Fraction(1_000), "MJ": Fraction(1_000_000), "kWh": Fraction(3_600_000)}
_UNIT_LABELS = {"J": "J", "kJ": "kJ", "MJ": "MJ", "kWh": "kWh"}

_FACTOR_J_KJ = _FACTORS_J["kJ"]  # 1,000
_FACTOR_KJ_MJ = _FACTORS_J["MJ"] / _FACTORS_J["kJ"]  # 1,000
_FACTOR_KJ_KWH = _FACTORS_J["kWh"] / _FACTORS_J["kJ"]  # 3,600


def _shaped_distractors(answer_text: str, answer: Fraction, wrong_values: list) -> list[str]:
    """Distractors shaped like the key (same decimal places, so the point never
    gives the answer away): keep the misconception values that land exactly on
    the answer's decimal grid, then top up with power-of-ten slips of the answer."""
    places = len(answer_text.partition(".")[2])
    scale = 10**places
    pool = list(wrong_values) + [answer * 10, answer * 100, answer / 10, answer / 100, answer * 1000]
    return pick_distractors(
        answer_text,
        [stated_precision_string(Fraction(v), places) for v in pool if (Fraction(v) * scale).denominator == 1],
    )


def _build_j_kj(rng, kj_hi: int) -> dict:
    """Easy tier: single-step J<->kJ conversion, always an exact result
    (the kJ figure is chosen first, as an integer, and the J figure is
    derived as an exact multiple of it)."""
    direction = rng.choice(["j_to_kj", "kj_to_j"])
    kj_value = Fraction(rng.randint(1, kj_hi))
    j_value = kj_value * _FACTOR_J_KJ

    if direction == "j_to_kj":
        source_value, source_unit, target_unit = j_value, "J", "kJ"
        answer = kj_value
        direction_wrong = source_value * _FACTOR_J_KJ  # multiplies instead of dividing
        wrong_factor = source_value / Fraction(10_000)  # 10,000 instead of 1,000 (wrong-power-of-ten)
    else:
        source_value, source_unit, target_unit = kj_value, "kJ", "J"
        answer = j_value
        direction_wrong = source_value / _FACTOR_J_KJ  # divides instead of multiplying
        wrong_factor = source_value * Fraction(10_000)  # 10,000 instead of 1,000

    check_conversion(source_value, source_unit, target_unit, answer, _FACTORS_J, label="conversion")

    answer_text = exact_decimal_string(answer, max_places=3)
    source_text = exact_decimal_string(source_value, max_places=3)
    text = f"Convert {source_text} {_UNIT_LABELS[source_unit]} to {_UNIT_LABELS[target_unit]}."

    return dict(
        text=text,
        latex=rf"{source_text}\,\mathrm{{{_UNIT_LABELS[source_unit]}}}=\square\,\mathrm{{{_UNIT_LABELS[target_unit]}}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"1 kJ = 1000 J, so {source_text} {_UNIT_LABELS[source_unit]} = {answer_text} {_UNIT_LABELS[target_unit]}."
        ],
        distractors=_shaped_distractors(answer_text, answer, [wrong_factor, direction_wrong]),
    )


def _build_kj_mj(rng, mj_numerator_lo: int, mj_numerator_hi: int) -> dict:
    """Medium tier: single-step kJ<->MJ conversion (factor 1000, the same
    metric-prefix shift as easy but at the next magnitude). Allow genuine
    decimal results. For mj_to_kj, draw an MJ source with up to 3 decimal
    places so the kJ result is a whole number. For kj_to_mj, draw a kJ
    source NOT a multiple of 1000 so the MJ result is a genuine decimal."""
    direction = rng.choice(["kj_to_mj", "mj_to_kj"])

    if direction == "kj_to_mj":
        kj_value = Fraction(rng.randint(1001, 49999))
        source_value, source_unit, target_unit = kj_value, "kJ", "MJ"
        answer = kj_value / _FACTOR_KJ_MJ
        direction_wrong = source_value * _FACTOR_KJ_MJ
        wrong_factor = source_value / Fraction(10_000)  # 10,000 instead of 1,000
    else:
        mj_value = Fraction(rng.randint(mj_numerator_lo, mj_numerator_hi), 1000)
        source_value, source_unit, target_unit = mj_value, "MJ", "kJ"
        answer = mj_value * _FACTOR_KJ_MJ
        direction_wrong = source_value / _FACTOR_KJ_MJ
        wrong_factor = source_value * Fraction(10_000)  # 10,000 instead of 1,000

    check_conversion(source_value, source_unit, target_unit, answer, _FACTORS_J, label="conversion")

    answer_text = exact_decimal_string(answer, max_places=3)
    source_text = exact_decimal_string(source_value, max_places=3)
    text = f"Convert {source_text} {_UNIT_LABELS[source_unit]} to {_UNIT_LABELS[target_unit]}."

    return dict(
        text=text,
        latex=rf"{source_text}\,\mathrm{{{_UNIT_LABELS[source_unit]}}}=\square\,\mathrm{{{_UNIT_LABELS[target_unit]}}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"1 MJ = 1000 kJ, so {source_text} {_UNIT_LABELS[source_unit]} = {answer_text} {_UNIT_LABELS[target_unit]}."
        ],
        distractors=_shaped_distractors(answer_text, answer, [wrong_factor, direction_wrong]),
    )


def _build_kwh_kj(rng, kj_hi: int) -> dict:
    """Hard tier: single-step kWh<->kJ conversion using the irregular factor
    3600 (not a power of ten) -- a genuine reasoning escalation requiring
    recall of the real-world household-energy constant (1 kWh = 3600 kJ).
    Direction randomised; choose values so results are always exact integers."""
    direction = rng.choice(["kwh_to_kj", "kj_to_kwh"])
    kwh_value = Fraction(rng.randint(1, kj_hi))
    kj_value = kwh_value * _FACTOR_KJ_KWH

    if direction == "kwh_to_kj":
        source_value, source_unit, target_unit = kwh_value, "kWh", "kJ"
        answer = kj_value
        direction_wrong = source_value / _FACTOR_KJ_KWH  # divides instead of multiplying
        wrong_factor = source_value * Fraction(1_000)  # 1,000 (kwh-as-plain-kilo) instead of 3,600
    else:
        source_value, source_unit, target_unit = kj_value, "kJ", "kWh"
        answer = kwh_value
        direction_wrong = source_value * _FACTOR_KJ_KWH  # multiplies instead of dividing
        wrong_factor = source_value / Fraction(1_000)  # 1,000 (kwh-as-plain-kilo) instead of 3,600

    check_conversion(source_value, source_unit, target_unit, answer, _FACTORS_J, label="conversion")

    answer_text = exact_decimal_string(answer, max_places=3)
    source_text = exact_decimal_string(source_value, max_places=3)
    text = f"Convert {source_text} {_UNIT_LABELS[source_unit]} to {_UNIT_LABELS[target_unit]}."

    return dict(
        text=text,
        latex=rf"{source_text}\,\mathrm{{{_UNIT_LABELS[source_unit]}}}=\square\,\mathrm{{{_UNIT_LABELS[target_unit]}}}",
        answer=numeric_answer(answer),
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[
            f"1 kWh = 3600 kJ, so {source_text} {_UNIT_LABELS[source_unit]} = "
            f"{answer_text} {_UNIT_LABELS[target_unit]}."
        ],
        distractors=_shaped_distractors(answer_text, answer, [wrong_factor, direction_wrong]),
    )


@generator(
    "convert_energy_units",
    level=11,
    presets={
        "easy": dict(mode="j_kj"),
        "medium": dict(mode="kj_mj"),
        "hard": dict(mode="kwh_kj"),
    },
)
def convert_energy_units(
    rng,
    mode="j_kj",
    kj_hi=50,
    mj_numerator_lo=1001,
    mj_numerator_hi=49999,
):
    """Convert an energy measurement between joules, kilojoules, megajoules,
    and kilowatt-hours; see module docstring for the hard-tier reasoning
    escalation using the irregular 3600 factor."""
    if mode == "j_kj":
        return _build_j_kj(rng, kj_hi)
    if mode == "kj_mj":
        return _build_kj_mj(rng, mj_numerator_lo, mj_numerator_hi)
    if mode == "kwh_kj":
        return _build_kwh_kj(rng, kj_hi)
    raise ValueError(f"unknown mode: {mode!r}")
