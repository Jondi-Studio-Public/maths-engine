"""Independent oracle for the `p10d_map_scale_distance` questions: it re-derives each answer from the rendered
question text.

Every value in this skill is an exact rational (curated pools, never an irrational factor), so there is never
a rounding disagreement to tolerate -- a mismatch here is a defect, full stop.

Three sentence shapes, one per tier:

- easy:   "a scale of 1 cm to {d} {unit} is used ... {distance} cm apart
           ... in {unit}, to 1 decimal place" -- verbal scale, forward.
- medium: "a scale of 1 : {n} ... distance ... on the map is {distance} cm
           ... actual distance ... in {unit}[, to 1 decimal place]" --
           ratio scale, forward, with a forced unit conversion.
- hard:   "a scale of 1 : {n} ... real-world distance ... is {value} {unit}
           ... distance ... on the map, in cm" -- ratio scale, REVERSE.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

TOPIC = "map_scale_distance"
PARAMS = {
    "easy": dict(mode="verbal_forward"),
    "medium": dict(mode="ratio_forward"),
    "hard": dict(mode="ratio_reverse"),
}


def _format_exact(value: Fraction, places: int) -> str:
    scaled = value * (10**places)
    assert scaled.denominator == 1, f"{value} is not exact to {places} decimal places"
    n = scaled.numerator
    sign = "-" if n < 0 else ""
    n = abs(n)
    if places == 0:
        return f"{sign}{n}"
    digits = str(n).zfill(places + 1)
    whole, frac = digits[:-places], digits[-places:]
    return f"{sign}{whole}.{frac}"


# --- easy: verbal scale, forward -------------------------------------------

_EASY_RE = re.compile(
    r"a scale of 1 cm to (\d+(?:\.\d+)?) (km|m) is used\. "
    r"Two landmarks are (\d+(?:\.\d+)?) cm apart"
)


def _oracle_easy(text: str) -> tuple[Fraction, int]:
    match = _EASY_RE.search(text)
    assert match, f"could not parse verbal scale from: {text!r}"
    scale, unit, distance = match.groups()
    real_value = Fraction(scale) * Fraction(distance)
    assert "to 1 decimal place" in text, f"easy tier always states its precision: {text!r}"
    return real_value, 1


# --- medium: ratio scale, forward, forced unit conversion ------------------

_RATIO_RE = re.compile(r"a scale of 1 : (\d+)\b")
_MEDIUM_DISTANCE_RE = re.compile(r"map is (\d+(?:\.\d+)?) cm\. What is the actual distance")
_TARGET_UNIT_RE = re.compile(r"in (km|m)(, to 1 decimal place)?\?")


def _oracle_medium(text: str) -> tuple[Fraction, int]:
    ratio_match = _RATIO_RE.search(text)
    distance_match = _MEDIUM_DISTANCE_RE.search(text)
    unit_match = _TARGET_UNIT_RE.search(text)
    assert ratio_match and distance_match and unit_match, f"could not parse: {text!r}"
    n = int(ratio_match.group(1))
    map_distance = Fraction(distance_match.group(1))
    unit = unit_match.group(1)
    places = 1 if unit_match.group(2) else 0

    real_cm = map_distance * n
    real_m = real_cm / 100
    real_value = real_m if unit == "m" else real_m / 1000
    return real_value, places


# --- hard: ratio scale, REVERSE ---------------------------------------------

_HARD_GIVEN_RE = re.compile(
    r"real-world distance between two towns is (\d+(?:\.\d+)?) (km|m)\. "
    r"What is the distance between the towns on the map, in cm"
)


def _oracle_hard(text: str) -> tuple[Fraction, int]:
    ratio_match = _RATIO_RE.search(text)
    given_match = _HARD_GIVEN_RE.search(text)
    assert ratio_match and given_match, f"could not parse: {text!r}"
    n = int(ratio_match.group(1))
    real_value = Fraction(given_match.group(1))
    unit = given_match.group(2)

    real_cm = real_value * (100 if unit == "m" else 100_000)
    assert real_cm.denominator == 1, f"{real_cm} is not a whole number of centimetres: {text!r}"
    map_distance = real_cm / n
    return map_distance, 0


_ORACLES = {
    "easy": _oracle_easy,
    "medium": _oracle_medium,
    "hard": _oracle_hard,
}


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_map_scale_distance_oracle_agrees(tier):
    oracle = _ORACLES[tier]
    checked = 0
    for seed in range(60):
        q = make(TOPIC, seed=seed, difficulty=tier, **PARAMS[tier])
        expected_value, places = oracle(q.text)
        expected_text = _format_exact(expected_value, places)
        assert expected_text == q.answer_text, (
            f"seed={seed} tier={tier}: oracle recomputed {expected_text} "
            f"from the prompt, generator's answer_text is {q.answer_text}\n{q.text}"
        )
        checked += 1
    assert checked == 60
