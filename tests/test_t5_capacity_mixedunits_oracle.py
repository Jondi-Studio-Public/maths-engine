"""Independent oracle for the `t5_capacity_mixedunits` questions: it re-derives each answer from the rendered
question text.

Every value in this pair of skills is an exact rational (no pi, no irrational factor), so there is never a
rounding disagreement to tolerate -- a mismatch here is a defect, full stop.

Capacity route: convert all three dimensions to CENTIMETRES, multiply, and
convert to the requested unit via the exact ratios 1 L = 1000 cm^3 and
1 m^3 = 1000 L (both domain facts the oracle is
allowed to know). Rectangle route: convert both sides to MILLIMETRES,
compute area/perimeter, and convert to the requested target unit.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

# --- shared, oracle-local exact-decimal formatting (deliberately NOT the
# generator's own `_format_exact`/`_fixed_string` -- an independent
# re-implementation, so a bug shared by both would be a coincidence, not a
# structural blind spot). ---------------------------------------------------


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


_PRECISION_RE = re.compile(r"to (\d+) decimal place")
_WHOLE_NUMBER_RE = re.compile(r"whole number")


def _requested_places(text: str) -> int:
    m = _PRECISION_RE.search(text)
    if m:
        return int(m.group(1))
    assert _WHOLE_NUMBER_RE.search(text), f"no precision statement found in: {text!r}"
    return 0


# ===========================================================================
# Capacity oracle (prism-capacity-metric-units / prism_capacity_conversion)
# ===========================================================================

_DIM_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(mm|cm|m)\s+(?:long|wide|high)\b")
_CAPACITY_TARGET_RE = re.compile(r"capacity in (litres \(L\)|cubic metres \(m³\))")

_TO_CM = {"cm": Fraction(1), "mm": Fraction(1, 10), "m": Fraction(100)}


def _capacity_factor(label: str) -> Fraction:
    if label.startswith("litres"):
        return Fraction(1000)  # 1 L = 1000 cm^3
    return Fraction(1_000_000)  # 1 m^3 = 1000 L = 1,000,000 cm^3


def _oracle_capacity(text: str) -> tuple[Fraction, int]:
    dims = _DIM_RE.findall(text)
    assert len(dims) == 3, f"expected 3 dimensions, found {len(dims)} in: {text!r}"
    volume_cm3 = Fraction(1)
    for value, unit in dims:
        volume_cm3 *= Fraction(value) * _TO_CM[unit]
    target_match = _CAPACITY_TARGET_RE.search(text)
    assert target_match, f"no capacity target unit found in: {text!r}"
    factor = _capacity_factor(target_match.group(1))
    places = _requested_places(text)
    return volume_cm3 / factor, places


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_capacity_oracle_agrees(tier):
    checked = 0
    for seed in range(60):
        q = make("prism_capacity_conversion", seed=seed, difficulty=tier, tier=tier)
        expected_value, places = _oracle_capacity(q.text)
        expected_text = _format_exact(expected_value, places)
        assert expected_text == q.answer_text, (
            f"seed={seed} tier={tier}: oracle recomputed {expected_text} "
            f"from the prompt, generator's answer_text is {q.answer_text}\n{q.text}"
        )
        checked += 1
    assert checked == 60


# ===========================================================================
# Rectangle oracle (rectangle-area-perimeter-mixed-units /
# rectangle_mixed_unit_area_perimeter)
# ===========================================================================

_SIDE1_RE = re.compile(r"one side of (-?\d+(?:\.\d+)?)\s*(mm|cm|m)\b")
_SIDE2_RE = re.compile(r"other side of (-?\d+(?:\.\d+)?)\s*(mm|cm|m)\b")
_QUANTITY_RE = re.compile(r"its (area|perimeter)")
_TARGET_LABEL_RE = re.compile(
    r"in (millimetres \(mm\)|centimetres \(cm\)|metres \(m\)|"
    r"square millimetres \(mm²\)|square centimetres \(cm²\)|square metres \(m²\))"
)

_TO_MM = {"mm": Fraction(1), "cm": Fraction(10), "m": Fraction(1000)}
_LABEL_TO_UNIT = {
    "millimetres (mm)": "mm",
    "centimetres (cm)": "cm",
    "metres (m)": "m",
    "square millimetres (mm²)": "mm",
    "square centimetres (cm²)": "cm",
    "square metres (m²)": "m",
}


def _oracle_rectangle(text: str) -> tuple[Fraction, int]:
    s1 = _SIDE1_RE.search(text)
    s2 = _SIDE2_RE.search(text)
    assert s1 and s2, f"could not find both sides in: {text!r}"
    v1_mm = Fraction(s1.group(1)) * _TO_MM[s1.group(2)]
    v2_mm = Fraction(s2.group(1)) * _TO_MM[s2.group(2)]

    quantity_match = _QUANTITY_RE.search(text)
    assert quantity_match, f"could not find area/perimeter in: {text!r}"
    quantity = quantity_match.group(1)

    label_match = _TARGET_LABEL_RE.search(text)
    assert label_match, f"could not find a target unit label in: {text!r}"
    target_unit = _LABEL_TO_UNIT[label_match.group(1)]
    target_factor_mm = _TO_MM[target_unit]

    if quantity == "perimeter":
        raw_mm = 2 * (v1_mm + v2_mm)
        value = raw_mm / target_factor_mm
    else:
        raw_mm2 = v1_mm * v2_mm
        value = raw_mm2 / (target_factor_mm**2)

    places = _requested_places(text)
    return value, places


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_rectangle_oracle_agrees(tier):
    checked = 0
    for seed in range(60):
        q = make("rectangle_mixed_unit_area_perimeter", seed=seed, difficulty=tier, tier=tier)
        expected_value, places = _oracle_rectangle(q.text)
        expected_text = _format_exact(expected_value, places)
        assert expected_text == q.answer_text, (
            f"seed={seed} tier={tier}: oracle recomputed {expected_text} "
            f"from the prompt, generator's answer_text is {q.answer_text}\n{q.text}"
        )
        checked += 1
    assert checked == 60
