"""Derives the expected answer PURELY from `question.text` -- parsing the
stated latitudes/longitudes and the stated rule/constants with its own
regexes. Does NOT import `mathgen.generators.p9b_earth_distance_time_build`.

Oracle route, skill 1: parse each latitude as either "D.D°H" (decimal) or
"D°MM'H" (degrees+minutes); same-hemisphere pairs subtract, opposite
hemispheres add; the result in arcminutes is the nautical-mile distance
directly (1 arcminute of latitude = 1 nautical mile, stated in the text);
if the question asks for kilometres, multiply by the stated "1 nautical
mile = 1.852 km exactly" and round to the nearest whole kilometre.

Oracle route, skill 2 (easy/medium): parse both longitudes as signed
degrees (E positive, W negative) from their own "°E"/"°W" suffix; same-side
pairs (both E or both W) subtract, opposite sides add; divide by the stated
"15° of longitude = 1 hour" rule and round to 2 d.p.

Oracle route, skill 2 (hard/inverse): parse place A's longitude, the
stated Δt (hours) and the stated direction word; longitude of place B is
place A's longitude plus (east) or minus (west) Δt × 15, to 1 d.p.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

SEEDS = range(60)

NM_PER_KM = Fraction(1852, 1000)


def _round_half_up(value: Fraction, dp: int = 0) -> Fraction:
    scale = 10**dp
    scaled = value * scale + Fraction(1, 2)
    return Fraction(scaled.numerator // scaled.denominator, scale)


# --- skill 1: earth_meridian_distance ----------------------------------------

_DECIMAL_LAT_RE = re.compile(r"latitude (\d+\.\d+)°([NS])")
_DM_LAT_RE = re.compile(r"latitude (\d+)°(\d+)′([NS])")


def _oracle_earth_meridian_distance(text: str) -> Fraction:
    dm_matches = _DM_LAT_RE.findall(text)
    if dm_matches:
        assert len(dm_matches) == 2, f"expected 2 degrees+minutes latitudes: {text}"
        (deg_a, min_a, hemi_a), (deg_b, min_b, hemi_b) = dm_matches
        deg_a, min_a, deg_b, min_b = int(deg_a), int(min_a), int(deg_b), int(min_b)
    else:
        decimal_matches = _DECIMAL_LAT_RE.findall(text)
        assert len(decimal_matches) == 2, f"expected 2 decimal latitudes: {text}"
        (dec_a, hemi_a), (dec_b, hemi_b) = decimal_matches
        dec_a, dec_b = Fraction(dec_a), Fraction(dec_b)
        deg_a, min_a = int(dec_a), int(round((dec_a - int(dec_a)) * 60))
        deg_b, min_b = int(dec_b), int(round((dec_b - int(dec_b)) * 60))

    same_hemi = hemi_a == hemi_b
    arcmin_a = deg_a * 60 + min_a
    arcmin_b = deg_b * 60 + min_b
    nm = abs(arcmin_a - arcmin_b) if same_hemi else arcmin_a + arcmin_b

    if "kilometres" in text:
        conv_match = re.search(r"1 nautical mile = ([\d.]+) km exactly", text)
        assert conv_match, f"question asks for km but never states the nm->km factor: {text}"
        factor = Fraction(conv_match.group(1))
        assert factor == NM_PER_KM, f"stated factor {factor} != expected {NM_PER_KM}"
        return _round_half_up(Fraction(nm) * factor, 0)
    assert "nautical miles" in text, f"unrecognised units: {text}"
    return Fraction(nm)


@pytest.mark.parametrize("mode", ["same_hemisphere", "opposite_hemisphere", "combined_km"])
@pytest.mark.parametrize("seed", SEEDS)
def test_earth_meridian_distance_oracle(seed, mode):
    difficulty = {"same_hemisphere": "easy", "opposite_hemisphere": "medium", "combined_km": "hard"}[mode]
    question = make("earth_meridian_distance", seed=seed, difficulty=difficulty, mode=mode)
    expected = _oracle_earth_meridian_distance(question.text)
    actual = Fraction(question.answer_text)
    assert actual == expected, (
        f"seed {seed}, mode {mode}: oracle expected {expected}, got {actual} from text: {question.text}"
    )
    for distractor in question.distractors:
        assert Fraction(distractor) != expected, (
            f"seed {seed}, mode {mode}: distractor {distractor} equals oracle answer {expected}"
        )


# --- skill 2: earth_longitude_time_difference --------------------------------

_LON_RE = re.compile(r"longitude ([\d.]+)°([EW])")
_DT_STATED_RE = re.compile(
    r"place B is ([\d.]+) hours? different in solar time from \S+.*?"
    r"further (east|west)",
    re.DOTALL,
)


def _signed(value_str: str, hemi: str) -> Fraction:
    value = Fraction(value_str)
    return value if hemi == "E" else -value


def _oracle_forward(text: str) -> Fraction:
    matches = _LON_RE.findall(text)
    assert len(matches) == 2, f"expected 2 longitudes: {text}"
    (val_a, hemi_a), (val_b, hemi_b) = matches
    lon_a, lon_b = _signed(val_a, hemi_a), _signed(val_b, hemi_b)

    rule = "15° of longitude corresponds to exactly 1 hour of solar time"
    assert rule in text, f"question never states the solar-time rule: {text}"

    same_side = (lon_a >= 0) == (lon_b >= 0)
    delta = abs(lon_a - lon_b) if same_side else abs(lon_a) + abs(lon_b)
    return _round_half_up(delta / 15, 2)


def _oracle_inverse(text: str) -> Fraction:
    matches = _LON_RE.findall(text)
    assert len(matches) == 1, f"expected exactly 1 stated longitude (place A): {text}"
    val_a, hemi_a = matches[0]
    lon_a = _signed(val_a, hemi_a)

    dt_match = _DT_STATED_RE.search(text)
    assert dt_match, f"could not parse the stated Δt/direction: {text}"
    dt = Fraction(dt_match.group(1))
    direction = dt_match.group(2)

    mag = dt * 15
    return lon_a + mag if direction == "east" else lon_a - mag


@pytest.mark.parametrize("mode", ["same_side", "opposite_side"])
@pytest.mark.parametrize("seed", SEEDS)
def test_earth_longitude_time_difference_forward_oracle(seed, mode):
    difficulty = {"same_side": "easy", "opposite_side": "medium"}[mode]
    question = make("earth_longitude_time_difference", seed=seed, difficulty=difficulty, mode=mode)
    expected = _oracle_forward(question.text)
    actual = Fraction(question.answer_text)
    assert actual == expected, (
        f"seed {seed}, mode {mode}: oracle expected {expected}, got {actual} from text: {question.text}"
    )
    for distractor in question.distractors:
        assert Fraction(distractor) != expected, (
            f"seed {seed}, mode {mode}: distractor {distractor} equals oracle answer {expected}"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_earth_longitude_time_difference_inverse_oracle(seed):
    question = make("earth_longitude_time_difference", seed=seed, difficulty="hard", mode="inverse")
    expected = _oracle_inverse(question.text)
    actual = Fraction(question.answer_text)
    assert actual == expected, f"seed {seed}: oracle expected {expected}, got {actual} from text: {question.text}"
    for distractor in question.distractors:
        assert Fraction(distractor) != expected, f"seed {seed}: distractor {distractor} equals oracle answer {expected}"
