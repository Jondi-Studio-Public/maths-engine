"""Derives every answer from `question.text`/`question.answer_text` alone, with its OWN regex parsing and its OWN
re-typed reference facts. Never imports `mathgen.generators.p9b_timezone_australia_build` -- if this file
needed that module to work, the question would be under-specified, which is itself the finding.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make

TZ_TOPIC = "australia_timezone_conversion"
CALL_TOPIC = "timezone_call_window_overlap"
SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]

PLACE_RE = r"(Perth|Darwin|Brisbane|Port Moresby|Jakarta|Adelaide|Sydney|Melbourne|Hobart|Canberra|Auckland)"


def _parse_offset(text: str) -> Fraction:
    """ "8:00" -> 8, "9:30" -> 9.5, and the abbreviation key's own bare
    "10" (no ":00") -> 10."""
    if ":" not in text:
        return Fraction(int(text))
    hours, minutes = text.split(":")
    return Fraction(int(hours)) + Fraction(int(minutes), 60)


# ============================================================================
# Skill 1: australia_timezone_conversion
# ============================================================================

_TZ_PATTERN = re.compile(
    rf"^{PLACE_RE}'s standard time zone offset is UTC\+([\d:]+)\. "
    rf"\1 (is currently observing daylight saving|does not observe daylight saving)\. "
    rf"{PLACE_RE}'s standard time zone offset is UTC\+([\d:]+)\. "
    rf"\4 (is currently observing daylight saving|does not observe daylight saving)\. "
    rf"Daylight saving advances local clocks by 1 hour\. "
)


def _oracle_tz_answer(text: str) -> Fraction:
    match = _TZ_PATTERN.match(text)
    if not match:
        raise ValueError(f"could not parse tz question: {text!r}")
    off_a = _parse_offset(match.group(2))
    dst_a = match.group(3) == "is currently observing daylight saving"
    off_b = _parse_offset(match.group(5))
    dst_b = match.group(6) == "is currently observing daylight saving"
    eff_a = off_a + (1 if dst_a else 0)
    eff_b = off_b + (1 if dst_b else 0)
    return abs(eff_b - eff_a)


@pytest.mark.parametrize("tier", TIERS)
def test_tz_oracle_matches_answer_and_rejects_distractors(tier):
    for seed in SEEDS:
        question = make(TZ_TOPIC, seed=seed, difficulty=tier)
        expected = _oracle_tz_answer(question.text)
        actual = Fraction(question.answer_text)
        assert expected == actual, f"seed {seed} {tier}: oracle {expected} != answer_text {actual}\n{question.text}"
        for distractor in question.distractors:
            assert Fraction(distractor) != expected, (
                f"seed {seed} {tier}: distractor {distractor} equals the oracle answer\n{question.text}"
            )


# ============================================================================
# Skill 2: timezone_call_window_overlap
# ============================================================================

_WINDOW_TO_BOUNDS = {
    "8 am to 10 pm": (8, 22),
    "9 am to 9 pm": (9, 21),
}

_DIRECT_OFFSET_RE = r"Place [AB] \({place}\) is at UTC\+([\d:]+)\."
_DST_OFFSET_RE = (
    r"Place [AB] \({place}\)'s standard time zone offset is UTC\+([\d:]+)\. "
    r"{place} is currently observing daylight saving; daylight saving advances local clocks by 1 hour\."
)
_ABBREV_LOOKUP_RE = r"Place [AB] \({place}\)'s time zone is (\w+)\."
_WINDOW_RE = r"Place [AB] \({place}\)'s reasonable hours to call are ([^,]+), local time\."
_KEY_RE = re.compile(r"(\w+) = \+([\d:]+)")


def _oracle_place_offset(text: str, place: str, key_values: dict) -> Fraction:
    m = re.search(_DIRECT_OFFSET_RE.format(place=re.escape(place)), text)
    if m:
        return _parse_offset(m.group(1))
    m = re.search(_DST_OFFSET_RE.format(place=re.escape(place)), text)
    if m:
        return _parse_offset(m.group(1)) + 1
    m = re.search(_ABBREV_LOOKUP_RE.format(place=re.escape(place)), text)
    if m:
        return key_values[m.group(1)]
    raise ValueError(f"could not find an offset for {place!r} in: {text!r}")


def _oracle_call_answer(text: str) -> Fraction:
    place_a = re.search(r"Place A \((.+?)\)", text).group(1)
    place_b = re.search(r"Place B \((.+?)\)", text).group(1)

    key_values: dict[str, Fraction] = {}
    key_match = re.search(r"Zone abbreviation key: (.+?)\.", text)
    if key_match:
        for abbrev, value in _KEY_RE.findall(key_match.group(1)):
            key_values[abbrev] = _parse_offset(value)

    off_a = _oracle_place_offset(text, place_a, key_values)
    off_b = _oracle_place_offset(text, place_b, key_values)

    window_a = re.search(_WINDOW_RE.format(place=re.escape(place_a)), text).group(1)
    window_b = re.search(_WINDOW_RE.format(place=re.escape(place_b)), text).group(1)
    a1, a2 = _WINDOW_TO_BOUNDS[window_a]
    b1, b2 = _WINDOW_TO_BOUNDS[window_b]

    d = off_b - off_a
    assert abs(d) <= 6, f"oracle: |d|={abs(d)} exceeds the real-world bound of 6"
    lo = max(Fraction(a1), Fraction(b1) + d)
    hi = min(Fraction(a2), Fraction(b2) + d)
    return max(Fraction(0), hi - lo)


@pytest.mark.parametrize("tier", TIERS)
def test_call_oracle_matches_answer_and_rejects_distractors(tier):
    for seed in SEEDS:
        question = make(CALL_TOPIC, seed=seed, difficulty=tier)
        expected = _oracle_call_answer(question.text)
        actual = Fraction(question.answer_text)
        assert expected == actual, f"seed {seed} {tier}: oracle {expected} != answer_text {actual}\n{question.text}"
        for distractor in question.distractors:
            assert Fraction(distractor) != expected, (
                f"seed {seed} {tier}: distractor {distractor} equals the oracle answer\n{question.text}"
            )
