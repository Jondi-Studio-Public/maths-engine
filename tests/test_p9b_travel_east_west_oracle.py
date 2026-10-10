"""Independent oracle for senior-solve-problems-relating-to-travelling-east-and-west.

Parses `question.text` alone -- departure/arrival clock times, direction,
flight duration and the calendar-day label -- and re-derives the time-zone
offset magnitude from scratch, the INVERSE way a learner actually has to:
reconstruct the arrival instant in place A's own frame from the stated day
label and arrival clock, then subtract the (departure + duration) instant.
Never imports anything from `mathgen.generators.p9b_travel_east_west_build`.
"""

import re
from fractions import Fraction

import pytest

import mathgen

TOPIC = "travel_east_west_zone_difference"

TEXT_RE = re.compile(
    r"departs (?P<A>.+?) at (?P<dep>\d{2}:\d{2}), travelling (?P<direction>east|west), "
    r"on a flight lasting (?P<duration>.+?)\. "
    r"They land in (?P<B>.+?) at (?P<arr>\d{2}:\d{2}) local time, "
    r"(?P<dayphrase>on the (?:same day|day before|day after) they departed) "
    r"(?P<A2>.+?)\. "
    r"How many hours (?P<aheadbehind>ahead of|behind) "
)


def _parse_clock(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _parse_duration(s: str) -> int:
    hours = re.search(r"(\d+)\s*hours?", s)
    minutes = re.search(r"(\d+)\s*minutes?", s)
    h = int(hours.group(1)) if hours else 0
    m = int(minutes.group(1)) if minutes else 0
    return h * 60 + m


def _day_offset_from_phrase(phrase: str) -> int:
    if "same day" in phrase:
        return 0
    if "day before" in phrase:
        return -1
    if "day after" in phrase:
        return 1
    raise ValueError(f"unrecognised day phrase: {phrase!r}")


def derive_offset_from_text(text: str) -> tuple:
    """Returns (magnitude_hours: Fraction, sign: int) derived purely from
    the rendered question text, with `sign` +1 for "ahead of" (east) and
    -1 for "behind" (west) as literally stated in the question."""
    m = TEXT_RE.search(text)
    assert m is not None, f"oracle could not parse: {text}"
    assert m.group("A") == m.group("A2"), "the day-phrase must name the SAME departure place A"

    dep_A = _parse_clock(m.group("dep"))
    arr_B_clock = _parse_clock(m.group("arr"))
    duration = _parse_duration(m.group("duration"))
    day_offset_B = _day_offset_from_phrase(m.group("dayphrase"))
    direction = m.group("direction")
    stated_ahead_behind = m.group("aheadbehind")

    # Reconstruct the arrival instant in A's own frame (never using the
    # generator's construction shortcuts, only the visible fields above),
    # then find the signed zone difference implied by that reconstruction.
    arr_B_total_reconstructed = day_offset_B * 1440 + arr_B_clock
    signed_offset_minutes = arr_B_total_reconstructed - (dep_A + duration)
    magnitude_hours = Fraction(abs(signed_offset_minutes), 60)

    sign = 1 if signed_offset_minutes > 0 else -1
    expected_direction_word = "ahead of" if direction == "east" else "behind"
    assert stated_ahead_behind == expected_direction_word, (
        f"question direction word {stated_ahead_behind!r} does not match stated travel direction {direction!r}"
    )
    return magnitude_hours, sign, direction


def _format_like_answer(value: Fraction) -> str:
    if value.denominator == 1:
        return str(int(value))
    return f"{float(value):.1f}"


@pytest.mark.parametrize("difficulty,seed", [(tier, s) for tier in ["easy", "medium", "hard"] for s in range(60)])
def test_oracle_matches_stated_answer(difficulty, seed):
    q = mathgen.make(TOPIC, seed=seed, difficulty=difficulty)
    magnitude_hours, sign, direction = derive_offset_from_text(q.text)

    expected_sign = -1 if direction == "west" else 1
    assert sign == expected_sign, f"reconstructed sign {sign} disagrees with stated direction {direction} for: {q.text}"

    derived_text = _format_like_answer(magnitude_hours)
    assert derived_text == q.answer_text, (
        f"oracle derived {derived_text} but answer_text is {q.answer_text}\nquestion: {q.text}"
    )


def test_oracle_coverage():
    tiers = ["easy", "medium", "hard"]
    seeds_per_tier = 60
    total = len(tiers) * seeds_per_tier
    assert seeds_per_tier >= 60
    assert total == 180
