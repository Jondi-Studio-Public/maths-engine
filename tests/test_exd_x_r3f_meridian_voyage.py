"""Exam chain of `senior-find-distances-between-two-places-on-earth-on`: nautical miles, hours, km and fuel."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from exd_x_r3f_support import (
    assert_all_wrong_earns_nothing,
    assert_answer_space,
    assert_clean_and_deterministic,
    assert_follow_through,
    assert_full_marks_and_distractors,
    assert_shapes,
    assert_wired,
    exam_question,
)

TOPIC = "exd_x_r3f_meridian_voyage"
SKILL_ID = "senior-find-distances-between-two-places-on-earth-on"
_LAT = re.compile(r"(\d+)°(\d\d)′ ([NS])")


def _arcminutes(match) -> int:
    value = int(match[0]) * 60 + int(match[1])
    return value if match[2] == "N" else -value


def read_question(q):
    lat1, lat2 = (_arcminutes(m) for m in _LAT.findall(q.stem))
    speed = int(re.search(r"(\d+) knots", q.stem).group(1))
    rate = int(re.search(r"(\d+) litres", q.stem).group(1))
    direction = re.search(r"due (north|south)", q.stem).group(1)
    return lat1, lat2, speed, rate, direction


def oracle(q):
    lat1, lat2, speed, rate, direction = read_question(q)
    assert (lat2 > lat1) == (direction == "north")
    nm = abs(lat2 - lat1)
    hours = Fraction(nm, speed)
    km = Fraction(nm) * Fraction(1852, 1000)
    return nm, hours, int(km + Fraction(1, 2)), hours * rate


@pytest.fixture(scope="module")
def questions():
    return [exam_question(TOPIC, s) for s in range(300)]


def test_source_wired():
    assert_wired(SKILL_ID, TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 1, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompts_and_determinism():
    assert_shapes(SKILL_ID, TOPIC)
    assert_clean_and_deterministic(TOPIC)


def test_key_rederived_from_rendered_prompt(questions):
    crossings = 0
    for q in questions:
        nm, hours, km, fuel = oracle(q)
        lat1, lat2, *_ = read_question(q)
        crossings += (lat1 > 0) != (lat2 > 0)
        assert [p.answer_text for p in q.parts] == [str(nm), f"{float(hours):.1f}", str(km), str(int(fuel))], q.text
        assert fuel.denominator == 1
    assert 80 <= crossings <= 220  # about half the voyages cross the equator


def test_decimal_place_form_of_every_option(questions):
    for q in questions[:100]:
        for part, dp in zip(q.parts, (0, 1, 0, 0)):
            for text in (part.answer_text, *part.distractors):
                assert (len(text.split(".")[1]) if "." in text else 0) == dp, (part.label, text)


def test_grader_accepts_key_and_rejects_distractors():
    assert_full_marks_and_distractors(SKILL_ID, TOPIC, min_options=2)


def test_answer_space():
    assert_answer_space(TOPIC)


def _wrong_distance_replies(q):
    _lat1, _lat2, speed, rate, _d = read_question(q)
    nm_wrong = int(q.parts[0].distractors[0])
    hours = Fraction(nm_wrong, speed)
    hours_text = f"{float(hours):.1f}"
    if (hours * 10) % 1 == Fraction(1, 2):
        return None  # a tie, where half-up and half-even rounding disagree
    km = int(Fraction(nm_wrong) * Fraction(1852, 1000) + Fraction(1, 2))
    fuel = Fraction(hours_text) * rate
    if fuel.denominator != 1:
        return None
    return {"a": str(nm_wrong), "b": hours_text, "c": str(km), "d": str(int(fuel))}


def test_follow_through_from_a_wrong_distance():
    assert_follow_through(SKILL_ID, TOPIC, _wrong_distance_replies, ["a"], ["b", "c", "d"], seeds=range(300))


def test_rounded_time_then_fuel_matches_the_key(questions):
    """The time is exact to one decimal place, so the fuel from the rounded time is the key."""
    for q in questions:
        hours = Fraction(q.parts[1].answer_text)
        rate = int(re.search(r"(\d+) litres", q.stem).group(1))
        assert hours * rate == Fraction(q.parts[3].answer_text)


def test_all_wrong_earns_nothing():
    assert_all_wrong_earns_nothing(SKILL_ID, TOPIC, {"a": "99991", "b": "992.2", "c": "99993", "d": "99994"})
