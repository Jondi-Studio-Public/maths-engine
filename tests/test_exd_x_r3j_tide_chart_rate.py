"""Exam source of `senior-interpret-complex-timetables-such-as-tide-charts-sunrise`: rate of rise from a tide chart."""

from __future__ import annotations

import re
from fractions import Fraction

from exd_r3j_support import (
    canonical,
    check_answer_space,
    check_deterministic,
    check_distractors,
    check_full_marks,
    check_shapes,
    check_source_wired,
    skill,
)

from mathgen import make
from mathgen.engine import grade

TOPIC = "exd_x_r3j_tide_chart_rate"
SKILL_ID = "senior-interpret-complex-timetables-such-as-tide-charts-sunrise"


def _half_up(x: Fraction) -> int:
    return int((x + Fraction(1, 2)).__floor__())


def _minutes(clock: str) -> int:
    hours, minutes = clock.split(":")
    return int(hours) * 60 + int(minutes)


def _oracle(q):
    tides = re.findall(r"(Low|High) tide at (\d\d:\d\d), height (\d+\.\d\d) m", q.stem)
    (kind1, t1, h1), (kind2, t2, h2) = tides[0], tides[1]
    assert (kind1, kind2) == ("Low", "High")
    duration = _minutes(t2) - _minutes(t1)
    rise = (Fraction(h2) - Fraction(h1)) * 100
    rate = rise * 60 / duration
    # the ramp level is the last height quoted in the stem
    level = Fraction(re.findall(r"(\d+\.\d\d) m", q.stem)[-1])
    minutes = (level - Fraction(h1)) * 100 / rate * 60
    assert rise.denominator == 1 and rate.denominator == 1 and minutes.denominator == 1
    return {"a": str(duration), "b": str(int(rise)), "c": str(int(rate)), "d": str(int(minutes))}


def test_source_wired():
    check_source_wired(SKILL_ID, TOPIC)


def test_shape_and_marks():
    for s in range(60):
        q = make(TOPIC, seed=s)
        assert [p.marks for p in q.parts] == [1, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompt_shapes_new_against_hard_tier():
    check_shapes(TOPIC, "p19i_timetable_condition_lookup", {"difficulty": "hard"})


def test_key_rederived_from_prompt():
    for s in range(300):
        q = make(TOPIC, seed=s)
        assert _oracle(q) == canonical(q), (s, q.text)


def test_chart_is_plausible():
    for s in range(300):
        q = make(TOPIC, seed=s)
        times = [_minutes(t) for t in re.findall(r"tide at (\d\d:\d\d)", q.stem)]
        assert times == sorted(times) and times[-1] < 24 * 60
        assert 300 <= times[1] - times[0] <= 420


def test_full_marks_and_distractors_rejected():
    check_full_marks(SKILL_ID, TOPIC)
    check_distractors(SKILL_ID, TOPIC)


def test_answer_space():
    check_answer_space(TOPIC)


def test_follow_through_from_wrong_duration():
    """A wrong duration carries into the rate and then into the ramp time."""
    sk = skill(SKILL_ID)
    done = 0
    for s in range(100):
        q = make(TOPIC, seed=s)
        canon = canonical(q)
        duration, rise = int(canon["a"]), int(canon["b"])
        ramp_rise = int(q.parts[3].follow_through.split("*")[0])
        wrong = duration + 30
        rate = _half_up(Fraction(rise * 60, wrong))
        ramp = _half_up(Fraction(ramp_rise * 60, rate))
        reply = {**canon, "a": str(wrong), "c": str(rate), "d": str(ramp)}
        if len({reply["a"], reply["b"], reply["c"], reply["d"]}) < 4 or rate == int(canon["c"]):
            continue
        result = grade(sk, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (s, reply)
        if reply["d"] != canon["d"]:
            assert result.parts["d"].marks == 2 and result.parts["d"].follow_through, (s, reply)
        done += 1
    assert done >= 50


def test_all_wrong_earns_nothing():
    q = make(TOPIC, seed=3)
    assert grade(skill(SKILL_ID), q, {"a": "11", "b": "12", "c": "13", "d": "14"}).marks == 0


def test_deterministic():
    check_deterministic(TOPIC)
