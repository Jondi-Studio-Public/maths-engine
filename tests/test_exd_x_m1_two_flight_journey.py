"""Tests the exam source of `f10-circle-and-prism-duration-rates`: a two-flight journey across three clocks."""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_m1_two_flight_journey"
SKILL_ID = "f10-circle-and-prism-duration-rates"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_TIME_RE = re.compile(r"at (\d\d):(\d\d) on (\w+day)")
_DIFF_RE = re.compile(r"time is (\d+) hours? (ahead of|behind) ")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _local_gap(first, second) -> int:
    """Minutes from one printed (hour, minute, day) to the next on the SAME clock; the day gap is within 3 days."""
    (h1, m1, d1), (h2, m2, d2) = first, second
    days = ((_DAYS.index(d2) - _DAYS.index(d1) + 3) % 7) - 3
    return days * 1440 + (h2 * 60 + m2) - (h1 * 60 + m1)


def _oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the rendered prompt alone."""
    times = [(int(h), int(m), day) for h, m, day in _TIME_RE.findall(q.stem)]
    dep1, arr1, dep2, arr2 = times
    shifts = [(1 if word == "ahead of" else -1) * 60 * int(n) for n, word in _DIFF_RE.findall(q.stem)]
    shift1, shift2 = shifts
    a = _local_gap(dep1, arr1) - shift1
    b = _local_gap(arr1, dep2)
    c = _local_gap(dep2, arr2) - shift2
    total_direct = _local_gap(dep1, arr2) - shift1 - shift2
    assert total_direct == a + b + c
    return {"a": str(a), "b": str(b), "c": str(c), "d": str(a + b + c)}


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape():
    q = _exam(1)
    assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
    assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        assert _oracle(q) == _canonical(q), (seed, q.text)


def test_prompt_does_not_contain_a_key():
    for seed in range(300):
        q = _exam(seed)
        numbers = set(re.findall(r"\d+", q.stem))
        assert not numbers & {p.answer_text for p in q.parts}, (seed, q.text)


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(120)]


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6


def test_distractors_rejected(questions):
    shown = 0
    for q in questions:
        canon = _canonical(q)
        for p in q.parts:
            assert p.distractors, (q.text, p.label)
            for d in p.distractors:
                shown += 1
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (q.text, p.label, d)
    assert shown >= 300


def test_follow_through_into_total(questions):
    """Wrong flying times used correctly in the total still earn the total."""
    for q in questions:
        canon = _canonical(q)
        a_wrong = int(q.parts[0].distractors[0])
        total = a_wrong + int(canon["b"]) + int(canon["c"])
        result = grade(_SKILL, q, {**canon, "a": str(a_wrong), "d": str(total)})
        assert result.parts["a"].marks == 0
        assert result.parts["d"].marks == result.parts["d"].max_marks, (q.text, a_wrong, total)
        assert result.parts["d"].follow_through is True
        assert not result.correct


def test_total_without_stopover_earns_nothing(questions):
    for q in questions:
        canon = _canonical(q)
        left_out = str(int(canon["a"]) + int(canon["c"]))
        assert grade(_SKILL, q, {**canon, "d": left_out}).parts["d"].marks == 0
