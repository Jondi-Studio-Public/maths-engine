"""Tests the multi-part exam source of `senior-calculate-float-times-for-non-critical-activities`.

The oracle parses the network out of the rendered stem and works it with a forward and a backward pass written
independently of the generator (memoised recursion, not a route list).
"""

from __future__ import annotations

import re
from functools import lru_cache

from mathgen.engine import grade
from tests.exd_x_r3c_support import (
    check_all_wrong,
    check_answer_space,
    check_deterministic,
    check_follow_through,
    check_full_marks_and_distractors,
    check_oracle,
    check_shapes,
    check_text,
    check_wired,
    exam_questions,
    skill_for,
)

TOPIC = "exd_x_r3c_network_float"
SKILL_ID = "senior-calculate-float-times-for-non-critical-activities"
SKILL = skill_for(SKILL_ID)
_ACT = re.compile(r"Activity ([A-Z]): duration (\d+) days, (?:no predecessors|predecessors: ([A-Z, ]+))\.")
_DELAY = re.compile(r"Activity ([A-Z]) takes (\d+) days longer")


def parse(q):
    dur, preds = {}, {}
    for name, d, p in _ACT.findall(q.stem):
        dur[name] = int(d)
        preds[name] = [x.strip() for x in p.split(",")] if p else []
    x, t = _DELAY.search(q.stem).groups()
    return preds, dur, x, int(t)


def finish_of(preds, dur):
    @lru_cache(None)
    def ef(a):
        return dur[a] + max((ef(p) for p in preds[a]), default=0)

    return max(ef(a) for a in preds)


def float_of(preds, dur, x):
    finish = finish_of(preds, dur)
    succ = {a: [b for b in preds if a in preds[b]] for a in preds}

    @lru_cache(None)
    def ef(a):
        return dur[a] + max((ef(p) for p in preds[a]), default=0)

    @lru_cache(None)
    def lf(a):
        return min((lf(s) - dur[s] for s in succ[a]), default=finish)

    return lf(x) - ef(x)


def oracle(q):
    preds, dur, x, t = parse(q)
    finish, flt = finish_of(preds, dur), float_of(preds, dur, x)
    late = dict(dur, **{x: dur[x] + t})
    new = finish_of(preds, late)
    return {"a": str(finish), "b": str(flt), "c": str(new - finish), "d": str(new)}


def test_source_wired():
    check_wired(SKILL, TOPIC)


def test_shapes():
    check_shapes(SKILL, TOPIC)


def test_text_quality():
    check_text(TOPIC)


def test_key_rederived_from_stem():
    check_oracle(TOPIC, oracle, n=300)


def test_full_marks_and_distractors_rejected():
    check_full_marks_and_distractors(SKILL, TOPIC, n=40)


def test_all_wrong_earns_nothing():
    check_all_wrong(SKILL, TOPIC)


def test_follow_through_from_a_wrong_completion_time():
    """A completion time that is one day short, used correctly in (b) and (c), earns both."""
    done = 0
    for q in exam_questions(TOPIC, 60):
        preds, dur, x, t = parse(q)
        finish, flt = finish_of(preds, dur), float_of(preds, dur, x)
        wrong = finish - 1
        b = wrong - (finish - flt)
        c = t - b
        if b <= 0 or c <= 0 or c in (wrong, b, t - flt):
            continue  # a non-positive value is outside the part's sensible range; repeats are not followed through
        got = grade(SKILL, q, {"a": str(wrong), "b": str(b), "c": str(c), "d": str(wrong + c)})
        assert got.parts["a"].marks == 0
        for label in "bc":
            assert got.parts[label].marks == got.parts[label].max_marks and got.parts[label].follow_through, (
                q.text,
                label,
            )
        assert got.parts["d"].marks == got.parts["d"].max_marks  # (d) is the same whatever the slip in (a)
        done += 1
    assert done >= 10


def test_follow_through_from_a_wrong_float():
    for q in exam_questions(TOPIC, 40):
        preds, dur, x, t = parse(q)
        finish = finish_of(preds, dur)
        wrong = int(q.parts[1].distractors[0])
        c = t - wrong
        if wrong <= 0 or c <= 0 or c == wrong:
            continue
        got = grade(SKILL, q, {"a": str(finish), "b": str(wrong), "c": str(c), "d": str(finish + c)})
        assert got.parts["b"].marks == 0
        assert got.parts["c"].marks == got.parts["c"].max_marks and got.parts["c"].follow_through


def test_answer_space():
    check_answer_space(TOPIC)


def test_parts_shape():
    for q in exam_questions(TOPIC, 30):
        assert [p.label for p in q.parts] == list("abcd")
        assert [p.marks for p in q.parts] == [2, 2, 1, 1]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_deterministic():
    check_deterministic(TOPIC)
