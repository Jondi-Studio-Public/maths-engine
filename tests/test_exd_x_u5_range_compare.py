"""Exam source of `calculate-range`: the range of each of two rainfall data sets, then how much greater one is."""

from __future__ import annotations

import re

from tests.exd_x_u5_support import (
    canonical_reply,
    check_key_and_distractors,
    check_no_text_defects,
    check_shapes,
    exam_question,
    exam_sources,
    skill_of,
)

from mathgen.engine import grade

SKILL = "calculate-range"


def read(q):
    stem = q.text.split(" (a)")[0]
    first, second = (list(map(int, run.split(", "))) for run in re.findall(r"\d+(?:, \d+){5}", stem))
    return first, second


def oracle(q):
    first, second = read(q)
    ra, rb = max(first) - min(first), max(second) - min(second)
    return ra, rb, ra - rb


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam_question(SKILL, seed)
        assert tuple(int(p.answer_text) for p in q.parts) == oracle(q), (seed, q.text)


def test_grader_accepts_key_rejects_distractors():
    check_key_and_distractors(SKILL)


def test_follow_through_from_wrong_range():
    skill = skill_of(SKILL)
    checked = 0
    for seed in range(100):
        q = exam_question(SKILL, seed)
        wrong_a = int(q.parts[0].distractors[0])
        _, rb, _ = oracle(q)
        carried = wrong_a - rb
        if carried <= 0 or carried == wrong_a or carried == rb:
            continue
        checked += 1
        result = grade(skill, q, {**canonical_reply(q), "a": str(wrong_a), "c": str(carried)})
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through, (seed, wrong_a)
    assert checked >= 30


def test_all_wrong_earns_nothing():
    q = exam_question(SKILL, 1)
    assert grade(skill_of(SKILL), q, {"a": "9999", "b": "9998", "c": "9997"}).marks == 0
