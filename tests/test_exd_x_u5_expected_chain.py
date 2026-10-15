"""Exam source of `expected-frequency`: a probability from counts, then expected frequencies for two outcomes."""

from __future__ import annotations

import re
from fractions import Fraction

from tests.exd_x_u5_support import (
    as_fraction,
    canonical_reply,
    check_key_and_distractors,
    check_no_text_defects,
    check_shapes,
    exam_question,
    exam_sources,
    skill_of,
)

from mathgen.engine import grade

SKILL = "expected-frequency"


def read(q):
    stem = q.text.split(" (a)")[0]
    r, b, g = map(int, re.search(r"(\d+) red, (\d+) blue and (\d+) green", stem).groups())
    trials = int(re.search(r"(\d+) times", stem).group(1))
    return r, b, g, trials


def oracle(q):
    r, b, g, trials = read(q)
    p = Fraction(r + b, r + b + g)
    expected = p * trials
    return p, expected, Fraction(g, r + b + g) * trials


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = exam_question(SKILL, seed)
        assert tuple(as_fraction(p.answer_text) for p in q.parts) == oracle(q), (seed, q.text)


def test_grader_accepts_key_rejects_distractors():
    check_key_and_distractors(SKILL)


def test_follow_through_from_wrong_probability():
    skill = skill_of(SKILL)
    checked = 0
    for seed in range(100):
        q = exam_question(SKILL, seed)
        _, _, _, trials = read(q)
        for wrong_text in q.parts[0].distractors:
            wrong_p = as_fraction(wrong_text)
            expected = wrong_p * trials
            green = trials - expected
            if expected.denominator != 1 or expected <= 0 or green <= 0 or len({expected, green, wrong_p}) < 3:
                continue
            checked += 1
            result = grade(skill, q, {"a": wrong_text, "b": str(expected), "c": str(green)})
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 2 and result.parts["c"].marks == 1, (seed, wrong_p)
    assert checked >= 50


def test_follow_through_from_wrong_expected_frequency():
    skill = skill_of(SKILL)
    checked = 0
    for seed in range(100):
        q = exam_question(SKILL, seed)
        _, _, _, trials = read(q)
        wrong_b = int(as_fraction(q.parts[1].distractors[0]))
        carried = trials - wrong_b
        if carried <= 0 or carried == wrong_b:
            continue
        checked += 1
        result = grade(skill, q, {**canonical_reply(q), "b": str(wrong_b), "c": str(carried)})
        assert result.parts["b"].marks == 0 and result.parts["c"].marks == 1, (seed, wrong_b)
    assert checked >= 30


def test_all_wrong_earns_nothing():
    q = exam_question(SKILL, 1)
    assert grade(skill_of(SKILL), q, {"a": "9/7", "b": "9998", "c": "9997"}).marks == 0
