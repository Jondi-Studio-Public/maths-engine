"""Exam source of `f10-probability-scale-and-expectation`: probability from counts, expected frequency, trials needed."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u6_probability_expectation"
SKILL_ID = "f10-probability-scale-and-expectation"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _oracle(q):
    counts = re.search(r"(\d+) red(?: tickets| marbles)?,? (?:and )?(\d+) green(?: and |, )(\d+) blue", q.stem)
    if counts is None:
        counts = re.search(r"(\d+) are red, (\d+) are green and (\d+) are blue", q.stem)
    r, g, b = (int(v) for v in counts.groups())
    trials = int(re.search(r"expected in (\d+)", q.text).group(1))
    target = int(re.search(r"before (\d+) red", q.text).group(1))
    p = Fraction(r, r + g + b)
    return {"a": f"{float(p):.2f}", "b": str(int(p * trials)), "c": str(int(target / p))}


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        assert _oracle(q) == {p.label: p.answer_text for p in q.parts}, (seed, q.text)


def test_probabilities_are_probabilities():
    for seed in range(100):
        q = _exam(seed)
        assert 0 < float(q.parts[0].answer_text) < 1
        assert q.parts[0].sensible_range == (0, 1)
        for d in q.parts[0].distractors:
            assert 0 <= float(d) <= 1, (seed, d)


def test_grader_accepts_key_and_rejects_distractors():
    shown = 0
    for seed in range(100):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 5
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            for d in p.distractors:
                shown += 1
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d)
    assert shown >= 300


def test_follow_through_from_a_wrong_probability():
    for seed in range(100):
        q = _exam(seed)
        wrong = q.parts[0].distractors[0]
        w = Fraction(wrong)
        trials = int(re.search(r"expected in (\d+)", q.text).group(1))
        target = int(re.search(r"before (\d+) red", q.text).group(1))
        b = w * trials
        c = target / w
        reply = {"a": wrong, "b": str(int(b + Fraction(1, 2))), "c": str(int(c + Fraction(1, 2)))}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks, (seed, reply)
        assert result.parts["c"].marks == result.parts["c"].max_marks, (seed, reply)
        assert result.parts["b"].follow_through
