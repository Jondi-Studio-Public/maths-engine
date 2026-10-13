"""Tests the multi-part exam source `exd_x_m2_area_limits` for `f10-measurement-error-interpretation`."""

from __future__ import annotations

import re
from fractions import Fraction
from math import floor, isqrt

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_m2_area_limits"
SKILL_ID = "f10-measurement-error-interpretation"
LABELS = ["a", "b", "c", "d"]
MARKS = [1, 2, 1, 2]

SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _rnd(x, dp):
    scaled = Fraction(x) * 10**dp
    return f"{Fraction(floor(scaled + Fraction(1, 2)), 10**dp):.{dp}f}"


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _hard_shapes():
    hard = [s for s in _SKILL.sources if s.difficulty == "hard"][0]
    return {prompt_shape(make(hard.topic, seed=s, difficulty="hard", **hard.params)) for s in range(60)}


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_prompt_variety_and_new_shapes():
    shapes = {prompt_shape(_exam(s)) for s in range(60)}
    assert len(shapes) >= 3
    assert shapes - _hard_shapes()


def test_shape_of_question():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == LABELS
        assert [p.marks for p in q.parts] == MARKS
        assert all(p.answer_kind == "number" for p in q.parts)
        assert all(len(p.distractors) >= 3 for p in q.parts)


def test_key_rederived_from_prompt():
    for seed in SEEDS:
        q = _exam(seed)
        assert oracle(q) == _canonical(q), (seed, q.text)


def test_full_marks_and_distractors_rejected():
    for seed in range(120):
        q = _exam(seed)
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, (seed, q.answer_text)
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d)


def test_answer_space():
    answers = [_exam(s).answer_text for s in SEEDS]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 0.05 * len(answers)


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()


def _nums(q):
    length, width = (int(v) for v in re.search(r"length of (\d+) \w+ and a width of (\d+)", q.stem).groups())
    unit = int(re.search(r"nearest (\d+) ", q.stem).group(1))
    return length, width, unit


def oracle(q):
    length, width, unit = _nums(q)
    half = Fraction(unit, 2)
    area = length * width
    upper = (length + half) * (width + half)
    err = upper - area
    pct = 100 * err / area
    return {"a": str(area), "b": _rnd(upper, 2), "c": _rnd(err, 2), "d": _rnd(pct, 1)}


def test_follow_through_from_wrong_upper_area():
    for seed in range(100):
        q = _exam(seed)
        length, width, _ = _nums(q)
        area = length * width
        upper = Fraction(q.parts[1].distractors[0])
        err = upper - area
        if err <= 0:
            continue
        reply = {"a": str(area), "b": q.parts[1].distractors[0], "c": _rnd(err, 2), "d": _rnd(100 * err / area, 1)}
        result = grade(_SKILL, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through
