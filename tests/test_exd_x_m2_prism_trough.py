"""Tests the multi-part exam source `exd_x_m2_prism_trough` for `f10-triangular-prism-volume`."""

from __future__ import annotations

import re
from fractions import Fraction
from math import floor, isqrt

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_m2_prism_trough"
SKILL_ID = "f10-triangular-prism-volume"
LABELS = ["a", "b", "c"]
MARKS = [2, 2, 1]

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
    answers = [_canonical(_exam(s))[LABELS[-1]] for s in SEEDS]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 0.05 * len(answers)


def _nums(q):
    side = int(re.search(r"sides of (\d+) cm", q.stem).group(1))
    base = int(re.search(r"base of (\d+) cm", q.stem).group(1))
    length = int(re.search(r"(\d+) cm long", q.stem).group(1))
    return side, base, length


def oracle(q):
    side, base, length = _nums(q)
    h2 = side * side - (base // 2) ** 2
    h = isqrt(h2)
    assert h * h == h2
    volume = Fraction(base * h, 2) * length
    return {"a": str(h), "b": str(volume), "c": _rnd(Fraction(volume) / 1000, 1)}


def test_follow_through_from_wrong_height():
    for seed in range(100):
        q = _exam(seed)
        _, base, length = _nums(q)
        h = int(q.parts[0].distractors[0])
        volume = Fraction(base * h, 2) * length
        reply = {"a": str(h), "b": str(volume), "c": _rnd(Fraction(volume) / 1000, 1)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through


def test_follow_through_litres_from_wrong_volume():
    for seed in range(60):
        q = _exam(seed)
        wrong = int(q.parts[1].distractors[0])
        result = grade(_SKILL, q, {**_canonical(q), "b": str(wrong), "c": _rnd(Fraction(wrong, 1000), 1)})
        assert result.parts["b"].marks == 0 and result.parts["c"].marks == 1
