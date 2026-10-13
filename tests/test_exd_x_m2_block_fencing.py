"""Tests the multi-part exam source `exd_x_m2_block_fencing` for `f10-regular-irregular-area-perimeter`."""

from __future__ import annotations

import re
from fractions import Fraction
from math import floor, isqrt

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_m2_block_fencing"
SKILL_ID = "f10-regular-irregular-area-perimeter"
LABELS = ["a", "b", "c"]
MARKS = [1, 2, 2]

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
    answers = [_canonical(_exam(s))[LABELS[-1]] for s in SEEDS]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 0.05 * len(answers)


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()


def _nums(q):
    area, w = (int(v) for v in re.search(r"area of (\d+) m² and a width of (\d+) m", q.stem).groups())
    p, qq = (int(v) for v in re.search(r"(\d+) m by (\d+) m section", q.stem).groups())
    price = int(re.search(r"\$(\d+) per", q.stem).group(1))
    return area, w, p, qq, price


def oracle(q):
    area, w, p, qq, price = _nums(q)
    ln = Fraction(area, w)
    assert ln.denominator == 1
    return {"a": str(ln), "b": str(area - p * qq), "c": str(2 * (w + ln) * price)}


def test_follow_through_from_wrong_length():
    for seed in range(100):
        q = _exam(seed)
        _, w, p, qq, price = _nums(q)
        ln = int(q.parts[0].distractors[0])
        reply = {"a": str(ln), "b": str(w * ln - p * qq), "c": str(2 * (w + ln) * price)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through
