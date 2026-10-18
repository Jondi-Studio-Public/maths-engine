"""Tests the multi-part exam source `exd_x_m2_wall_paint` for `f10-irregular-composite-area`."""

from __future__ import annotations

import re
from fractions import Fraction
from math import floor, isqrt

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_m2_wall_paint"
SKILL_ID = "f10-irregular-composite-area"
LABELS = ["a", "b", "c"]
MARKS = [2, 1, 2]

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
    w, ht = (int(v) for v in re.search(r"(\d+) m wide and (\d+) m high", q.stem).groups())
    gable = int(re.search(r"height of (\d+) m", q.stem).group(1))
    n = int(re.search(r"has (\d+) rectangular", q.stem).group(1))
    wp, wq = (int(v) for v in re.search(r"(\d+) m by (\d+) m", q.stem).groups())
    cover = int(re.search(r"covers (\d+) m", q.stem).group(1))
    coats = int(re.search(r"needs (\d+) coats", q.stem).group(1))
    return w, ht, gable, n, wp, wq, cover, coats


def oracle(q):
    w, ht, gable, n, wp, wq, cover, coats = _nums(q)
    outline = w * ht + Fraction(w * gable, 2)
    paint = outline - n * wp * wq
    litres = coats * paint / cover
    return {"a": str(outline), "b": str(paint), "c": _rnd(litres, 2)}


def test_follow_through_from_wrong_outline():
    for seed in range(100):
        q = _exam(seed)
        _, _, _, n, wp, wq, cover, coats = _nums(q)
        outline = int(q.parts[0].distractors[0])
        paint = outline - n * wp * wq
        reply = {"a": str(outline), "b": str(paint), "c": _rnd(Fraction(coats * paint, cover), 2)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through
