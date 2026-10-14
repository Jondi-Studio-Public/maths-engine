"""Tests the three-part exam chain on a 3D parallelogram with an unknown coordinate (find k, then a dot product, then an angle)."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_r2d_vector_addition_chain"
SKILL_ID = "vector-addition"
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _hard(seed):
    src = [s for s in _SKILL.sources if s.difficulty != "exam"][-1]  # the hardest practice tier
    return make(src.topic, seed=seed, **src.params)


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam" and s.effective_key == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].params == {"tier": "exam"} and list(exam[0].stages) == ["exam"]


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == LABELS
        assert [p.marks for p in q.parts] == MARKS
        assert [p.follow_through is not None for p in q.parts] == FT_FLAGS
        assert all(p.distractors and all(d in p.option_notes for d in p.distractors) for p in q.parts)


def test_prompt_shapes_beyond_the_hard_tier():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard = {prompt_shape(_hard(s)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard


def test_oracle_agrees_with_the_rendered_prompt(questions):
    bad = [(q.text, oracle(q), _canonical(q)) for q in questions if oracle(q) != _canonical(q)]
    assert bad == [], bad[:2]


def test_full_marks_for_the_key(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(MARKS), q.text


def test_every_distractor_is_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_all_wrong_earns_nothing(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {p.label: f"9999{i}" for i, p in enumerate(q.parts)})
        assert result.marks == 0 and not result.correct


def test_answer_space(questions):
    tuples = Counter(tuple(_canonical(q).values()) for q in questions)
    assert len(tuples) >= 100, len(tuples)
    assert max(tuples.values()) / len(questions) <= 0.05, tuples.most_common(3)


def test_deterministic():
    assert make(TOPIC, seed=7).to_dict() == make(TOPIC, seed=7).to_dict()


import math

import sympy

LABELS = ["a", "b", "c"]
MARKS = [2, 2, 2]
FT_FLAGS = [False, True, True]
_NEXT = {"A": "B", "B": "C", "C": "D", "D": "A"}
_PREV = {"A": "D", "B": "A", "C": "B", "D": "C"}


def _points(q):
    found = dict(re.findall(r"([ABCD]) (?:is|is the point) \(([^)]*)\)", q.stem))
    return {n: [c.strip() for c in v.split(",")] for n, v in found.items()}


def _solve(q):
    raw = _points(q)
    k = sympy.Symbol("k")
    pts = {n: [k if c == "k" else sympy.Integer(int(c)) for c in v] for n, v in raw.items()}
    unknown = next(n for n, v in raw.items() if "k" in v)
    axis = raw[unknown].index("k")
    k_val = sympy.solve(pts["A"][axis] + pts["C"][axis] - pts["B"][axis] - pts["D"][axis], k)
    assert len(k_val) == 1
    return pts, unknown, k, int(k_val[0])


def _sides(q, k_value=None):
    pts, v, k, kv = _solve(q)
    kv = kv if k_value is None else k_value
    pt = {n: [sympy.sympify(c).subs(k, kv) for c in p] for n, p in pts.items()}
    u1 = [pt[_NEXT[v]][i] - pt[v][i] for i in range(3)]
    u2 = [pt[_PREV[v]][i] - pt[v][i] for i in range(3)]
    return u1, u2


def _answers(q, k_value=None):
    u1, u2 = _sides(q, k_value)
    dot = sum(x * y for x, y in zip(u1, u2))
    cos = dot / (sympy.sqrt(sum(x * x for x in u1)) * sympy.sqrt(sum(x * x for x in u2)))
    return dot, math.floor(math.degrees(math.acos(float(cos))) + 0.5)


def oracle(q):
    kv = _solve(q)[3]
    dot, angle = _answers(q)
    return {"a": str(kv), "b": str(int(dot)), "c": str(angle)}


def test_follow_through_from_a_wrong_k(questions):
    for q in questions[:100]:
        wrong = next(d for d in q.parts[0].distractors if d != "0")
        dot, angle = _answers(q, int(wrong))
        if str(int(dot)) == q.parts[1].answer_text or str(angle) == q.parts[2].answer_text:
            continue  # the wrong k happens to give the right dot product or angle
        reply = {"a": wrong, "b": str(int(dot)), "c": str(angle)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, q.text
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, q.text


def test_a_wrong_dot_product_still_carries_into_the_angle(questions):
    for q in questions[:100]:
        u1, u2 = _sides(q)
        wrong = int(q.parts[1].distractors[0])
        cos = wrong / (math.sqrt(sum(float(x) ** 2 for x in u1)) * math.sqrt(sum(float(x) ** 2 for x in u2)))
        if abs(cos) >= 1:
            continue
        angle = str(math.floor(math.degrees(math.acos(cos)) + 0.5))
        result = grade(_SKILL, q, {**_canonical(q), "b": str(wrong), "c": angle})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True


def test_the_unknown_is_never_zero(questions):
    assert all(q.parts[0].answer_text != "0" for q in questions)
