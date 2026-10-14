"""Tests for the multi-part exam source of `coordinate-plane-readiness`.

The oracle reads the starting point (as coordinates, or from its distances to the axes) and the two named axes
from the rendered stem, applies the reflections and measures the rectangle.
"""

from __future__ import annotations

import re

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n2_coordinate_reflections"
SKILL_ID = "coordinate-plane-readiness"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_HARD = next(s for s in _SKILL.sources if s.difficulty == "hard")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _reflect(p, axis):
    return (p[0], -p[1]) if axis == "x" else (-p[0], p[1])


def _start(stem):
    m = re.search(r"\((-?\d+), (-?\d+)\)", stem)
    if m:
        return int(m.group(1)), int(m.group(2))
    h = re.search(r"(\d+) units? (left|right)", stem)
    v = re.search(r"(\d+) units? (above|below)", stem)
    return (
        int(h.group(1)) * (1 if h.group(2) == "right" else -1),
        int(v.group(1)) * (1 if v.group(2) == "above" else -1),
    )


def oracle(q) -> dict[str, str]:
    p = _start(q.stem)
    first, second = re.findall(r"in the ([xy])-axis", q.stem)
    qq = _reflect(p, first)
    r = _reflect(qq, second)
    pq = abs(p[0] - qq[0]) + abs(p[1] - qq[1])
    qr = abs(qq[0] - r[0]) + abs(qq[1] - r[1])
    return {"a": f"({r[0]}, {r[1]})", "b": str(pq), "c": str(qr), "d": str(pq * qr)}


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["pair", "number", "number", "number"]
        assert [p.marks for p in q.parts] == [2, 1, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, False, True]


def test_prompt_shapes_new_against_hard_tier():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make(_HARD.topic, seed=s, difficulty="hard", **dict(_HARD.params or {}))) for s in range(60)}
    assert len(exam) >= 3 and exam - hard


def test_oracle_agrees_over_300_seeds(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_both_stem_forms_and_both_orders_occur(questions):
    assert {bool(re.search(r"\d+ units? (left|right|above|below)", q.stem)) for q in questions} == {True, False}
    assert {re.findall(r"in the ([xy])-axis", q.stem)[0] for q in questions} == {"x", "y"}


def test_key_accepted_and_every_distractor_rejected(questions):
    for q in questions[:120]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (q.text, p.label, d)


def test_tagged_distractors_carry_values(questions):
    for q in questions[:60]:
        for p in q.parts:
            assert p.distractor_values, (q.text, p.label)


def test_follow_through_from_wrong_side_lengths(questions):
    checked = 0
    for q in questions[:120]:
        b_wrong = int(q.parts[1].distractors[0])
        c = int(q.parts[2].answer_text)
        d_wrong = b_wrong * c
        if d_wrong in (int(q.parts[3].answer_text), b_wrong, c):
            continue
        result = grade(_SKILL, q, {**_canonical(q), "b": str(b_wrong), "d": str(d_wrong)})
        assert result.parts["b"].marks == 0
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through, (q.text, b_wrong, d_wrong)
        checked += 1
    assert checked >= 60
