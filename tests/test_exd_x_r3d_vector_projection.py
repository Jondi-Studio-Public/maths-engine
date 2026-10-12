"""Exam chain of `vector-dot-product`: AB . AC, |AC|, and the scalar projection of AB onto AC."""

from __future__ import annotations

import math
import re

import pytest

from tests import exd_x_r3d_support as S

TOPIC = "exd_x_r3d_vector_projection"
SKILL = S.load_skill("vector-dot-product")


def _points(q):
    out = {}
    for name, x, y, z in re.findall(r"([ABC])\((-?\d+), (-?\d+), (-?\d+)\)", q.stem):
        out[name] = (int(x), int(y), int(z))
    return out


def oracle(q):
    p = _points(q)
    ab = [b - a for a, b in zip(p["A"], p["B"])]
    ac = [c - a for a, c in zip(p["A"], p["C"])]
    dot = sum(x * y for x, y in zip(ab, ac))
    length = math.sqrt(sum(x * x for x in ac))
    proj = round(dot / length)
    perp2 = sum(x * x for x in ab) - proj * proj
    return {"a": str(dot), "b": str(round(length)), "c": str(proj), "d": str(perp2)}


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_perpendicular_distance_from_cross_product(questions):
    # |AB x AC|^2 / |AC|^2 is the squared distance from B to the line AC, by a route that never forms a projection.
    for q in questions[:100]:
        p = _points(q)
        ab = [b - a for a, b in zip(p["A"], p["B"])]
        ac = [c - a for a, c in zip(p["A"], p["C"])]
        cross = (
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        )
        assert sum(x * x for x in cross) == int(q.parts[3].answer_text) * sum(x * x for x in ac)


def test_length_is_whole(questions):
    for q in questions:
        p = _points(q)
        ac = [c - a for a, c in zip(p["A"], p["C"])]
        assert math.isqrt(sum(x * x for x in ac)) ** 2 == sum(x * x for x in ac)


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_number_format(questions)
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_dot_and_length(questions):
    def chain(q):
        a = int(q.parts[0].distractors[0])
        b = int(q.parts[1].distractors[0])
        c = round(a / b)
        if c in (a, b) or a % b:
            return None
        ab2 = sum((y - x) ** 2 for x, y in zip(_points(q)["A"], _points(q)["B"]))
        d = ab2 - c * c
        if d in (a, b, c):
            return None
        return {"a": str(a), "b": str(b), "c": str(c), "d": str(d)}, {"c", "d"}

    S.check_follow_through(SKILL, questions, chain, n=300)


def test_follow_through_with_correct_length(questions):
    seen = 0
    for q in questions[:100]:
        a = int(q.parts[0].distractors[0])
        b = int(q.parts[1].answer_text)
        c = round(a / b)
        if a % b or c in (a, b):
            continue
        d = sum((y - x) ** 2 for x, y in zip(_points(q)["A"], _points(q)["B"])) - c * c
        result = S.grade(SKILL, q, {"a": str(a), "b": str(b), "c": str(c), "d": str(d)})
        assert result.parts["b"].marks == q.parts[1].marks
        assert result.parts["c"].marks == q.parts[2].marks
        seen += 1
    assert seen >= 10


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "9999", "b": "8888", "c": "7777", "d": "6666"})


def test_deterministic():
    S.check_deterministic(TOPIC)
