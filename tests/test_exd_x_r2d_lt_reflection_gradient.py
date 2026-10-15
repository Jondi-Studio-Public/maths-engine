"""Tests the three-part exam chain on a reflection in an unknown line through the origin (gradient, then the image of a point)."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_r2d_lt_reflection_gradient"
SKILL_ID = "senior-define-and-use-basic-linear-transformations-dilations-of"
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


LABELS = ["a", "b", "c"]
MARKS = [2, 2, 2]
FT_FLAGS = [False, True, True]


def _txt(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def _points(q):
    pts = re.findall(r"\((-?\d+), (-?\d+)\)", q.text)
    (px, py), (p1x, p1y) = pts[0], pts[1]
    qx, qy = [int(v) for v in re.search(r"Q\((-?\d+), (-?\d+)\)", q.text).groups()]
    return int(px), int(py), int(p1x), int(p1y), qx, qy


def _image(m: Fraction, x: int, y: int):
    """Reflect (x, y) in y = m x by projecting onto the direction (1, m): image = 2 proj - point."""
    k = (x + m * y) / (1 + m * m)
    return 2 * k - x, 2 * k * m - y


def oracle(q):
    px, py, p1x, p1y, qx, qy = _points(q)
    m = Fraction(py + p1y, px + p1x)
    ix, iy = _image(m, qx, qy)
    # the stated image of P must be what this mirror gives, which checks the gradient independently
    assert _image(m, px, py) == (p1x, p1y)
    return {"a": _txt(m), "b": _txt(ix), "c": _txt(iy)}


def test_follow_through_from_a_wrong_gradient(questions):
    for q in questions[:100]:
        px, py, p1x, p1y, qx, qy = _points(q)
        wrong = Fraction(q.parts[0].distractors[0])
        ix, iy = _image(wrong, qx, qy)
        result = grade(_SKILL, q, {"a": _txt(wrong), "b": _txt(ix), "c": _txt(iy)})
        assert result.parts["a"].marks == 0
        for label, value in (("b", ix), ("c", iy)):
            if value == wrong:
                continue  # a reply that only repeats the earlier value is not follow-through
            assert result.parts[label].marks == 2, (q.text, label, _txt(wrong))
            if _txt(value) != q.parts["abc".index(label)].answer_text:
                assert result.parts[label].follow_through is True
        assert not result.correct and result.marks <= 4


def test_a_correct_gradient_alone_earns_only_its_own_marks(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": q.parts[0].answer_text})
        assert result.marks == 2
