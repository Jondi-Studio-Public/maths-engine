"""Tests the multi-part exam tier of `senior-use-formulas-to-find-the-volume-of-pyramids`: height, volume, litres."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_fu_pyramid_volume"
SKILL_ID = "senior-use-formulas-to-find-the-volume-of-pyramids"
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _dp1(x: Fraction) -> str:
    return f"{math.floor(x * 10 + Fraction(1, 2)) / 10:.1f}"


def oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the stem alone."""
    slant = int(re.search(r"slant height of (\d+) cm", q.stem).group(1))
    square = re.search(r"square base of side (\d+) cm", q.stem)
    if square:
        b = w = int(square.group(1))
    else:
        b, w = (int(n) for n in re.search(r"base (\d+) cm by (\d+) cm", q.stem).groups())
    h2 = slant**2 - (w // 2) ** 2
    h = math.isqrt(h2)
    assert h * h == h2
    volume = Fraction(b * w * h, 3)
    return {"a": str(h), "b": _dp1(volume), "c": _dp1(volume / 1000)}


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 2, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_distractors_rejected(questions):
    for q in questions[:100]:
        canon = _canonical(q)
        for p in q.parts:
            assert p.distractors, (q.text, p.label)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: str(d)})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_height(questions):
    checked = 0
    for q in questions[:100]:
        nums = [int(n) for n in re.findall(r"(\d+) cm", q.stem)]
        base_area = nums[0] * nums[-2]
        wrong_h = Fraction(int(q.parts[0].answer_text) + 7)
        v = Fraction(base_area) * wrong_h / 3
        reply = {"a": str(wrong_h), "b": _dp1(v), "c": _dp1(v / 1000)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label)
        checked += 1
    assert checked == 100


def test_all_wrong_earns_zero(questions):
    for q in questions[:50]:
        result = grade(_SKILL, q, {"a": "999", "b": "1.1", "c": "2.2"})
        assert result.marks == 0


def test_shapes_and_variety():
    shapes = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard = {prompt_shape(make("volume_of_pyramids_and_spheres", seed=s, mode="sphere")) for s in range(60)}
    assert len(shapes) >= 3 and shapes - hard
    assert len({make(TOPIC, seed=s).parts[1].answer_text for s in SEEDS}) >= 100
    for s in range(60):
        assert find_text_defects(make(TOPIC, seed=s)) == []
