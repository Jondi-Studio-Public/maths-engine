"""Proofs for `f10-angle-line-and-plane`: the angle between a line and a plane in a cuboid or a pyramid.

The oracle reads only the question text (the solid's dimensions and the named line and plane), places the solid on
coordinates and computes the angle with the vector formula, which is not the route the generator keys with.
"""

from __future__ import annotations

import math
import re

import pytest

from mathgen import make
from mathgen.core import stated_precision_tolerance
from mathgen.engine import grade as grade_skill
from mathgen.grading import grade
from mathgen.skills import load_skills

SKILL_ID = "f10-angle-line-and-plane"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(80)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
SOURCES = {s.difficulty: s for s in _SKILL.sources}
TOL = float(2 * stated_precision_tolerance(1))


def _make(tier, seed):
    src = SOURCES[tier]
    return make(src.topic, seed=seed, difficulty=src.difficulty, **(src.params or {}))


def _angle(direction, normal):
    dot = abs(sum(d * n for d, n in zip(direction, normal)))
    return math.degrees(math.asin(dot / (math.dist((0, 0, 0), direction) * math.dist((0, 0, 0), normal))))


def _oracle(text: str) -> float:
    cuboid = re.search(
        r"AB = (\d+) cm and BC = (\d+) cm, and the vertical edges AE, BF, CG and DH are each (\d+) cm", text
    )
    if cuboid and "cuboid" in text:
        a, b, h = map(int, cuboid.groups())
        pts = {
            "A": (0, 0, 0), "B": (a, 0, 0), "C": (a, b, 0), "D": (0, b, 0),
            "E": (0, 0, h), "F": (a, 0, h), "G": (a, b, h), "H": (0, b, h),
        }  # fmt: skip
        line = re.search(r"The (?:face|space) diagonal ([A-H])([A-H]) is drawn", text)
        p, q = pts[line.group(1)], pts[line.group(2)]
        d = tuple(qi - pi for pi, qi in zip(p, q))
        if "the base ABCD" in text:
            return _angle(d, (0, 0, 1))
        face = re.search(r"vertical face ([A-H]{4})", text).group(1)
        normal = {"ABFE": (0, 1, 0), "CDHG": (0, 1, 0), "BCGF": (1, 0, 0), "DAEH": (1, 0, 0)}[face]
        return _angle(d, normal)
    height = re.search(
        r"AB = (\d+) cm and BC = (\d+) cm\. The apex V is directly above the centre of the base, and each", text
    )
    if height:
        a, b = map(int, height.groups())
        e = int(re.search(r"sloping edge \(VA, VB, VC and VD\) is (\d+) cm", text).group(1))
        half = math.hypot(a, b) / 2
        return _angle((half, 0, -math.sqrt(e * e - half * half)), (0, 0, 1))
    apex = re.search(r"The apex V is (\d+) m directly above", text)
    h = int(apex.group(1))
    rect = re.search(r"AB = (\d+) m and BC = (\d+) m", text)
    if rect:
        a, b = map(int, rect.groups())
    else:
        a = b = int(re.search(r"side (\d+) m", text).group(1))
    return _angle((a / 2, b / 2, -h), (0, 0, 1))


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_the_oracle(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert abs(float(q.answer) - _oracle(q.text)) <= 0.05 + 1e-9, (tier, seed, q.text)
        assert q.answer_text.count(".") == 1 and len(q.answer_text.split(".")[1]) == 1


@pytest.mark.parametrize("tier", TIERS)
def test_grader_accepts_the_key_and_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = _make(tier, seed)
        assert grade("number", q, q.answer_text).correct
        assert len(q.distractors) == 3
        for d in q.distractors:
            assert not grade("number", q, d).correct, (tier, seed, d)
            assert len(d.split(".")[1]) == 1  # same number of decimal places as the key
            assert abs(float(d) - float(q.answer_text)) > TOL


def test_tiers_differ_in_shape():
    easy = [_make("easy", s).text for s in SEEDS]
    medium = [_make("medium", s).text for s in SEEDS]
    hard = [_make("hard", s).text for s in SEEDS]
    assert all("face diagonal" in t for t in easy)
    assert all("space diagonal" in t or "pyramid" in t for t in medium + hard)
    assert any("vertical face" in t for t in hard) and any("sloping edge" in t and "long" in t for t in hard)
    assert not any("vertical face" in t for t in easy + medium)


@pytest.mark.parametrize("tier", TIERS)
def test_determinism_and_variety(tier):
    assert _make(tier, 5).to_json() == _make(tier, 5).to_json()
    assert len({_make(tier, s).text for s in SEEDS}) >= 70


def test_exam_chain_follow_through_and_parts():
    src = SOURCES["exam"]
    for seed in SEEDS:
        q = make(src.topic, seed=seed, **(src.params or {}))
        a, b, c, d = q.parts
        assert [p.marks for p in q.parts] == [1, 2, 2, 3]
        base = int(a.answer_text)
        dims = re.search(r"AB = (\d+) cm and BC = (\d+) cm, and the vertical edges .* are (\d+) cm long", q.text)
        x, y, h = map(int, dims.groups())
        assert base * base == x * x + y * y
        assert abs(float(b.answer_text) - math.sqrt(base**2 + h**2)) <= 0.05 + 1e-9
        assert abs(float(c.answer_text) - _angle((x, y, h), (0, 0, 1))) <= 0.05 + 1e-9
        assert abs(float(d.answer_text) - _angle((x, y, h), (1, 0, 0))) <= 0.05 + 1e-9
        assert c.follow_through and b.follow_through
        full = {"a": a.answer_text, "b": b.answer_text, "c": c.answer_text, "d": d.answer_text}
        assert grade("number", q, full).correct
        for part in q.parts:
            for dist in part.distractors:
                assert not grade("number", part, dist).correct


def test_exam_follow_through_from_a_wrong_part_a():
    src = SOURCES["exam"]
    checked = 0
    for seed in range(30):
        q = make(src.topic, seed=seed, **(src.params or {}))
        a, b, c, d = q.parts
        x, y, h = map(
            int,
            re.search(r"AB = (\d+) cm and BC = (\d+) cm, and the vertical edges .* are (\d+) cm long", q.text).groups(),
        )
        wrong_base = int(a.answer_text) + 1
        space = f"{math.sqrt(wrong_base**2 + h**2):.1f}"
        angle = f"{math.degrees(math.atan2(h, wrong_base)):.1f}"
        if space == b.answer_text or angle == c.answer_text:
            continue
        reply = {"a": str(wrong_base), "b": space, "c": angle, "d": d.answer_text}
        result = grade_skill(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == b.marks and result.parts["b"].follow_through, (seed, reply)
        assert result.parts["c"].marks == c.marks and result.parts["c"].follow_through, (seed, reply)
        assert result.parts["d"].marks == d.marks
        checked += 1
    assert checked >= 20
