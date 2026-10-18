"""Tests the multi-part exam source of `composite-perimeter`: two missing sides, the perimeter, then the cost of a fence.

The oracle re-derives every part's answer from the rendered stem alone, with exact arithmetic and no code shared
with the generator.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.core import PRESETS
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_pr3_composite_perimeter"
SKILL_ID = "composite-perimeter"
LABELS = ["a", "b", "c", "d"]
MARKS = [1, 1, 2, 2]
FOLLOWS = [False, False, True, True]

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def half_up(x) -> int:
    return math.floor(Fraction(x) + Fraction(1, 2))


def cents(x) -> str:
    """Two decimal places, ties away from zero, from an exact fraction."""
    scaled = Fraction(x) * 100
    return f"{half_up(scaled) / 100:.2f}"


def parse(text):
    W, H = map(int, re.search(r"(\d+) m by (\d+) m rectangle", text).groups())
    cut = re.search(r"cut-out is (\d+) m along an? \d+ m side and (\d+) m along", text) or re.search(
        r"measures (\d+) m by (\d+) m", text
    )
    w, h = map(int, cut.groups())
    g = int(re.search(r"(\d+) m (?:gateway|gap)", text).group(1))
    rate = int(re.search(r"\$(\d+) per metre", text).group(1))
    return dict(W=W, H=H, w=w, h=h, g=g, rate=rate)


def oracle(text):
    v = parse(text)
    W, H, w, h = v["W"], v["H"], v["w"], v["h"]
    outline = [(0, 0), (W, 0), (W, H - h), (W - w, H - h), (W - w, H), (0, H)]
    perimeter = sum(abs(x2 - x1) + abs(y2 - y1) for (x1, y1), (x2, y2) in zip(outline, outline[1:] + outline[:1]))
    return {
        "a": str(W - w),
        "b": str(H - h),
        "c": str(perimeter),
        "d": str((perimeter - v["g"]) * v["rate"]),
    }


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in range(300)]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == LABELS
        assert [p.marks for p in q.parts] == MARKS
        assert [p.follow_through is not None for p in q.parts] == FOLLOWS
        assert all(p.answer_kind == "number" for p in q.parts)
        assert all(p.distractors and len(p.distractors) <= 3 for p in q.parts)
        assert all(p.option_notes for p in q.parts)


def test_prompt_shapes_add_to_hard_tier():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    src = [s for s in _SKILL.sources if s.difficulty != "exam"][-1]
    diff = src.difficulty if src.difficulty in PRESETS.get(src.topic, {}) else None
    hard = {prompt_shape(make(src.topic, seed=s, difficulty=diff, **src.params)) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard, "the exam source adds no prompt shape beyond the hard tier"


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(MARKS), q.text


def test_oracle_agrees_over_300_seeds(questions):
    bad = [(q.seed, q.stem) for q in questions if oracle(q.stem) != _canonical(q)]
    assert bad == []


def test_distractors_rejected(questions):
    for q in questions[:120]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.seed, p.label, d)


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {label: "999999" for label in LABELS})
        assert result.marks == 0 and not result.correct


def test_follow_through_from_wrong_first_side(questions):
    for q in questions[:100]:
        v = parse(q.stem)
        a_wrong = int(q.parts[0].distractors[0])
        c_wrong = v["W"] + v["H"] + v["w"] + v["h"] + a_wrong + (v["H"] - v["h"])
        d_wrong = (c_wrong - v["g"]) * v["rate"]
        canon = _canonical(q)
        result = grade(_SKILL, q, {**canon, "a": str(a_wrong), "c": str(c_wrong), "d": str(d_wrong)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through
        assert result.marks == 5 and not result.correct


def test_follow_through_cost_from_wrong_perimeter(questions):
    for q in questions[:60]:
        v = parse(q.stem)
        c_wrong = int(q.parts[2].distractors[0])
        d_wrong = (c_wrong - v["g"]) * v["rate"]
        result = grade(_SKILL, q, {**_canonical(q), "c": str(c_wrong), "d": str(d_wrong)})
        assert result.parts["c"].marks == 0 and result.parts["d"].marks == 2
