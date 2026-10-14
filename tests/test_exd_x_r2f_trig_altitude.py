"""Exam source of `senior-apply-the-cosine-and-sine-ratios-to-find`: a four-part chain on a right-angled triangle."""

from __future__ import annotations

import math
import re
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2f_trig_altitude"
SKILL_ID = "senior-apply-the-cosine-and-sine-ratios-to-find"
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _dp1(x: float) -> str:
    return str(Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the rendered stem alone, by coordinates rather than the generator's route."""
    h = int(re.search(r"AC\s*(?:=|is|measures)\s*(\d+)", q.stem).group(1))
    vertex, theta = re.search(r"Angle B([AC])[CA] is (\d+)°", q.stem).groups()
    t = math.radians(int(theta))
    # Put B at the origin with the legs along the axes: P = (px, 0) carries the given angle, Q = (0, qy).
    px, qy = h * math.cos(t), h * math.sin(t)
    # Foot of the perpendicular from B to line PQ, found by projection rather than by a trig identity.
    dx, dy = -px, qy
    s = px * px / (dx * dx + dy * dy)  # parameter along P -> Q
    fx, fy = px + s * dx, s * dy
    bd = math.hypot(fx, fy)
    pd = math.hypot(fx - px, fy)
    return {"a": _dp1(px), "b": _dp1(qy), "c": _dp1(bd), "d": _dp1(pd), "vertex": vertex}


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam" and s.effective_key == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)
        assert all(p.distractors and p.option_notes for p in q.parts)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {
        prompt_shape(make("sine_cosine_ratio_triangle", seed=s, difficulty="hard", variant="chained"))
        for s in range(60)
    }
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects(questions):
    for q in questions[:60]:
        assert find_text_defects(q) == []


def test_oracle_agrees(questions):
    for q in questions:
        o = oracle(q)
        canon = _canonical(q)
        given_p = o["vertex"]
        assert q.parts[0].prompt.startswith(f"Find the length of {'AB' if given_p == 'A' else 'BC'}"), q.text
        assert {k: o[k] for k in "abcd"} == canon, q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6


def test_distractors_rejected(questions):
    for q in questions[:120]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_legs(questions):
    """A wrong (a) and (b) used correctly in (c) and (d) earn those marks."""
    for q in questions[:80]:
        h = int(re.search(r"AC\s*(?:=|is|measures)\s*(\d+)", q.stem).group(1))
        theta = int(re.search(r"Angle B[AC][CA] is (\d+)°", q.stem).group(1))
        a, b = float(q.parts[0].distractors[0]), float(q.parts[1].distractors[0])
        c = a * b / h
        d = float(_dp1(c)) / math.tan(math.radians(theta))
        reply = {"a": q.parts[0].distractors[0], "b": q.parts[1].distractors[0], "c": _dp1(c), "d": _dp1(d)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 0
        if reply["c"] != q.parts[2].answer_text:
            assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (q.text, reply)
        if reply["d"] != q.parts[3].answer_text:
            assert result.parts["d"].marks == 2, (q.text, reply)


def test_follow_through_into_last_part_only(questions):
    for q in questions[:80]:
        theta = int(re.search(r"Angle B[AC][CA] is (\d+)°", q.stem).group(1))
        c_wrong = q.parts[2].distractors[-1]
        d = float(c_wrong) / math.tan(math.radians(theta))
        canon = _canonical(q)
        result = grade(_SKILL, q, {**canon, "c": c_wrong, "d": _dp1(d)})
        assert result.parts["c"].marks == 0
        assert result.parts["d"].marks == 2, (q.text, c_wrong)


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "999.9", "b": "1.1", "c": "2.2", "d": "3.3"})
        assert result.marks == 0 and not result.correct


def test_answer_space(questions):
    tuples = Counter(q.answer_text for q in questions)
    assert len(tuples) >= 100
    assert max(tuples.values()) / len(questions) <= 0.05


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
