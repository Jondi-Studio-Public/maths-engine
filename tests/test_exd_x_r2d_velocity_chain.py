"""Tests the four-part exam chain on an aircraft's velocity in a wind (ground speed, flight time, northward distance, bearing)."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_r2d_velocity_chain"
SKILL_ID = "senior-examine-examples-of-vectors-including-displacement-and-velocity"
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

LABELS = ["a", "b", "c", "d"]
MARKS = [3, 2, 2, 2]
FT_FLAGS = [False, True, True, False]


def _r(x: Fraction, dp: int) -> str:
    scaled = x * 10**dp
    n = math.floor(abs(scaled) + Fraction(1, 2)) * (1 if scaled >= 0 else -1)
    return f"{n / 10**dp:.{dp}f}"


def _facts(q):
    vecs = re.findall(r"(-?\d+)i ([+-]) (\d+)j", q.stem)
    (ai, sj, aj), (wi, sw, wj) = vecs
    ai, aj = int(ai), int(aj) * (1 if sj == "+" else -1)
    wi, wj = int(wi), int(wj) * (1 if sw == "+" else -1)
    dist = int(re.search(r"(\d+) km from", q.stem).group(1))
    return ai, aj, wi, wj, dist


def oracle(q):
    ai, aj, wi, wj, dist = _facts(q)
    p, qq = ai + wi, aj + wj
    s = math.isqrt(p * p + qq * qq)
    assert s * s == p * p + qq * qq
    t = Fraction(dist, s)
    bearing = math.degrees(math.atan2(p, qq)) % 360
    return {"a": str(s), "b": _r(t, 2), "c": _r(qq * t, 2), "d": str(math.floor(bearing + 0.5))}


def test_follow_through_from_a_wrong_ground_speed(questions):
    for q in questions[:100]:
        ai, aj, wi, wj, dist = _facts(q)
        a_wrong = q.parts[0].distractors[0]
        b_wrong = _r(Fraction(dist) / Fraction(a_wrong), 2)
        c_wrong = _r(Fraction(aj + wj) * Fraction(b_wrong), 2)
        reply = {"a": a_wrong, "b": b_wrong, "c": c_wrong, "d": q.parts[3].answer_text}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label)
            assert result.parts[label].follow_through is True
        assert result.parts["d"].marks == 2
        assert result.marks == 6 and not result.correct


def test_a_correct_a_with_rounded_b_still_carries(questions):
    for q in questions[:100]:
        ai, aj, wi, wj, dist = _facts(q)
        b = q.parts[1].answer_text
        c = _r(Fraction(aj + wj) * Fraction(b), 2)
        result = grade(_SKILL, q, {**_canonical(q), "c": c})
        assert result.parts["c"].marks == 2


def test_the_first_part_alone_does_not_buy_the_rest(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {"a": q.parts[0].answer_text})
        assert result.marks == 3
