"""Tests the four-part exam chain on two-phase geometric change (first-phase value, second-phase value, periods to a threshold, total periods)."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_r2d_geometric_phases_chain"
SKILL_ID = "senior-use-geometric-sequences-in-contexts-involving-geometric-growth"
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
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
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


from decimal import ROUND_HALF_UP, Decimal

LABELS = ["a", "b", "c", "d"]
MARKS = [2, 2, 3, 1]
FT_FLAGS = [False, True, True, True]


def _ints(text):
    return [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", text)]


def _facts(q):
    nums = _ints(q.stem)
    if "cells" in q.stem:
        p, n1, r1, r2 = nums
    else:
        p, r1, n1, r2 = nums
    m = _ints(q.parts[1].prompt.replace("(a)", ""))[0]
    t = _ints(q.parts[2].prompt.replace("(a)", ""))[0]
    grow = not ("falls" in q.stem or "loses" in q.stem)
    return p, n1, r1, r2, m, t, grow


def _round(x: Fraction) -> int:
    return int(Decimal(x.numerator) / Decimal(x.denominator) + Decimal("0.5") if x >= 0 else 0)


def _chain(p, n1, r1, r2, m, t, grow, first=None):
    s = 1 if grow else -1
    g1, g2 = 1 + s * Fraction(r1, 100), 1 + s * Fraction(r2, 100)
    v1 = Fraction(p)
    for _ in range(n1):
        v1 *= g1
    v1 = Fraction(_round(v1)) if first is None else Fraction(first)
    v2 = v1
    for _ in range(m):
        v2 *= g2
    k, v = 0, v1
    for _ in range(500):
        if (v > t) if grow else (v < t):
            break
        v *= g2
        k += 1
    else:
        raise AssertionError("the threshold is never crossed")
    return int(v1), _round(v2), k


def oracle(q):
    p, n1, r1, r2, m, t, grow = _facts(q)
    v1, v2, k = _chain(p, n1, r1, r2, m, t, grow)
    return {"a": str(v1), "b": str(v2), "c": str(k), "d": str(n1 + k)}


def test_follow_through_from_a_wrong_first_value(questions):
    checked = 0
    for q in questions[:150]:
        p, n1, r1, r2, m, t, grow = _facts(q)
        wrong = int(q.parts[0].distractors[0])
        # the threshold must still lie ahead of the wrong starting value
        if (wrong >= t) if grow else (wrong <= t):
            continue
        v1, v2, k = _chain(p, n1, r1, r2, m, t, grow, first=wrong)
        reply = {"a": str(wrong), "b": str(v2), "c": str(k), "d": str(n1 + k)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            if reply[label] == q.parts["abcd".index(label)].answer_text or k == 0:
                continue
            assert result.parts[label].marks == MARKS["abcd".index(label)], (q.text, label, reply)
        checked += 1
    assert checked >= 60


def test_all_three_contexts_occur(questions):
    kinds = {
        "cells" in q.stem or ("machine" in q.stem) or ("investment" in q.stem or "account" in q.stem) for q in questions
    }
    assert kinds == {True}
    assert {"cells" in q.stem for q in questions} == {True, False}
    assert {"machine" in q.stem for q in questions} == {True, False}


def test_the_threshold_is_crossed_inside_a_period(questions):
    for q in questions[:100]:
        p, n1, r1, r2, m, t, grow = _facts(q)
        v1_exact = Fraction(p)
        s = 1 if grow else -1
        for _ in range(n1):
            v1_exact *= 1 + s * Fraction(r1, 100)
        rounded = _chain(p, n1, r1, r2, m, t, grow)[2]
        unrounded = _chain(p, n1, r1, r2, m, t, grow, first=v1_exact)[2]
        assert rounded == unrounded
