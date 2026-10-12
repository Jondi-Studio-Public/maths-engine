"""Tests the multi-part exam source of `ucv1-chinese-remainder-two-congruences`: the inverse of m1 modulo m2, the multiplier t, then the least count."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u3_crt_leftovers"
SKILL_ID = "ucv1-chinese-remainder-two-congruences"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_HARD = next(s for s in _SKILL.sources if s.difficulty == "hard")
LABELS = ["a", "b", "c"]
MARKS = [2, 3, 1]


def _parse(q):
    m1, r1, m2, r2 = (
        int(v)
        for v in re.search(
            r"packed (\d+) to a \w+, (\d+) (?:is|are) left over\. When the same \w+ are packed (\d+) to a \w+ instead, "
            r"(\d+) (?:is|are) left over",
            q.stem,
        ).groups()
    )
    return m1, r1, m2, r2


def _num(text):
    return Fraction(str(text).strip())


def _round(value, places):
    """Round half up, once, from an exact value."""
    scale = 10**places
    return Fraction(int((Fraction(value) * scale + Fraction(1, 2)).__floor__()), scale)


def _tie(value, places=2):
    """True when `value` sits exactly on a rounding boundary, where graders may legitimately round either way."""
    return (Fraction(value) * 10**places) % 1 == Fraction(1, 2)


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in range(300)]


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages, s.params) for s in exam] == [(TOPIC, ["exam"], {"tier": "exam"})]


def test_shape_and_marks(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == LABELS
        assert [p.marks for p in q.parts] == MARKS
        assert all(p.answer_kind == "number" for p in q.parts)
        assert all(len(p.distractors) >= 2 for p in q.parts)
        assert all(p.follow_through is not None for p in q.parts[1:])


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard = {prompt_shape(make(_HARD.topic, seed=s, difficulty="hard", **_HARD.params)) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(make(TOPIC, seed=seed)) == []


def test_answer_space(questions):
    texts = [q.answer_text for q in questions]
    assert len(set(texts)) >= 100
    assert max(texts.count(t) for t in set(texts)) <= 15


def test_key_rederived_from_prompt(questions):
    for q in questions:
        m1, r1, m2, r2 = _parse(q)
        n = next(x for x in range(m1 * m2) if x % m1 == r1 and x % m2 == r2)
        assert [int(p.answer_text) for p in q.parts] == [pow(m1, -1, m2), (n - r1) // m1, n], q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:120]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(MARKS), q.text


def test_distractors_rejected(questions):
    for q in questions[:120]:
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_inverse(questions):
    for q in questions:
        m1, r1, m2, r2 = _parse(q)
        k = int(q.parts[0].distractors[0])
        t = k * (r2 - r1) % m2
        n = r1 + m1 * t
        if k == t or t == n or k == n:
            continue  # a reply that repeats an earlier reply earns no follow-through by design
        result = grade(_SKILL, q, {"a": q.parts[0].distractors[0], "b": str(t), "c": str(n)})
        assert result.parts["a"].marks == 0
        if t != 0:
            assert result.parts["b"].marks == 3, (q.text, t)
            assert result.parts["b"].follow_through is (str(t) != q.parts[1].answer_text), (q.text, t)
            assert result.parts["c"].marks == 1, (q.text, n)


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {p.label: "987654.321" for p in q.parts})
        assert result.marks == 0 and not result.correct


def test_deterministic():
    assert make(TOPIC, seed=7).to_dict() == make(TOPIC, seed=7).to_dict()


def test_chain_is_not_copyable():
    """No part repeats an earlier answer or follows through as a restatement, and junk replies earn nothing."""
    from tests.exd_junk_probe import assert_chain_is_not_copyable

    assert_chain_is_not_copyable("exd_x_u3_crt_leftovers", "ucv1-chinese-remainder-two-congruences")
