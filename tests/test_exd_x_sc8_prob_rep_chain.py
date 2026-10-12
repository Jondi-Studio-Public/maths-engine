"""Exam source of `probability-representations`: a decimal, a fraction and a percentage probability combined."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_sc8_prob_rep_chain"
SKILL_ID = "probability-representations"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _round(value: Fraction, places: int) -> Fraction:
    scale = 10**places
    return Fraction(int(value * scale + Fraction(1, 2)), scale)


def _fmt(value: Fraction, places: int = 0) -> str:
    return f"{float(_round(value, places)):.{places}f}"


def oracle(q):
    """Re-derive every part's answer from the stem alone."""
    n = int(re.search(r"(\d+) students", q.text).group(1))
    pw = Fraction(re.search(r" is (0\.\d+)\.", q.text).group(1))
    num, den = (int(v) for v in re.search(r" is (\d+)/(\d+)\.", q.text).groups())
    pct = Fraction(re.search(r" is (\d+(?:\.\d+)?)%\.", q.text).group(1))
    key_a = 100 - 100 * pw - Fraction(100 * num, den) - pct
    key_b = n * key_a / 100
    key_c = 100 * pw / (1 - key_a / 100)
    return {"a": _fmt(key_a), "b": _fmt(key_b), "c": _fmt(key_c, 1)}, n, pw, key_a


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(120)]


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages) for s in exam] == [(TOPIC, ["exam"])]


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [3, 1, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    easy = {prompt_shape(make("probability_representation", seed=s)) for s in range(60)}
    assert len(exam) >= 3
    assert exam - easy


def test_no_text_defects(questions):
    for q in questions[:60]:
        assert find_text_defects(q) == []


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        expected = oracle(q)[0]
        assert expected == {p.label: p.answer_text for p in q.parts}, (seed, q.text)


def test_answer_space():
    keys = [tuple(p.answer_text for p in _exam(s).parts) for s in range(300)]
    assert len(set(keys)) >= 100
    assert max(keys.count(k) for k in set(keys)) <= 12


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6, (q.text, q.answer_text)


def test_distractors_rejected_and_tagged(questions):
    for q in questions:
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            assert all(d in p.distractor_values for d in p.distractors)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_percentage(questions):
    for q in questions:
        _, n, pw, _ = oracle(q)
        wrong = Fraction(q.parts[0].distractors[0])
        if not 0 < wrong < 100:
            continue
        reply = {"a": q.parts[0].distractors[0], "b": _fmt(n * wrong / 100), "c": _fmt(100 * pw / (1 - wrong / 100), 1)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through, (q.text, reply)
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (q.text, reply)
        assert not result.correct


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        assert grade(_SKILL, q, {"a": "999", "b": "7777", "c": "888"}).marks == 0


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
