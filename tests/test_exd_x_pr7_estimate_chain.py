"""Exam tier of `f10-estimate-counts-and-check-reasonableness`: estimate, exact total, size of the error."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import prompt_shape

TOPIC = "exd_x_pr7_estimate_chain"
SKILL_ID = "f10-estimate-counts-and-check-reasonableness"
HARD = ("estimate_counts_and_check_reasonableness", {"tier": "hard"})
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _round(n, place):
    return int((Fraction(n, place) + Fraction(1, 2)) // 1) * place


def oracle(q):
    a, b = (int(x) for x in re.findall(r"\d+", q.stem)[:2])
    place = {"ten": 10, "hundred": 100}[re.search(r"nearest (ten|hundred)\.$", q.stem).group(1)]
    est = _round(a, 10) * _round(b, place)
    return {"a": str(est), "b": str(a * b), "c": str(abs(a * b - est))}


@pytest.fixture(scope="module")
def questions():
    return [_q(s) for s in range(300)]


def _canon(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    assert [s.topic for s in _SKILL.sources if s.difficulty == "exam"] == [TOPIC]
    exam = next(s for s in _SKILL.sources if s.difficulty == "exam")
    assert exam.stages == ["exam"]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[0], seed=s, **HARD[1])) for s in range(60)}
    assert len(exam) >= 3 and exam - hard


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is None for p in q.parts] == [True, True, False]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_key_matches_independent_derivation(questions):
    bad = [(q.seed, oracle(q), _canon(q)) for q in questions if oracle(q) != _canon(q)]
    assert bad == []


def test_grader_accepts_key_and_rejects_distractors(questions):
    for q in questions[:150]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5
        for p in q.parts:
            assert len(p.distractors) >= 2
            for d in p.distractors:
                assert grade(_SKILL, q, {**_canon(q), p.label: d}).parts[p.label].marks == 0, (q.seed, p.label, d)


def test_distractors_are_tagged(questions):
    for q in questions[:60]:
        for p in q.parts:
            assert p.distractor_values and set(p.option_notes) <= set(p.distractors)


def test_follow_through_from_wrong_estimate(questions):
    for q in questions[:100]:
        wrong = int(q.parts[0].distractors[0])
        exact = int(q.parts[1].answer_text)
        reply = {"a": str(wrong), "b": q.parts[1].answer_text, "c": str(abs(exact - wrong))}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks and result.parts["c"].follow_through
        assert not result.correct


def test_follow_through_from_wrong_exact_total(questions):
    for q in questions[:100]:
        wrong = int(q.parts[1].distractors[0])
        est = int(q.parts[0].answer_text)
        result = grade(_SKILL, q, {"a": str(est), "b": str(wrong), "c": str(abs(wrong - est))})
        assert result.parts["b"].marks == 0 and result.parts["c"].marks == 1
