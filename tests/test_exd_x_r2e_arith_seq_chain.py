"""Tests for the multi-part exam source of `senior-use-arithmetic-sequences-to-model-and-analyse-practical`
(`exd_x_r2e_arith_seq_chain`): common difference, first term, series total and the first period past a threshold."""

from __future__ import annotations

import collections
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2e_arith_seq_chain"
SKILL_ID = "senior-use-arithmetic-sequences-to-model-and-analyse-practical"
HARD = ("p12b_arith_seq_general", {})
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _half_up(x: Fraction) -> int:
    return int(x + Fraction(1, 2)) if x >= 0 else -int(-x + Fraction(1, 2))


def _facts(q):
    (vp, p), (vq, q2) = re.findall(r"\$?(\d+)(?: km| litres| copies)? (?:on|in) (?:day|week|month) (\d+)", q.stem)
    n_terms = int(re.search(r"first (\d+)", q.parts[2].prompt).group(1))
    limit = int(re.search(r"(?:more|fewer|less) than \$?(\d+)", q.parts[3].prompt).group(1))
    return int(vp), int(p), int(vq), int(q2), n_terms, limit


def _first_past(a: int, d: int, limit: int) -> int:
    n = 1
    while not (a + (n - 1) * d > limit if d > 0 else a + (n - 1) * d < limit):
        n += 1
    return n


def oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the stem and the part prompts alone."""
    vp, p, vq, q2, n_terms, limit = _facts(q)
    d = Fraction(vq - vp, q2 - p)
    a = vp - (p - 1) * d
    total = sum(a + k * d for k in range(n_terms))
    return {"a": str(d), "b": str(a), "c": str(total), "d": str(_first_past(int(a), int(d), limit))}


def test_wired_to_the_skill():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.params, s.stages) for s in exam] == [(TOPIC, {"tier": "exam"}, ["exam"])]


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 1, 2, 3]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompt_shapes_add_to_the_hard_tier():
    shapes = {prompt_shape(make(TOPIC, seed=s)) for s in range(60)}
    hard = {prompt_shape(make(HARD[0], seed=s, difficulty="hard", **HARD[1])) for s in range(60)}
    assert len(shapes) >= 3, shapes
    assert shapes - hard


def test_no_text_defects(questions):
    for q in questions[:60]:
        assert find_text_defects(q) == []


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_answer_space(questions):
    tuples = collections.Counter(tuple(p.answer_text for p in q.parts) for q in questions)
    assert len(tuples) >= 100
    assert max(tuples.values()) / len(questions) <= 0.05


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 7


def test_distractors_rejected_and_tagged(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (p.label, d)
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_a_wrong_difference(questions):
    """A wrong common difference used correctly earns (b), (c) and (d)."""
    for q in questions[:150]:
        vp, p, _vq, _q2, n_terms, limit = _facts(q)
        d_wrong = int(q.parts[0].answer_text) + 1
        if d_wrong == 0:
            continue
        a_wrong = vp - (p - 1) * d_wrong
        total_wrong = n_terms * (2 * a_wrong + (n_terms - 1) * d_wrong) // 2
        past = (limit - a_wrong) // d_wrong + 2
        if past in (d_wrong, a_wrong):
            continue  # a reply that only repeats an earlier answer it uses earns no follow-through
        reply = {"a": str(d_wrong), "b": str(a_wrong), "c": str(total_wrong), "d": str(past)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through
        assert result.parts["d"].marks == 3, (q.text, reply)
        assert result.marks == 6 and not result.correct


def test_follow_through_from_a_wrong_first_term(questions):
    for q in questions[:60]:
        _vp, _p, _vq, _q2, n_terms, limit = _facts(q)
        canon = _canonical(q)
        d = int(canon["a"])
        a_wrong = int(canon["b"]) + 7
        total_wrong = n_terms * (2 * a_wrong + (n_terms - 1) * d) // 2
        past = (limit - a_wrong) // d + 2
        if past in (d, a_wrong):
            continue
        result = grade(_SKILL, q, {**canon, "b": str(a_wrong), "c": str(total_wrong), "d": str(past)})
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through
        assert result.parts["d"].marks == 3


def test_all_wrong_earns_zero(questions):
    for q in questions[:50]:
        result = grade(_SKILL, q, {"a": "1000", "b": "7", "c": "9", "d": "99"})
        assert result.marks == 0 and not result.correct


def test_deterministic():
    assert make(TOPIC, seed=11).to_dict() == make(TOPIC, seed=11).to_dict()
