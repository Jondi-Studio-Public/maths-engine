"""Tests for the multi-part exam source of `discrete-expected-value` (`exd_x_r2e_expval_chain`): the constant c,
E(X), E(X^2) and Var(X) of a distribution given in multiples of c."""

from __future__ import annotations

import collections
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2e_expval_chain"
SKILL_ID = "discrete-expected-value"
HARD = ("discrete_expected_value", {})
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _weights(q):
    """(values, weights) read from the stem: either P(X = x) = c(x + s) or a list of multiples of c."""
    formula = re.search(r"takes the values ([-\d, ]+?), and P\(X = x\) = c\(x \+ (\d+)\)", q.stem)
    if formula:
        values = [int(v) for v in formula.group(1).split(",")]
        return values, [v + int(formula.group(2)) for v in values]
    pairs = re.findall(r"P\(X = (-?\d+)\) = (\d*)c", q.stem)
    return [int(v) for v, _ in pairs], [int(w) if w else 1 for _, w in pairs]


def _text(f: Fraction) -> str:
    return str(f)


def oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the stem alone, with exact fractions."""
    values, weights = _weights(q)
    c = Fraction(1, sum(weights))
    mean = sum(v * w * c for v, w in zip(values, weights))
    second = sum(v * v * w * c for v, w in zip(values, weights))
    return {"a": _text(c), "b": _text(mean), "c": _text(second), "d": _text(second - mean * mean)}


def test_wired_to_the_skill():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.params, s.stages) for s in exam] == [(TOPIC, {"tier": "exam"}, ["exam"])]


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 2, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)


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
        assert result.correct and result.marks == result.max_marks == 8


def test_distractors_rejected_and_tagged(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                assert p.option_notes.get(d) in CATALOGUE, (p.label, d)
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_a_wrong_constant(questions):
    """A wrong c used correctly earns E(X), E(X^2) and Var(X)."""
    checked = 0
    for q in questions[:150]:
        values, weights = _weights(q)
        c_wrong = Fraction(1, sum(weights) + 2)
        mean = sum(v * w * c_wrong for v, w in zip(values, weights))
        second = sum(v * v * w * c_wrong for v, w in zip(values, weights))
        reply = {"a": _text(c_wrong), "b": _text(mean), "c": _text(second), "d": _text(second - mean * mean)}
        if len(set(reply.values())) < 4:
            continue  # a reply that only repeats an earlier answer it uses earns no follow-through
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == 2 and result.parts[label].follow_through, (q.text, label, reply)
        assert result.marks == 6 and not result.correct
        checked += 1
    assert checked >= 100


def test_follow_through_from_a_wrong_mean(questions):
    for q in questions[:100]:
        canon = _canonical(q)
        mean_wrong = Fraction(canon["b"]) + 1
        var_wrong = Fraction(canon["c"]) - mean_wrong * mean_wrong
        if var_wrong == 0:
            continue
        result = grade(_SKILL, q, {**canon, "b": _text(mean_wrong), "d": _text(var_wrong)})
        assert result.parts["b"].marks == 0
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through


def test_all_wrong_earns_zero(questions):
    for q in questions[:50]:
        result = grade(_SKILL, q, {"a": "7", "b": "7777", "c": "9999", "d": "99999"})
        assert result.marks == 0 and not result.correct


def test_deterministic():
    assert make(TOPIC, seed=11).to_dict() == make(TOPIC, seed=11).to_dict()
