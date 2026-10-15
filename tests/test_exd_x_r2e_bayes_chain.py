"""Tests for the multi-part exam source of `ucv1-bayes-finite-counts` (`exd_x_r2e_bayes_chain`): total probability,
Bayes' rule, a two-item draw without replacement and an expected count, on one batch from two machines."""

from __future__ import annotations

import collections
import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2e_bayes_chain"
SKILL_ID = "ucv1-bayes-finite-counts"
HARD = ("bayes_finite_counts", {})
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _facts(q):
    total = int(re.search(r"batch of (\d+)", q.stem).group(1))
    r, t = (int(v) for v in re.search(r"ratio (\d+) : (\d+)", q.stem).groups())
    (a1, b1), (a2, b2) = ((int(x), int(y)) for x, y in re.findall(r"(\d+) out of every (\d+)", q.stem))
    k = int(re.search(r"(\d+) of them are chosen", q.parts[3].prompt).group(1))
    return total, r, t, Fraction(a1, b1), Fraction(a2, b2), k


def _text(f: Fraction) -> str:
    return str(f)


def oracle(q) -> dict[str, str]:
    """Re-derive every part's answer from the stem by counting the items."""
    total, r, t, p1, p2, k = _facts(q)
    n1, n2 = Fraction(total * r, r + t), Fraction(total * t, r + t)
    assert n1.denominator == n2.denominator == 1
    f1, f2 = n1 * p1, n2 * p2
    assert f1.denominator == f2.denominator == 1
    faulty = f1 + f2
    return {
        "a": _text(faulty / total),
        "b": _text(f1 / faulty),
        "c": _text(faulty * (faulty - 1) / (total * (total - 1))),
        "d": _text(k * f1 / faulty),
    }


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


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


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


def test_follow_through_from_a_wrong_total_probability(questions):
    """A wrong (a) carried into Bayes' rule (b), the two-item draw (c) and the expected count (d)."""
    checked = 0
    for q in questions[:150]:
        total, r, t, p1, _p2, k = _facts(q)
        joint = Fraction(total * r, (r + t) * total) * p1
        a_wrong = Fraction(oracle(q)["a"]) + Fraction(1, 20)
        b_wrong = joint / a_wrong
        faulty_wrong = total * a_wrong
        c_wrong = faulty_wrong * (faulty_wrong - 1) / (total * (total - 1))
        d_wrong = k * b_wrong
        reply = {"a": _text(a_wrong), "b": _text(b_wrong), "c": _text(c_wrong), "d": _text(d_wrong)}
        if not (0 < b_wrong < 1 and 0 < c_wrong < 1) or len(set(reply.values())) < 4:
            continue
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bcd":
            assert result.parts[label].marks == 2 and result.parts[label].follow_through, (q.text, label, reply)
        assert result.marks == 6 and not result.correct
        checked += 1
    assert checked >= 100


def test_follow_through_from_a_wrong_posterior(questions):
    for q in questions[:100]:
        canon = _canonical(q)
        k = _facts(q)[5]
        b_wrong = Fraction(canon["b"]) / 2
        d_wrong = k * b_wrong
        result = grade(_SKILL, q, {**canon, "b": _text(b_wrong), "d": _text(d_wrong)})
        assert result.parts["b"].marks == 0
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through


def test_all_wrong_earns_zero(questions):
    for q in questions[:50]:
        result = grade(_SKILL, q, {"a": "7", "b": "7777", "c": "9999", "d": "99999"})
        assert result.marks == 0 and not result.correct
