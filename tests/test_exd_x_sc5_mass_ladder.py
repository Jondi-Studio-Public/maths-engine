"""Tests the multi-part exam tier of `senior-eview-metric-units-of-mass-and-weight-their`: kilograms to milligrams, a tablet count, and grams in a pack."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_sc5_mass_ladder"
SKILL_ID = "senior-eview-metric-units-of-mass-and-weight-their"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
MARKS = [2, 2, 1]
FT_LABELS = [False, True, False]
TOP_SHARE = 0.05


def _exam(seed):
    return make(TOPIC, seed=seed)


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _frac(text):
    return Fraction(text)


def _inputs(q):
    k = int(re.search(r"(\d+) kg", q.stem).group(1))
    m = int(re.search(r"(\d+) mg", q.stem).group(1))
    p = int(re.search(r"(?:packs of|holds) (\d+)", q.stem).group(1))
    return k, m, p


def oracle(q):
    k, m, p = _inputs(q)
    mg = k * 1000 * 1000
    text_c = Fraction(p * m, 1000)
    return {"a": str(mg), "b": str(Fraction(mg, m)), "c": str(text_c)}


def follow_through_cases(q):
    k, m, p = _inputs(q)
    a = k * 10**6 + 10**6  # one kilogram too many
    reply = {"a": str(a), "b": str(a // m)}
    return [(reply, "b", "a")]


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(300)]


def test_shape(questions):
    for q in questions[:100]:
        assert [p.label for p in q.parts] == list("abcd")[: len(MARKS)]
        assert [p.marks for p in q.parts] == MARKS
        assert [p.follow_through is not None for p in q.parts] == FT_LABELS
        assert all(p.answer_kind == "number" for p in q.parts)


def test_key_rederived_from_prompt(questions):
    for q in questions:
        got = oracle(q)
        want = _canonical(q)
        assert set(got) == set(want)
        for label in want:
            assert _frac(got[label]) == _frac(want[label]), (q.text, label, got[label], want[label])


def test_composed_answer_full_marks(questions):
    for q in questions[:120]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == sum(MARKS), q.answer_text


def test_distractors_rejected(questions):
    shown = 0
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert p.distractors, (q.text, p.label)
            assert len(p.distractors) == len(set(p.distractors))
            for d in p.distractors:
                shown += 1
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)
    assert shown >= 600


def test_distractors_are_tagged(questions):
    for q in questions[:100]:
        for p in q.parts:
            assert p.distractor_values, (q.text, p.label)
            assert set(p.option_notes) <= set(p.distractors)


def test_follow_through(questions):
    checked = 0
    for q in questions[:100]:
        for reply, earns, loses in follow_through_cases(q):
            result = grade(_SKILL, q, reply)
            for label in earns:
                part = result.parts[label]
                assert part.marks == part.max_marks, (q.text, reply, label)
                assert part.follow_through is True, (q.text, reply, label)
            for label in loses:
                assert result.parts[label].marks == 0, (q.text, reply, label)
            assert not result.correct
            checked += 1
    assert checked >= 100


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {"a": "987654", "b": "7654321", "c": "-31", "d": "-9"})
        assert result.marks == 0 and not result.correct
