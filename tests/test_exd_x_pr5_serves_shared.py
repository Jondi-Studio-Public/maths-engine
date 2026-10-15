"""Tests the exam source of `f10-unit-fractions-combine-whole`: unit-fraction pieces combined, the whole completed, the remainder shared equally.

The key of every part is re-derived from the rendered stem with independent code (exact fractions), and a wrong
earlier answer carried correctly into a later part is checked to earn that part's marks.
"""

from __future__ import annotations

import collections
import re
from fractions import Fraction

import pytest

import mathgen.generators.exd_x_pr5_serves_shared  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_pr5_serves_shared"
SKILL_ID = "f10-unit-fractions-combine-whole"
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _frac(text):
    return Fraction(text)


def oracle(q):
    pieces = int(re.search(r"cut into (\d+)", q.stem).group(1))
    taken = [int(v) for v in re.findall(r"takes (\d+)", q.stem)]
    k = int(re.search(r"among the (\d+)", q.stem).group(1))
    taken_f = Fraction(sum(taken), pieces)
    left = 1 - taken_f
    return {"a": str(taken_f), "b": str(left), "c": str(left / k)}


def test_shape_and_marks(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 1, 2]
        assert [p.answer_kind for p in q.parts] == ["fraction"] * 3
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_oracle_agrees_with_the_key_over_300_seeds(questions):
    bad = [(q.text, oracle(q), _canonical(q)) for q in questions if oracle(q) != _canonical(q)]
    assert bad == []


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks, q.text


def test_every_distractor_rejected_over_300_seeds(questions):
    for q in questions[:300]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 3, (q.text, p.label, p.distractors)
            for d in p.distractors:
                reply = {**canon, p.label: d}
                result = grade(_SKILL, q, reply)
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_distractors_are_tagged(questions):
    for q in questions[:60]:
        for p in q.parts:
            assert set(p.distractors) <= set(p.option_notes), (q.text, p.label)
            assert all(d in p.distractor_values for d in p.distractors)


def test_follow_through_from_a_wrong_total(questions):
    checked = 0
    for q in questions:
        k = int(re.search(r"among the (\d+)", q.stem).group(1))
        a = Fraction(q.parts[0].distractors[0])
        b = 1 - a
        c = b / k
        if not 0 < b < 1 or b == a or c == b:
            continue
        result = grade(_SKILL, q, {"a": q.parts[0].distractors[0], "b": str(b), "c": str(c)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through
        checked += 1
    assert checked >= 20


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {"a": "1/99", "b": "1/98", "c": "1/97"})
        assert result.marks == 0 and not result.correct
