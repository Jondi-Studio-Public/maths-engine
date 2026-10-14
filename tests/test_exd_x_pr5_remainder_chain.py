"""Tests the exam source of `fraction-of-quantity`: a fraction of the remainder, as a three-part count.

The key of every part is re-derived from the rendered stem with independent code (exact fractions), and a wrong
earlier answer carried correctly into a later part is checked to earn that part's marks.
"""

from __future__ import annotations

import collections
import re
from fractions import Fraction

import pytest

import mathgen.generators.exd_x_pr5_remainder_chain  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_pr5_remainder_chain"
SKILL_ID = "fraction-of-quantity"
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
    text = q.stem
    f1, f2 = (Fraction(int(a), int(b)) for a, b in re.findall(r"(\d+)/(\d+)", text))
    total = int(re.search(r"(?:has|sold|packed) (\d+)", text).group(1))
    rest = total - total * f1
    second = rest * f2
    return {"a": str(rest), "b": str(second), "c": str(rest - second)}


def test_shape_and_marks(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [1, 2, 1]
        assert [p.answer_kind for p in q.parts] == ["number"] * 3
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


def test_follow_through_from_a_wrong_remainder(questions):
    for q in questions:
        wrong = int(q.parts[0].distractors[0])
        f2 = Fraction(*(int(v) for v in re.findall(r"(\d+)/(\d+)", q.stem)[1]))
        b = wrong * f2
        if b.denominator != 1:
            continue  # a learner's non-whole count is not the case under test
        result = grade(_SKILL, q, {"a": str(wrong), "b": str(b), "c": str(wrong - b)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {"a": "9999", "b": "5000", "c": "6000"})
        assert result.marks == 0 and not result.correct
