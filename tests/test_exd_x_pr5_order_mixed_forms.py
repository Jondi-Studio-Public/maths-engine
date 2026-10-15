"""Tests the exam source of `f10-fraction-equivalence-numberline`: a fraction, decimals and percentages ordered together, then a distance gap.

The key of every part is re-derived from the rendered stem with independent code (exact fractions), and a wrong
earlier answer carried correctly into a later part is checked to earn that part's marks.
"""

from __future__ import annotations

import collections
import re
from fractions import Fraction

import pytest

import mathgen.generators.exd_x_pr5_order_mixed_forms  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_pr5_order_mixed_forms"
SKILL_ID = "f10-fraction-equivalence-numberline"
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


def _value(item):
    item = item.strip()
    if item.endswith("%"):
        return Fraction(item[:-1]) / 100
    return Fraction(item)


def oracle(q):
    (total,) = re.search(r"is (\d+) (?:kilo)?metres", q.stem).groups()
    shown = q.stem.split(": ", 1)[1].rstrip(".")
    items = [piece.rsplit(" ", 1)[1] for piece in shown.split(", ")]
    ordered = sorted(items, key=_value)
    gap = (_value(ordered[-1]) - _value(ordered[0])) * int(total)
    return {"a": ", ".join(ordered), "b": str(gap)}


def test_shape_and_marks(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b"]
        assert [p.marks for p in q.parts] == [2, 1]
        assert [p.answer_kind for p in q.parts] == ["ordered_list", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, False]


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
            assert len(p.distractors) >= 2, (q.text, p.label, p.distractors)
            for d in p.distractors:
                reply = {**canon, p.label: d}
                result = grade(_SKILL, q, reply)
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_distractors_are_tagged(questions):
    for q in questions[:60]:
        for p in q.parts:
            assert set(p.distractors) <= set(p.option_notes), (q.text, p.label)
            assert all(d in p.distractor_values for d in p.distractors)


def test_part_b_is_independent_of_a_wrong_order(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {"a": q.parts[0].distractors[0], "b": q.parts[1].answer_text})
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 1


def test_forms_are_mixed(questions):
    for q in questions:
        shown = q.stem.split(": ", 1)[1]
        assert "%" in shown and "/" in shown and re.search(r"\b0\.\d", shown), q.text


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {"a": "1, 2, 3, 4", "b": "9999"})
        assert result.marks == 0 and not result.correct
