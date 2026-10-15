"""Tests the exam source of `f10-percentage-equivalence-comparison`: a percentage win rate converted to a fraction and used to compare two teams.

The key of every part is re-derived from the rendered stem with independent code (exact fractions), and a wrong
earlier answer carried correctly into a later part is checked to earn that part's marks.
"""

from __future__ import annotations

import collections
import re
from fractions import Fraction

import pytest

import mathgen.generators.exd_x_pr5_season_rates  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_pr5_season_rates"
SKILL_ID = "f10-percentage-equivalence-comparison"
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
    wins1, games1, pct, games2 = (
        int(v)
        for v in re.search(r"won (\d+) of their (\d+) games\. The \w+ won (\d+)% of their (\d+) games", q.stem).groups()
    )
    rate = Fraction(pct, 100)
    wins2 = rate * games2
    return {"a": str(rate), "b": str(wins2), "c": str(wins1 - wins2)}


def test_shape_and_marks(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [1, 2, 1]
        assert [p.answer_kind for p in q.parts] == ["fraction", "number", "number"]
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


def test_follow_through_from_a_wrong_fraction(questions):
    checked = 0
    for q in questions:
        wins1, pct, games2 = (
            int(v)
            for v in re.search(
                r"won (\d+) of their \d+ games\. The \w+ won (\d+)% of their (\d+) games", q.stem
            ).groups()
        )
        a = 1 - Fraction(pct, 100)  # the complement: a distractor value that gives a whole number of games
        b = a * games2
        if wins1 - b <= 0 or wins1 - b == b:
            continue
        reply = {"a": str(a), "b": str(b), "c": str(wins1 - b)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through, (q.text, reply)
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through
        checked += 1
    assert checked >= 20


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {"a": "1/99", "b": "5000", "c": "6000"})
        assert result.marks == 0 and not result.correct
