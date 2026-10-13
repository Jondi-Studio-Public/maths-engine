"""Tests the exam source of `f10-fraction-add-subtract-equivalent`: three unrelated fractions of one whole: added, taken from the whole, then counted.

The key of every part is re-derived from the rendered stem with independent code (exact fractions), and a wrong
earlier answer carried correctly into a later part is checked to earn that part's marks.
"""

from __future__ import annotations

import collections
import re
from fractions import Fraction

import pytest

import mathgen.generators.exd_x_pr5_book_fractions  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_pr5_book_fractions"
SKILL_ID = "f10-fraction-add-subtract-equivalent"
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
    fracs = [Fraction(int(a), int(b)) for a, b in re.findall(r"(\d+)/(\d+)", q.stem)]
    total = int(re.search(r"\d+", re.sub(r"\d+/\d+", "", q.stem)).group())
    first_two = fracs[0] + fracs[1]
    left = 1 - first_two - fracs[2]
    return {"a": str(first_two), "b": str(left), "c": str(left * total)}


def test_source_wired_last_with_exam_stage():
    src = _SKILL.sources[-1]
    assert (src.topic, src.difficulty, src.stages, src.params) == (TOPIC, "exam", ["exam"], {"tier": "exam"})
    assert sum(1 for s in _SKILL.sources if s.difficulty == "exam") == 1


def test_shape_and_marks(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 2, 1]
        assert [p.answer_kind for p in q.parts] == ["fraction", "fraction", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_no_text_defects():
    for seed in range(60):
        q = _exam(seed)
        assert find_text_defects(q) == [], (seed, q.text)


def test_prompt_variety_and_new_shape_over_hard_tier():
    hard = next(s for s in _SKILL.sources if s.difficulty == "hard")
    exam_shapes = {prompt_shape(_exam(s)) for s in range(60)}
    hard_shapes = {prompt_shape(make(hard.topic, seed=s, **hard.params)) for s in range(60)}
    assert len(exam_shapes) >= 3, exam_shapes
    assert exam_shapes - hard_shapes


def test_answer_space(questions):
    counts = collections.Counter(q.answer_text for q in questions)
    assert len(counts) >= 100, len(counts)
    assert max(counts.values()) <= 0.05 * len(questions), counts.most_common(2)


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


def test_follow_through_from_a_wrong_sum(questions):
    checked = 0
    for q in questions:
        fracs = [Fraction(int(a), int(b)) for a, b in re.findall(r"(\d+)/(\d+)", q.stem)]
        total = int(re.search(r"\d+", re.sub(r"\d+/\d+", "", q.stem)).group())
        a = Fraction(q.parts[0].distractors[0])
        b = 1 - a - fracs[2]
        c = b * total
        if not (0 < b < 1) or c.denominator != 1 or b == a:  # repeating an earlier value earns no follow-through
            continue
        result = grade(_SKILL, q, {"a": q.parts[0].distractors[0], "b": str(b), "c": str(c)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through
        checked += 1
    assert checked >= 20


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {"a": "1/99", "b": "1/98", "c": "5000"})
        assert result.marks == 0 and not result.correct


def test_deterministic():
    assert _exam(7).to_dict() == _exam(7).to_dict()
