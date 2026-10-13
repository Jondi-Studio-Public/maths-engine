"""Tests the exam source of `f10-equivalent-fractions-related-denominators`: two stopping fractions of a marked trail, converted to markers, then the gap as a fraction.

The key of every part is re-derived from the rendered stem with independent code (exact fractions), and a wrong
earlier answer carried correctly into a later part is checked to earn that part's marks.
"""

from __future__ import annotations

import collections
import re
from fractions import Fraction

import pytest

import mathgen.generators.exd_x_pr5_trail_markers  # noqa: F401  (registers the topic)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_pr5_trail_markers"
SKILL_ID = "f10-equivalent-fractions-related-denominators"
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
    length = int(re.search(r"It is (\d+) kilometres", q.stem).group(1))
    f1, f2 = (Fraction(int(a), int(b)) for a, b in re.findall(r"(\d+)/(\d+)", q.stem))
    return {"a": str(f1 * length), "b": str(f2 * length), "c": str(abs(f2 - f1))}


def test_source_wired_last_with_exam_stage():
    src = _SKILL.sources[-1]
    assert (src.topic, src.difficulty, src.stages, src.params) == (TOPIC, "exam", ["exam"], {"tier": "exam"})
    assert sum(1 for s in _SKILL.sources if s.difficulty == "exam") == 1


def test_shape_and_marks(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [1, 1, 2]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "fraction"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True]


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


def test_follow_through_from_wrong_markers(questions):
    checked = 0
    for q in questions:
        length = int(re.search(r"It is (\d+) kilometres", q.stem).group(1))
        a, b = int(q.parts[0].distractors[0]), int(q.parts[1].distractors[0])
        forward = int(q.parts[1].answer_text) > int(q.parts[0].answer_text)  # who stops further on
        c = Fraction(b - a if forward else a - b, length)
        if not 0 < c < 1 or str(c) == q.parts[2].answer_text:
            continue
        result = grade(_SKILL, q, {"a": str(a), "b": str(b), "c": str(c)})
        assert result.parts["a"].marks == 0 and result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through, (q.text, a, b, c)
        checked += 1
    assert checked >= 20


def test_all_wrong_earns_zero(questions):
    for q in questions[:60]:
        result = grade(_SKILL, q, {"a": "999", "b": "998", "c": "1/99"})
        assert result.marks == 0 and not result.correct


def test_deterministic():
    assert _exam(7).to_dict() == _exam(7).to_dict()
