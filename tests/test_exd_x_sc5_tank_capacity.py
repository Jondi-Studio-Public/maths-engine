"""Tests the multi-part exam tier of `senior-estimate-volume-and-capacity-of-various-objects`: capacity in litres, a partial fill, and a top-up in millilitres."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_sc5_tank_capacity"
SKILL_ID = "senior-estimate-volume-and-capacity-of-various-objects"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
MARKS = [2, 2, 1]
FT_LABELS = [False, True, True]
TOP_SHARE = 0.05


def _exam(seed):
    return make(TOPIC, seed=seed)


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _frac(text):
    return Fraction(text)


def _r10(value):
    """A measured value rounded to the nearest 10 cm (the draw never produces a half)."""
    return int(Fraction(value, 10) + Fraction(1, 2)) * 10


def _dims(q):
    shown = q.stem.split(" To estimate")[0]
    l, w, h = (_r10(int(v)) for v in re.search(r"(\d+) cm long, (\d+) cm wide and (\d+) cm", shown).groups())
    d = _r10(int(re.findall(r"(\d+) cm", shown)[-1]))
    start = int(re.search(r"holds (\d+) L", q.parts[2].prompt).group(1))
    return l, w, h, d, start


def oracle(q):
    l, w, h, d, start = _dims(q)
    capacity = Fraction(l * w * h, 1000)
    held = Fraction(l * w * d, 1000)
    return {"a": str(capacity), "b": str(held), "c": str((held - start) * 1000)}


def follow_through_cases(q):
    l, w, h, d, start = _dims(q)
    a = Fraction(l * w * h, 1000) + h  # a capacity that is wrong by a whole height
    b = a * d / h
    reply = {"a": str(a), "b": str(b), "c": str((b - start) * 1000)}
    return [(reply, "bc", "a")]


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(300)]


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_shape(questions):
    for q in questions[:100]:
        assert [p.label for p in q.parts] == list("abcd")[: len(MARKS)]
        assert [p.marks for p in q.parts] == MARKS
        assert [p.follow_through is not None for p in q.parts] == FT_LABELS
        assert all(p.answer_kind == "number" for p in q.parts)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("estimate_volume_capacity_reference", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


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


def test_answer_space(questions):
    answers = [q.answer_text for q in questions]
    distinct = set(answers)
    assert len(distinct) >= 100, len(distinct)
    top = max(answers.count(a) for a in distinct)
    assert top / len(answers) <= TOP_SHARE, top


def test_deterministic():
    assert _exam(7).to_dict() == _exam(7).to_dict()
