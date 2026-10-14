"""Tests for the multi-part exam source of `senior-use-metric-units-of-volume-their-abbreviations-conversions`: a tank measured in mixed units: volume in cm3, capacity in litres, time to fill.

Each key is re-derived from the rendered prompt with independent code, every distractor is rejected by the
skill's own grader, and a wrong earlier answer used correctly in the next part earns that part's marks.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_m6_volume_tank"
SKILL_ID = "senior-use-metric-units-of-volume-their-abbreviations-conversions"
HARD = ("use_metric_units_of_volume", "hard")
SEEDS = range(300)

_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _dims(stem):
    return [Fraction(v) * (100 if u == "m" else 1) for v, u in re.findall(r"([\d.]+) (m|cm)\b", stem)[:3]]


def oracle(q):
    lc, wc, hc = _dims(q.stem)
    rate = int(re.search(r"(\d+) litres", q.stem).group(1))
    volume = lc * wc * hc
    litres = volume / 1000
    return {"a": str(int(volume)), "b": str(int(litres)), "c": str(int(litres / rate))}


def carried(q, a_wrong):
    rate = int(re.search(r"(\d+) litres", q.stem).group(1))
    b = _r(Fraction(a_wrong) / 1000, 0)
    return {"b": _s(b, 0), "c": _s(_r(b / rate, 0), 0)}


@pytest.fixture(scope="module")
def questions():
    return [make(TOPIC, seed=s, difficulty="exam") for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def _r(value, dp):
    """`value` rounded half up to `dp` places, as a learner would write it."""
    scale = 10**dp
    return Fraction(math.floor(Fraction(value) * scale + Fraction(1, 2)), scale)


def _s(value, dp):
    return f"{float(value):.{dp}f}"


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "number" and len(p.distractors) == 3 for p in q.parts)
        assert q.parts[0].option_notes


def test_prompt_shapes_add_to_hard_tier():
    exam = {prompt_shape(make(TOPIC, seed=s, difficulty="exam")) for s in range(80)}
    hard = {prompt_shape(make(HARD[0], seed=s, difficulty=HARD[1])) for s in range(80)}
    assert len(exam) >= 3, exam
    assert exam - hard


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_key_accepted_and_distractors_rejected(questions):
    for q in questions[:150]:
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_a_wrong_first_answer(questions):
    checked = 0
    for q in questions[:150]:
        for a_wrong in sorted(q.parts[0].distractors, key=Fraction, reverse=True):
            if Fraction(a_wrong) < 1:
                continue  # a reply that rounds to zero or is not a length carries no follow-through
            replies = carried(q, a_wrong)
            if all(Fraction(v) != 0 for v in replies.values()):
                break
        else:
            continue
        checked += 1
        result = grade(_SKILL, q, {"a": a_wrong, **replies})
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (q.text, label, replies)
            assert result.parts[label].follow_through is True
        assert not result.correct
    assert checked >= 120
