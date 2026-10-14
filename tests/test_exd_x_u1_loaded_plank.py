"""Exam source of `beam-equilibrium-seesaw`: a uniform plank on an off-centre pivot."""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u1_loaded_plank"
SKILL_ID = "beam-equilibrium-seesaw"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed)


def _read(q):
    stem = q.stem
    p = Fraction(re.search(r"([\d.]+) m from its left", stem).group(1))
    rest = re.sub(r"([\d.]+) m from its left", "", stem, count=1)
    length = Fraction(re.search(r"([\d.]+) m\b", rest).group(1))
    w, load = (Fraction(v) for v in re.findall(r"(\d+) N", stem))
    return length, p, w, load


def oracle(q):
    """Take moments about the RIGHT end (not the pivot) to find the reaction, then F from the vertical balance."""
    length, p, w, load = _read(q)
    moment = w * (length / 2 - p)
    # Pivot reaction R and balancing force F from two independent equations:
    #   forces:           R = load + w + F
    #   moments, right end: R (L - p) = load L + w L / 2
    reaction = (load * length + w * length / 2) / (length - p)
    force = reaction - load - w
    return {"a": str(int(moment)), "b": str(int(force)), "c": str(int(reaction))}


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(300)]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [1, 3, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_composed_answer_full_marks(questions):
    for q in questions:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 5


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_moment(questions):
    """A wrong moment for the plank's weight, used correctly in (b) and (c), earns those marks."""
    checked = 0
    for q in questions:
        length, p, w, load = _read(q)
        wrong_a = w * p  # the moment arm taken as the pivot's distance from the end
        force = (load * p - wrong_a) / (length - p)
        if wrong_a == w * (length / 2 - p) or force <= 0 or force.denominator != 1:
            continue
        reaction = load + w + force
        result = grade(_SKILL, q, {"a": str(int(wrong_a)), "b": str(int(force)), "c": str(int(reaction))})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 3 and result.parts["b"].follow_through is True
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True
        checked += 1
    assert checked >= 20
