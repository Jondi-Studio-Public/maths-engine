"""Exam tier for vector-projection: the point of an offset line closest to Q, as a parameter then coordinates.

(a) several prompt shapes, some absent from the hard tier; (b) each part's key re-derived from the RENDERED
PROMPT with exact fractions over 300 seeds; (c) the skill's grader accepts the key and rejects every distractor;
(d) a wrong parameter used correctly in the coordinate parts earns their marks.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u2_projection_foot"
SKILL_ID = "vector-projection"
SEEDS = range(300)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _printed(q):
    """Base point, direction and Q, from the printed stem."""
    q_x, q_y = (int(v) for v in re.search(r"Q\((-?\d+), (-?\d+)\)", q.stem).groups())
    if "direction vector" in q.stem:
        px, py, dx, dy = (
            int(v) for v in re.search(r"P\((-?\d+), (-?\d+)\).*d = \((-?\d+), (-?\d+)\)", q.stem).groups()
        )
    elif "vector equation" in q.stem:
        px, py, dx, dy = (
            int(v) for v in re.search(r"r = \((-?\d+), (-?\d+)\) \+ t\((-?\d+), (-?\d+)\)", q.stem).groups()
        )
    else:
        px, py, bx, by = (
            int(v) for v in re.search(r"A\((-?\d+), (-?\d+)\) and B\((-?\d+), (-?\d+)\)", q.stem).groups()
        )
        dx, dy = bx - px, by - py
    return px, py, dx, dy, q_x, q_y


def _oracle(q) -> dict[str, str]:
    px, py, dx, dy, qx, qy = _printed(q)
    t = Fraction((qx - px) * dx + (qy - py) * dy, dx * dx + dy * dy)
    return {"a": str(t), "b": str(px + t * dx), "c": str(py + t * dy)}


@pytest.fixture(scope="module")
def questions():
    return [_q(s) for s in SEEDS]


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_q(seed)) == []


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages) for s in exam] == [(TOPIC, ["exam"])]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_q(s)) for s in range(60)}
    hard = {prompt_shape(make("vector_projection", seed=s, kind="line_through_points")) for s in range(60)}
    assert len(exam) >= 3, exam
    assert exam - hard


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [3, 1, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_both_line_descriptions_occur(questions):
    styles = {
        "direction" if "direction vector" in q.stem else "vector" if "vector equation" in q.stem else "points"
        for q in questions
    }
    assert styles == {"direction", "vector", "points"}


def test_deterministic():
    assert _q(9).to_dict() == _q(9).to_dict()


def test_key_matches_independent_derivation_over_300_seeds(questions):
    bad = [(q.text, q.answer_text) for q in questions if {p.label: p.answer_text for p in q.parts} != _oracle(q)]
    assert bad == []


def test_answer_space(questions):
    texts = [q.answer_text for q in questions]
    assert len(set(texts)) >= 100
    assert max(texts.count(t) for t in set(texts)) <= 15


def test_key_accepted_and_every_distractor_rejected(questions):
    bad = []
    for q in questions[:150]:
        full = grade(_SKILL, q, q.answer_text)
        if not (full.correct and full.marks == full.max_marks == 5):
            bad.append(("key rejected", q.text))
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert len(p.distractors) >= 2, (q.text, p.label)
            assert all(note in CATALOGUE for note in p.option_notes.values())
            for d in p.distractors:
                if grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks != 0:
                    bad.append(("distractor accepted", q.text, p.label, d))
    assert bad == []


def test_follow_through_from_a_wrong_parameter(questions):
    for q in questions[:100]:
        px, py, dx, dy, _, _ = _printed(q)
        key = Fraction(_oracle(q)["a"])
        for offset in range(1, 9):
            wrong = key + offset
            x_wrong, y_wrong = px + wrong * dx, py + wrong * dy
            # The engine withholds follow-through from a reply that merely repeats an earlier value.
            if wrong != 0 and len({wrong, x_wrong, y_wrong}) == 3:
                break
        reply = {"a": str(wrong), "b": str(x_wrong), "c": str(y_wrong)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == 1 and result.parts[label].follow_through, (q.text, reply)
        assert result.marks == 2 < result.max_marks and not result.correct


def test_all_wrong_earns_zero(questions):
    for q in questions[:30]:
        assert grade(_SKILL, q, {"a": "99", "b": "7777", "c": "-7777"}).marks == 0
