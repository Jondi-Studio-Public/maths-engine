"""Exam source of `f10-scientific-notation-measurements`: journey time in hours, then in days."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n3_probe_travel_time"
SKILL_ID = "f10-scientific-notation-measurements"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _half_up(x: Fraction, places: int = 1) -> Fraction:
    scale = 10**places
    return Fraction(int(x * scale + Fraction(1, 2)), scale)


def oracle(q):
    s, a = re.search(r"speed of (\d+) × 10\^(\d+) km/h", q.stem).groups()
    c, e = re.search(r"distance to travel is ([\d.]+) × 10\^(\d+) km", q.stem).groups()
    hours = Fraction(c) * 10 ** int(e) / (int(s) * 10 ** int(a))
    return [hours, _half_up(hours / 24)]


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages) for s in exam] == [(TOPIC, ["exam"])]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {
        prompt_shape(make("scientific_notation_measurements", seed=s, difficulty="hard", variant="compare_magnitude"))
        for s in range(60)
    }
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        assert [Fraction(p.answer_text) for p in q.parts] == oracle(q), (seed, q.text)


def test_grader_accepts_key_and_rejects_distractors():
    for seed in range(120):
        q = _exam(seed)
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 4
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert p.distractors
            for d in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (seed, p.label, d)


def test_follow_through_from_wrong_hours():
    checked = 0
    for seed in range(120):
        q = _exam(seed)
        wrong = int(q.parts[0].distractors[0])
        reply = f"{_half_up(Fraction(wrong, 24)):.1f}"
        result = grade(_SKILL, q, {"a": str(wrong), "b": reply})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks and result.parts["b"].follow_through, seed
        checked += 1
    assert checked == 120


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
