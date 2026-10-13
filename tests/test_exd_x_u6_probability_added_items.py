"""Exam source of `simple-event-probability`: probability, a target-probability equation, a new probability."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u6_probability_added_items"
SKILL_ID = "simple-event-probability"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _oracle(q):
    r, g, b = (int(v) for v in re.search(r"(\d+) red, (\d+) green and (\d+) blue", q.stem).groups())
    p, qq = (int(v) for v in re.search(r"is (\d+)/(\d+)\. How many", q.text).groups())
    # Search for the number added instead of solving the equation the generator solves.
    x = next(k for k in range(0, 500) if Fraction(r + k, r + g + b + k) == Fraction(p, qq))
    return {"a": Fraction(r, r + g + b), "b": Fraction(x), "c": Fraction(b, r + g + b + x)}


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    easy = {prompt_shape(make("simple_probability", seed=s)) for s in range(60)}
    assert len(exam) >= 3
    assert exam - easy


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        expected = _oracle(q)
        for p in q.parts:
            assert Fraction(p.answer_text) == expected[p.label], (seed, q.text, p.label)


def test_every_probability_option_is_a_probability():
    for seed in range(100):
        q = _exam(seed)
        for label in "ac":
            part = next(p for p in q.parts if p.label == label)
            assert part.sensible_range == (0, 1)
            assert 0 < Fraction(part.answer_text) < 1
            for d in part.distractors:
                assert 0 <= Fraction(d) <= 1, (seed, d)


def test_grader_accepts_key_and_rejects_distractors():
    shown = 0
    for seed in range(100):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 6
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            for d in p.distractors:
                shown += 1
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d)
    assert shown >= 250


def test_follow_through_from_a_wrong_number_added():
    checked = 0
    for seed in range(100):
        q = _exam(seed)
        r, g, b = (int(v) for v in re.search(r"(\d+) red, (\d+) green and (\d+) blue", q.stem).groups())
        wrong = Fraction(q.parts[1].distractors[0])
        if wrong != int(wrong):
            continue
        follow = Fraction(b, r + g + b + int(wrong))
        if follow == Fraction(q.parts[2].answer_text):
            continue
        reply = {"a": q.parts[0].answer_text, "b": q.parts[1].distractors[0], "c": str(follow)}
        result = grade(_SKILL, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == result.parts["c"].max_marks, (seed, reply)
        checked += 1
    assert checked >= 10


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
