"""Exam source of `beam-reactions-point-load`: reactions under two unequal point loads."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u6_beam_two_point_loads"
SKILL_ID = "beam-reactions-point-load"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _oracle(q):
    nums = [int(v) for v in re.findall(r"\d+", q.stem)]
    L = nums[0]
    if q.stem.split(". ", 1)[1].startswith("Loads of"):
        w1, w2, a1, a2 = nums[1:]
    else:
        w1, a1, w2, a2 = nums[1:]
    # Moments about B (the generator takes moments about A).
    r_a = Fraction(w1 * (L - a1) + w2 * (L - a2), L)
    return {"a": w1 + w2 - r_a, "b": r_a}


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    easy = {prompt_shape(make("structural", seed=s, kinds=("reactions",))) for s in range(60)}
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


def test_grader_accepts_key_and_rejects_distractors():
    shown = 0
    for seed in range(100):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 4
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            for d in p.distractors:
                shown += 1
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (seed, p.label, d)
    assert shown >= 250


def test_follow_through_from_a_wrong_r_b():
    checked = 0
    for seed in range(100):
        q = _exam(seed)
        total = sum(int(v) for v in re.findall(r"(\d+) kN", q.stem))
        wrong = Fraction(q.parts[0].distractors[0])
        follow = total - wrong
        if follow <= 0 or follow == wrong:
            continue
        result = grade(_SKILL, q, {"a": q.parts[0].distractors[0], "b": f"{float(follow):.1f}"})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks, (seed, wrong, follow)
        assert result.parts["b"].follow_through
        checked += 1
    assert checked >= 30


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
