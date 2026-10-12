"""Exam source of `f10-factor-multiple-divisibility-algorithm`: flagging rule with divisors sharing a factor."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n3_divisibility_count_chance"
SKILL_ID = "f10-factor-multiple-divisibility-algorithm"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _run_rule(q):
    """Re-run the stated rule over the stated range."""
    top = int(re.search(r"from 1 to (\d+) inclusive", q.stem).group(1))
    rule = re.search(r"flags a number when the number is (.*)\.$", q.stem).group(1)
    nums = [int(v) for v in re.findall(r"multiple of (\d+)", rule)]
    d1, d2 = nums
    if "but not both" in rule:
        flagged = [n for n in range(1, top + 1) if (n % d1 == 0) + (n % d2 == 0) == 1]
    else:
        flagged = [n for n in range(1, top + 1) if n % d1 == 0 and n % d2 != 0]
    return top, len(flagged)


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages) for s in exam] == [(TOPIC, ["exam"])]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {
        prompt_shape(make("p9e_factor_multiple_divisibility_algorithm", seed=s, difficulty="hard")) for s in range(60)
    }
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        top, count = _run_rule(q)
        assert int(q.parts[0].answer_text) == count, (seed, q.text)
        assert Fraction(q.parts[1].answer_text) == Fraction(count, top), (seed, q.text)


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


def test_follow_through_from_wrong_count():
    checked = 0
    for seed in range(120):
        q = _exam(seed)
        top, _ = _run_rule(q)
        wrong = int(q.parts[0].distractors[0])
        f = Fraction(wrong, top)
        reply = f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)
        result = grade(_SKILL, q, {"a": str(wrong), "b": reply})
        assert result.parts["a"].marks == 0
        if f < 1:
            assert result.parts["b"].marks == result.parts["b"].max_marks and result.parts["b"].follow_through, seed
            checked += 1
    assert checked >= 100


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
