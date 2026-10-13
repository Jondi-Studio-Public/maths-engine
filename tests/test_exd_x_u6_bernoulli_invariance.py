"""Exam source of `bernoulli-mle`: estimate p from a recorded sequence, then p squared and the odds."""

from __future__ import annotations

from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u6_bernoulli_invariance"
SKILL_ID = "bernoulli-mle"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _oracle(q):
    seq = q.stem.split("Outcomes in order: ")[1].rstrip(".").split()
    letters = sorted(set(seq))
    win = {"G", "Y", "P"} & set(letters)
    (win,) = win
    n, k = len(seq), seq.count(win)
    p = Fraction(k, n)
    return {"a": p, "b": p * p, "c": p / (1 - p)}


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("bernoulli_mle", seed=s, kind="pooled")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


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


def test_probability_parts_stay_in_unit_interval():
    for seed in range(100):
        q = _exam(seed)
        for part in q.parts[:2]:
            assert part.sensible_range == (0, 1)
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


def test_follow_through_from_a_wrong_estimate():
    checked = 0
    for seed in range(100):
        q = _exam(seed)
        wrong = Fraction(q.parts[0].distractors[0])
        b, c = wrong**2, wrong / (1 - wrong)
        if len({wrong, b, c}) < 3:
            continue
        reply = {"a": q.parts[0].distractors[0], "b": str(b), "c": str(c)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        for label in "bc":
            assert result.parts[label].marks == result.parts[label].max_marks, (seed, label, reply)
            assert result.parts[label].follow_through
        checked += 1
    assert checked >= 30


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
