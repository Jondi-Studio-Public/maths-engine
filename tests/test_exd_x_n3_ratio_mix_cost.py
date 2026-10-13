"""Exam source of `ratio-rate-proportion`: ratio share, cost of the mix, and its rate per kilogram."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n3_ratio_mix_cost"
SKILL_ID = "ratio-rate-proportion"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _half_up(x: Fraction, places: int) -> Fraction:
    scale = 10**places
    return Fraction(int(x * scale + Fraction(1, 2)), scale)


def oracle(q):
    ratio = tuple(int(v) for v in re.search(r"ratio (\d+) : (\d+) : (\d+) by", q.stem).groups())
    mass = Fraction(re.search(r"total mass of ([\d.]+) kg", q.stem).group(1))
    prices = [Fraction(v) for v in re.findall(r"\$([\d.]+) per kg", q.stem)]
    asked = re.search(r"mass of (.+?) in the batch", q.parts[0].prompt).group(1)
    names = re.search(r"is made from (.+?), (.+?) and (.+?) in the ratio", q.stem).groups()
    j = names.index(asked)
    share = mass * ratio[j] / sum(ratio)
    cost = sum(mass * r / sum(ratio) * p for r, p in zip(ratio, prices))
    rate = _half_up(cost / mass, 2)
    return None, share, cost, rate


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages) for s in exam] == [(TOPIC, ["exam"])]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("ratio", seed=s, difficulty="hard", kind="mixed_units")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_key_rederived_from_prompt():
    for seed in range(300):
        q = _exam(seed)
        _, share, cost, rate = oracle(q)
        got = [Fraction(p.answer_text) for p in q.parts]
        assert got == [share, _half_up(cost, 2), rate], (seed, q.text, q.answer_text)


def test_grader_accepts_key_and_rejects_distractors():
    for seed in range(120):
        q = _exam(seed)
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 6
        canon = {p.label: p.answer_text for p in q.parts}
        for p in q.parts:
            assert p.distractors
            for d in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: d}).parts[p.label].marks == 0, (seed, p.label, d)


def test_follow_through_from_wrong_share():
    checked = 0
    for seed in range(120):
        q = _exam(seed)
        wrong = Fraction(q.parts[0].distractors[0])
        ratio = tuple(int(v) for v in re.search(r"ratio (\d+) : (\d+) : (\d+) by", q.stem).groups())
        mass = Fraction(re.search(r"total mass of ([\d.]+) kg", q.stem).group(1))
        prices = [Fraction(v) for v in re.findall(r"\$([\d.]+) per kg", q.stem)]
        names = re.search(r"is made from (.+?), (.+?) and (.+?) in the ratio", q.stem).groups()
        j = names.index(re.search(r"mass of (.+?) in the batch", q.parts[0].prompt).group(1))
        cost_wrong = _half_up(wrong / ratio[j] * sum(r * p for r, p in zip(ratio, prices)), 2)
        rate_wrong = _half_up(cost_wrong / mass, 2)
        result = grade(_SKILL, q, {"a": str(wrong), "b": f"{cost_wrong:.2f}", "c": f"{rate_wrong:.2f}"})
        assert result.parts["a"].marks == 0
        if cost_wrong != Fraction(q.parts[1].answer_text):
            assert result.parts["b"].marks == result.parts["b"].max_marks, (seed, q.text)
            assert result.parts["b"].follow_through
            checked += 1
    assert checked >= 60


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
