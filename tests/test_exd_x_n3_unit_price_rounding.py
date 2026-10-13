"""Exam source of `f10-exact-approximate-repeated-calculation`: a rounded unit price scaled up to an order."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_n3_unit_price_rounding"
SKILL_ID = "f10-exact-approximate-repeated-calculation"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _half_up(x: Fraction, places: int = 2) -> Fraction:
    scale = 10**places
    return Fraction(int(x * scale + Fraction(1, 2)), scale)


def _inputs(q):
    n, pack = re.search(r"A pack of (\d+) .*? costs \$([\d.]+)\.", q.stem).groups()
    m = re.search(r"sells (\d+) ", q.stem).group(1)
    return int(n), Fraction(pack), int(m)


def oracle(q):
    n, pack, m = _inputs(q)
    unit = _half_up(pack / n)
    order = m * unit
    exact = m * pack / n
    return [unit, order, _half_up(abs(order - exact))]


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [(s.topic, s.stages) for s in exam] == [(TOPIC, ["exam"])]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("p10e_exact_approx_repeated_calc", seed=s, difficulty="hard")) for s in range(60)}
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


def test_follow_through_from_wrong_unit_price():
    checked = 0
    for seed in range(120):
        q = _exam(seed)
        n, pack, m = _inputs(q)
        wrong = Fraction(q.parts[0].distractors[0])
        b_wrong = m * wrong
        c_wrong = _half_up(abs(b_wrong - m * pack / n))
        result = grade(_SKILL, q, {"a": str(wrong), "b": f"{b_wrong:.2f}", "c": f"{c_wrong:.2f}"})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == result.parts["b"].max_marks and result.parts["b"].follow_through
        if c_wrong != Fraction(q.parts[2].answer_text):
            assert result.parts["c"].marks == result.parts["c"].max_marks, (seed, q.text)
            checked += 1
    assert checked >= 50


def test_answer_space():
    answers = [_exam(s).answer_text for s in range(300)]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 15


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
