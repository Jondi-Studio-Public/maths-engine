"""Exam source of `simple-harmonic-angular-frequency`: spring constant, angular frequency, period."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_u1_spring_oscillation"
SKILL_ID = "simple-harmonic-angular-frequency"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _exam(seed):
    return make(TOPIC, seed=seed)


def _dp2(value) -> str:
    return str(Decimal(str(sympy.N(value, 40))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def oracle(q):
    """Re-derive the three answers from the rendered text, in exact arithmetic until the final rounding."""
    m0 = Fraction(re.search(r"(\d+\.\d) kg", q.stem).group(1))
    x0 = Fraction(re.search(r"(\d+) cm", q.stem).group(1)) / 100
    masses = {Fraction(v) for v in re.findall(r"(\d+\.\d) kg", q.parts[1].prompt)}
    (m,) = masses - {m0}
    k = m0 * Fraction(98, 10) / x0
    omega = sympy.sqrt(sympy.Rational(k.numerator, k.denominator) / sympy.Rational(m.numerator, m.denominator))
    return {"a": str(k), "b": _dp2(omega), "c": _dp2(2 * sympy.pi / omega)}


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in range(300)]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert [s.stages for s in exam] == [["exam"]]


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {
        prompt_shape(make("simple_harmonic_angular_frequency", seed=s, difficulty="hard", kind="solve_for_k"))
        for s in range(60)
    }
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects():
    for seed in range(60):
        assert find_text_defects(_exam(seed)) == []


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 2, 1]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_oracle_agrees(questions):
    for q in questions:
        assert oracle(q) == _canonical(q), q.text


def test_answer_space(questions):
    answers = [q.answer_text for q in questions]
    assert len(set(answers)) >= 100
    assert max(answers.count(a) for a in set(answers)) <= 0.05 * len(answers)


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


def test_follow_through_from_wrong_spring_constant(questions):
    checked = 0
    for q in questions:
        canon = _canonical(q)
        k_wrong = int(canon["a"]) + 20
        masses = {Fraction(v) for v in re.findall(r"(\d+\.\d) kg", q.parts[1].prompt)}
        m0 = Fraction(re.search(r"(\d+\.\d) kg", q.stem).group(1))
        (m,) = masses - {m0}
        omega = sympy.sqrt(sympy.Rational(k_wrong) / sympy.Rational(m.numerator, m.denominator))
        b_wrong = _dp2(omega)
        c_wrong = _dp2(2 * sympy.pi / sympy.Float(b_wrong, 30))
        result = grade(_SKILL, q, {"a": str(k_wrong), "b": b_wrong, "c": c_wrong})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True
        checked += 1
    assert checked == len(questions)


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
