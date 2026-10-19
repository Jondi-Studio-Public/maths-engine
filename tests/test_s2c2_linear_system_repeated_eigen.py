"""Tests `uc2-linear-system-repeated-eigen`: the topic `s2c2_linear_system_repeated_eigen` (easy, medium, hard) and the
exam chain `exd_x_s2c2_linear_system_repeated_eigen`.

The oracle reads the system and initial values back out of the question text and solves x' = A x with a matrix
exponential (`mpmath.expm`), a route that shares nothing with the generator's nilpotent-part or Jordan construction.
It also checks with SymPy that the repeated eigenvalue is defective.
"""

from __future__ import annotations

import re

import mpmath
import pytest
import sympy

import mathgen.generators.s2c2_linear_system_repeated_eigen_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.misconceptions import CATALOGUE
from mathgen.text_quality import find_text_defects
from tests.s2c2_oracle import half_up, parse_system, read_expr, t

SKILL_ID = "uc2-linear-system-repeated-eigen"
TOPIC = "s2c2_linear_system_repeated_eigen"
EXAM = "exd_x_s2c2_linear_system_repeated_eigen"
TIERS = ("easy", "medium", "hard")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _state(text, at):
    matrix, x0, _ = parse_system(text)
    return mpmath.expm(mpmath.matrix(matrix.tolist()) * at) * mpmath.matrix([int(v) for v in x0])


def test_source_wired_last():
    last = _SKILL.sources[-1]
    assert (last.topic, last.difficulty, last.stages) == (EXAM, "exam", ["exam"])
    assert [src.difficulty for src in _SKILL.sources] == ["easy", "medium", "hard", "exam"]


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_matrix_exponential(tier):
    for seed in range(25):
        q = _q(tier, seed)
        k = "xyz".index(re.findall(r"([xyz])\(t\)", q.text)[-1])
        answer = read_expr(q.answer_text)
        for at in (0.4, 0.9):
            state = _state(q.text, at)
            assert abs(float(state[k]) - float(answer.subs(t, at))) < 1e-6 * max(1, abs(float(state[k]))), (
                seed,
                q.text,
                q.answer_text,
            )


@pytest.mark.parametrize("tier", TIERS)
def test_the_matrix_really_is_defective(tier):
    for seed in range(25):
        matrix, _, _ = parse_system(_q(tier, seed).text)
        repeated = [lam for lam, mult in matrix.eigenvals().items() if mult == 2]
        assert len(repeated) == 1, (tier, seed)
        assert len((matrix - repeated[0] * sympy.eye(matrix.shape[0])).nullspace()) == 1
        assert matrix.shape[0] == (3 if tier == "hard" else 2)


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(25):
        q = _q(tier, seed)
        assert grade(_SKILL, q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) >= 3
        for text in q.distractors:
            assert not grade(_SKILL, q, text).correct, (seed, q.text, text)


@pytest.mark.parametrize("tier", TIERS)
def test_distractors_are_tagged_wrong_answers(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert set(q.option_notes) == set(q.distractors)
        assert all(note in CATALOGUE for note in q.option_notes.values())
        assert q.answer_text not in q.distractors
        assert len(set(q.distractors)) == len(q.distractors)


def test_a_differently_written_but_equal_answer_is_accepted():
    q = _q("easy", 2)
    assert grade(_SKILL, q, q.answer_text.replace(" ", "")).correct
    assert grade(_SKILL, q, "x(t) = " + q.answer_text).correct


def test_easy_asks_x_and_medium_asks_y():
    for seed in range(30):
        assert re.findall(r"([xyz])\(t\)", _q("easy", seed).text)[-1] == "x"
        assert re.findall(r"([xyz])\(t\)", _q("medium", seed).text)[-1] == "y"


@pytest.mark.parametrize("tier", TIERS)
def test_variety(tier):
    answers = [_q(tier, seed).answer_text for seed in range(80)]
    assert len(set(answers)) >= 30
    assert max(answers.count(a) for a in set(answers)) <= 6


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(40):
        q = _q(tier, seed)
        assert find_text_defects(q) == [], (seed, find_text_defects(q))
        assert q.to_json() == _q(tier, seed).to_json()


# ---------------------------------------------------------------------------------------------------------------
# the exam chain


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 3, 1]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "expression", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert find_text_defects(q) == []
        assert all(len(p.distractors) >= 2 for p in q.parts)


def test_exam_keys_match_independent_derivation():
    for seed in range(25):
        q = _exam(seed)
        matrix, x0, _ = parse_system(q.stem)
        a, b, c, d = q.parts
        lam = next(iter(matrix.eigenvals()))
        assert sympy.Integer(a.answer_text) == lam
        n = matrix - lam * sympy.eye(2)
        assert sympy.Integer(b.answer_text) == (n * sympy.Matrix(x0))[0]
        for at in (0.5, 1.3):
            state = _state(q.stem, at)
            assert abs(float(state[0]) - float(read_expr(c.answer_text).subs(t, at))) < 1e-6 * max(
                1, abs(float(state[0]))
            )
        arg = d.prompt.split("x(")[1].split(")")[0]
        assert half_up(_state(q.stem, mpmath.mpf(arg))[0]) == d.answer_text, (seed, d.answer_text)


def test_exam_key_accepted_every_distractor_rejected():
    for seed in range(20):
        q = _exam(seed)
        full = grade(_SKILL, q, q.answer_text)
        assert full.correct and full.marks == full.max_marks == 7
        canon = _canonical(q)
        for p in q.parts:
            for text in p.distractors:
                assert grade(_SKILL, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)


def test_follow_through_from_a_wrong_eigenvalue():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        canon = _canonical(q)
        matrix, x0, _ = parse_system(q.stem)
        for wrong in q.parts[0].distractors:
            lam = sympy.Integer(wrong)
            reply_b = (matrix[0, 0] - lam) * x0[0] + matrix[0, 1] * x0[1]
            if lam == 0 or str(reply_b) in (canon["b"], wrong):
                continue
            result = grade(_SKILL, q, {**canon, "a": wrong, "b": str(reply_b)})
            assert result.parts["a"].marks == 0
            assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, wrong, result)
            checked += 1
    assert checked >= 30


def test_follow_through_from_a_wrong_expression_into_the_value():
    checked = 0
    for seed in range(40):
        q = _exam(seed)
        canon = _canonical(q)
        arg = sympy.Rational(q.parts[3].prompt.split("x(")[1].split(")")[0])
        for wrong in q.parts[2].distractors:
            value = read_expr(wrong).subs(t, arg).evalf(30)
            if abs(value) < 0.01 or abs(value) > 1e4 or half_up(value) == canon["d"] or not read_expr(wrong).has(t):
                continue
            result = grade(_SKILL, q, {**canon, "c": wrong, "d": half_up(value)})
            assert result.parts["c"].marks == 0
            assert result.parts["d"].marks == 1 and result.parts["d"].follow_through is True, (seed, wrong, result)
            checked += 1
    assert checked >= 40


def test_exam_answer_space_and_determinism():
    answers = [_exam(seed).answer_text for seed in range(150)]
    assert len(set(answers)) >= 60
    assert _exam(7).to_json() == _exam(7).to_json()
