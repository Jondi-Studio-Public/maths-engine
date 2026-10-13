"""Exam source of `ucv1-bayes-normal-posterior`: a four-part chain over two rounds of normal-normal updating."""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2f_bayes_two_rounds"
SKILL_ID = "ucv1-bayes-normal-posterior"
SEEDS = range(300)
MU = sympy.Symbol("mu")
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_STEM = re.compile(
    r"prior N\((-?\d+), (\d+)\)\. Measurements have known variance σ² = (\d+)\. "
    r"Round one gives n₁ = (\d+) measurements with sample mean (-?\d+); "
    r"round two gives n₂ = (\d+) measurements with sample mean (-?\d+)\."
)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _frac(value) -> str:
    value = sympy.Rational(value)
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


def _complete_the_square(prior_mean, prior_var, n, xbar, noise):
    """Posterior (mean, variance) as the minimum and curvature of -2 log of prior x likelihood, by calculus."""
    q = (MU - prior_mean) ** 2 / prior_var + n * (MU - xbar) ** 2 / noise
    mean = sympy.solve(sympy.diff(q, MU), MU)[0]
    return mean, 2 / sympy.diff(q, MU, 2)


def oracle(q):
    mu0, tau0, noise, n1, x1, n2, x2 = (sympy.Integer(v) for v in _STEM.search(q.stem).groups())
    mu1, var1 = _complete_the_square(mu0, tau0, n1, x1, noise)
    mu2, var2 = _complete_the_square(mu1, var1, n2, x2, noise)
    return mu1, var1, mu2, var2 + noise, noise, n2, x2


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_source_wired():
    exam = [s for s in _SKILL.sources if s.difficulty == "exam"]
    assert [s.topic for s in exam] == [TOPIC]
    assert exam[0].stages == ["exam"]


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 1, 3, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert all(p.answer_kind == "fraction" for p in q.parts)
        assert all(len(p.distractors) >= 2 and p.option_notes for p in q.parts)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("p14e_normal_normal_posterior", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects(questions):
    for q in questions[:60]:
        assert find_text_defects(q) == []


def test_oracle_agrees(questions):
    for q in questions:
        mu1, var1, mu2, predictive, *_ = oracle(q)
        assert _canonical(q) == {"a": _frac(mu1), "b": _frac(var1), "c": _frac(mu2), "d": _frac(predictive)}, q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 8


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_variance(questions):
    """A round-one variance that is wrong (here, the precision) is carried into round two and the prediction."""
    checked = 0
    for q in questions[:120]:
        mu1, _var1, _mu2, _pred, noise, n2, x2 = oracle(q)
        b_wrong = sympy.Rational(q.parts[1].distractors[0])
        mu2_w, var2_w = _complete_the_square(mu1, b_wrong, n2, x2, noise)
        pred_w = var2_w + noise
        values = [mu1, b_wrong, mu2_w, pred_w]
        if 0 in values or len(set(values)) < 4:  # follow-through is withheld from a zero or a repeated reply
            continue
        reply = {"a": _frac(mu1), "b": _frac(b_wrong), "c": _frac(mu2_w), "d": _frac(pred_w)}
        result = grade(_SKILL, q, reply)
        assert result.parts["b"].marks == 0
        for label, marks in (("c", 3), ("d", 2)):
            assert result.parts[label].marks == marks, (q.text, label, reply)
            assert result.parts[label].follow_through is True
        assert result.marks == 7 and not result.correct
        checked += 1
    assert checked >= 60


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "17", "b": "19", "c": "23", "d": "29"})
        assert result.marks == 0 and not result.correct


def test_answer_space(questions):
    tuples = Counter(q.answer_text for q in questions)
    assert len(tuples) >= 100
    assert max(tuples.values()) / len(questions) <= 0.05


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
