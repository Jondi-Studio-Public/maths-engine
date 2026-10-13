"""Exam source of `ucv1-heat-equation-mode`: a three-part chain from the decay rate to a boundary quantity."""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import pytest
import sympy
from sympy import exp, log, pi
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2f_heat_rod_mode"
SKILL_ID = "ucv1-heat-equation-mode"
SEEDS = range(300)
X = sympy.Symbol("x", real=True)
T = sympy.Symbol("t", positive=True)
LAM = sympy.Symbol("lambda_", positive=True)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_STEM = re.compile(
    r"satisfies u_t = (\d+)u_xx for 0 < x < (\d+), with its ends (held|insulated).*"
    r"form u\(x, t\) = (-?\d+) e\^\(−λt\) (sin|cos)\((\d*)πx/(\d+)\)"
)


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _expr(text):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"pi": pi})


def _facts(q):
    alpha, length, ends, c, fn, num, den = _STEM.search(q.stem).groups()
    k = sympy.Rational(int(num or 1), int(den)) * pi  # n pi / L, from the displayed profile
    assert sympy.Rational(int(num or 1), int(den)) == sympy.Rational(1, 1) * sympy.Rational(int(num or 1), int(den))
    assert (ends == "held") == (fn == "sin"), q.stem
    return int(alpha), int(length), ends, int(c), fn, k


def oracle(q):
    """Substitute the displayed mode into the PDE to find lambda, then solve for the half-life and differentiate."""
    alpha, _length, ends, c, fn, k = _facts(q)
    lam, half, boundary = _solve_mode(alpha, ends, fn, k)
    return lam, half, c * boundary


@lru_cache(maxsize=None)
def _solve_mode(alpha, ends, fn, k):
    """The slow sympy oracle for a unit amplitude. The decay rate and half-life do not depend on the amplitude and the
    boundary quantity is proportional to it, so each distinct (alpha, ends, profile) is solved once."""
    trig = sympy.sin if fn == "sin" else sympy.cos
    u = exp(-LAM * T) * trig(k * X)
    # Substituting the mode into u_t = alpha u_xx leaves an equation that is linear in lambda; read it off the polynomial.
    equation = sympy.simplify(
        sympy.expand((sympy.diff(u, T) - alpha * sympy.diff(u, X, 2)) / exp(-LAM * T)) / trig(k * X)
    )
    poly = sympy.Poly(equation, LAM)
    lam = sympy.simplify(-poly.coeff_monomial(1) / poly.coeff_monomial(LAM))
    # exp(-lambda T) = 1/2: take logarithms of both sides, which leaves an equation that is linear in T.
    half = sympy.solve(sympy.Eq(-lam * T, log(sympy.Rational(1, 2))), T)[0]
    u_t = u.subs(LAM, lam)
    boundary = sympy.diff(u_t, X) if ends == "held" else u_t
    return lam, half, boundary.subs({X: 0, T: half})


@lru_cache(maxsize=None)
def _agrees(text, expected):
    """Whether the stated answer equals the oracle's value to 25 places; repeated (answer, value) pairs are checked once."""
    return abs(sympy.N(_expr(text) - expected, 30)) < sympy.Float("1e-25")


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
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 2, 3]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert all(p.answer_kind == "expression" for p in q.parts)
        assert all(len(p.distractors) >= 2 and p.option_notes for p in q.parts)


def test_prompt_variety_and_new_shapes():
    exam = {prompt_shape(_exam(s)) for s in range(60)}
    hard = {prompt_shape(make("p13e_heat_equation_mode", seed=s, difficulty="hard")) for s in range(60)}
    assert len(exam) >= 3
    assert exam - hard


def test_no_text_defects(questions):
    for q in questions[:60]:
        assert find_text_defects(q) == []


def test_oracle_agrees(questions):
    for q in questions:
        lam, half, boundary = oracle(q)
        canon = _canonical(q)
        for label, expected in (("a", lam), ("b", half), ("c", boundary)):
            assert _agrees(canon[label], expected), (q.text, label, canon[label], expected)


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 7


def test_learner_spellings(questions):
    """Typed the way a learner would: ln for the logarithm, pi^2 and a fraction bar."""
    for q in questions[:40]:
        lam, half, _boundary = oracle(q)
        canon = _canonical(q)
        spelled = str(half).replace("log", "ln").replace("**", "^")
        assert grade(_SKILL, q, {**canon, "b": spelled}).parts["b"].marks == 2, (q.text, spelled)


def test_distractors_rejected(questions):
    for q in questions[:100]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_wrong_decay_rate(questions):
    """A wrong lambda gives a wrong T that still halves the amplitude, so (b) follows through and (c) is unchanged."""
    checked = 0
    for q in questions[:100]:
        alpha, length, ends, c, fn, k = _facts(q)
        lam_w = _expr(q.parts[0].distractors[0])
        half_w = log(2) / lam_w
        canon = _canonical(q)
        result = grade(_SKILL, q, {**canon, "a": q.parts[0].distractors[0], "b": str(half_w)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (q.text, half_w)
        assert result.parts["c"].marks == 3
        checked += 1
    assert checked == 100


def test_follow_through_from_wrong_time(questions):
    """T = 1/lambda (ln 2 forgotten) carried into the boundary quantity, e^(-lambda T) = 1/e, earns (c)."""
    checked = 0
    for q in questions[:100]:
        alpha, length, ends, c, fn, k = _facts(q)
        canon = _canonical(q)
        lam = _expr(canon["a"])
        t_w = _expr(q.parts[1].distractors[0])
        assert abs(sympy.N(t_w - 1 / lam, 30)) < sympy.Float("1e-25")
        value = c * exp(-lam * t_w) * (k if ends == "held" else 1)
        result = grade(
            _SKILL, q, {**canon, "b": q.parts[1].distractors[0], "c": str(sympy.simplify(sympy.expand(value)))}
        )
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 3 and result.parts["c"].follow_through is True, (q.text, value)
        checked += 1
    assert checked == 100


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "7", "b": "pi", "c": "3/11"})
        assert result.marks == 0 and not result.correct


def test_answer_space(questions):
    tuples = Counter(q.answer_text for q in questions)
    assert len(tuples) >= 100
    assert max(tuples.values()) / len(questions) <= 0.05


def test_deterministic():
    assert _exam(7).to_json() == _exam(7).to_json()
