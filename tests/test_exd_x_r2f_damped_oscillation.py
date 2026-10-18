"""Exam source of `ucv1-second-order-characteristic-roots`: a four-part chain from the roots to a value of the solution."""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy
from sympy import exp, pi
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from mathgen.engine import grade, load_skills
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_r2f_damped_oscillation"
SKILL_ID = "ucv1-second-order-characteristic-roots"
SEEDS = range(300)
R, X = sympy.symbols("r x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)
_STEM = re.compile(r"(y'' .*?) = 0, with y\(0\) = (-?\d+) and y'\(0\) = (-?\d+)\.")


def _exam(seed):
    return make(TOPIC, seed=seed, difficulty="exam")


def _frac(value) -> str:
    value = sympy.Rational(value)
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


def _coefficients(ode: str):
    """b and c from the displayed equation y'' + b y' + c y, read term by term."""
    by_order = {}
    for sign, digits, primes in re.findall(r"([+-])?\s*(\d*)y('*)", ode):
        by_order[len(primes)] = (-1 if sign == "-" else 1) * int(digits or 1)
    assert by_order[2] == 1 and set(by_order) == {0, 1, 2}, ode
    return by_order[1], by_order[0]


def oracle(q):
    """Solve the displayed initial value problem from the characteristic equation, with SymPy's own differentiation."""
    ode, y0, v0 = _STEM.search(q.stem).groups()
    b, c = _coefficients(ode)
    roots = sympy.solve(R**2 + b * R + c, R)
    root = next(w for w in roots if sympy.im(w) > 0)
    alpha, k = sympy.re(root), sympy.im(root)
    cc, dd = sympy.symbols("C D")
    y = exp(alpha * X) * (cc * sympy.cos(k * X) + dd * sympy.sin(k * X))
    sol = sympy.solve([y.subs(X, 0) - int(y0), sympy.diff(y, X).subs(X, 0) - int(v0)], [cc, dd])
    y = y.subs(sol)
    return alpha, k, sol[dd], sympy.simplify(y.subs(X, pi / (2 * k)))


@pytest.fixture(scope="module")
def questions():
    return [_exam(s) for s in SEEDS]


def _canonical(q):
    return {p.label: p.answer_text for p in q.parts}


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]
        assert [p.answer_kind for p in q.parts] == ["fraction", "fraction", "fraction", "expression"]
        assert all(len(p.distractors) >= 2 and p.option_notes for p in q.parts)


def test_oracle_agrees(questions):
    for q in questions:
        alpha, k, d, y_end = oracle(q)
        canon = _canonical(q)
        assert (canon["a"], canon["b"], canon["c"]) == (_frac(alpha), _frac(k), _frac(d)), q.text
        shown = parse_expr(canon["d"].replace("^", "**"), transformations=_TRANSFORMS)
        assert sympy.simplify(shown - y_end) == 0, q.text


def test_composed_answer_full_marks(questions):
    for q in questions[:100]:
        result = grade(_SKILL, q, q.answer_text)
        assert result.correct and result.marks == result.max_marks == 7


def test_distractors_rejected(questions):
    for q in questions[:150]:
        canon = _canonical(q)
        for p in q.parts:
            for d in p.distractors:
                result = grade(_SKILL, q, {**canon, p.label: d})
                assert result.parts[p.label].marks == 0, (q.text, p.label, d)


def test_follow_through_from_sign_flipped_alpha(questions):
    """alpha with the wrong sign leaves k unchanged, but changes D and the final value, which follow through."""
    checked = 0
    for q in questions[:100]:
        ode, y0, v0 = _STEM.search(q.stem).groups()
        alpha, k, _d, _y = oracle(q)
        aw = -alpha
        dw = sympy.Rational(int(v0) - aw * int(y0)) / k
        yw = dw * exp(aw * pi / (2 * k))
        values = [aw, k, dw]
        if len({sympy.simplify(v) for v in values}) < 3 or dw in (aw, k, 0):
            continue  # follow-through is withheld from a reply that only repeats an earlier one
        reply = {"a": _frac(aw), "b": _frac(k), "c": _frac(dw), "d": str(yw)}
        result = grade(_SKILL, q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (q.text, reply)
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (q.text, reply)
        assert result.marks == 6 and not result.correct
        checked += 1
    assert checked >= 60


def test_follow_through_from_wrong_k(questions):
    """A doubled k (the quadratic formula without the halving) carried into D and the final value."""
    checked = 0
    for q in questions[:100]:
        ode, y0, v0 = _STEM.search(q.stem).groups()
        alpha, k, _d, _y = oracle(q)
        kw = 2 * k
        dw = sympy.Rational(int(v0) - alpha * int(y0)) / kw
        yw = dw * exp(alpha * pi / (2 * kw))
        if dw in (alpha, kw) or dw == 0:
            continue
        reply = {"a": _frac(alpha), "b": _frac(kw), "c": _frac(dw), "d": str(yw)}
        result = grade(_SKILL, q, reply)
        assert result.parts["b"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["d"].marks == 2, (q.text, reply)
        checked += 1
    assert checked >= 60


def test_all_wrong_earns_zero(questions):
    for q in questions[:40]:
        result = grade(_SKILL, q, {"a": "17", "b": "19", "c": "23/5", "d": "7*pi"})
        assert result.marks == 0 and not result.correct


def test_chain_is_not_copyable():
    """No part repeats an earlier answer or follows through as a restatement, and junk replies earn nothing."""
    from tests.exd_junk_probe import assert_chain_is_not_copyable

    assert_chain_is_not_copyable("exd_x_r2f_damped_oscillation", "ucv1-second-order-characteristic-roots")
