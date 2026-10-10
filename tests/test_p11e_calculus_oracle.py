"""Reads only `question.text` / `answer_text` / `distractors` via `mathgen.make`.
Imports NOTHING from `mathgen.generators.p11e_calculus_build`.

Skill 1 (`derivative_definition`): the polynomial and the evaluation point are
parsed back out of the rendered prose, and f'(a) is re-derived with
`sympy.diff` -- a THIRD route, distinct from both the generator's own limit-
definition construction (expand the difference quotient, substitute h = 0)
and its closed-form power-rule cross-check (plain Python-int arithmetic on
the coefficients).

Skill 2 (`double_integral_rectangle`): g(x), h(y) and the rectangle bounds are
parsed back out of the rendered prose, and the double integral is re-derived
with `sympy.integrate()` over the full 2-variable product -- never reading
the generator's own Ix/Iy split.
"""

from __future__ import annotations

import re

import pytest
from sympy import diff, integrate, symbols, sympify

from mathgen import make

SEEDS = range(60)
TIERS = ("easy", "medium", "hard")

TOPIC_1 = "derivative_definition"
TOPIC_2 = "double_integral_rectangle"

_x = symbols("x")
_x2, _y2 = symbols("x y")


# --- Skill 1: derivative_definition -----------------------------------------

_HEAD_RE_1 = re.compile(
    r"^f\(x\) = (?P<fexpr>.+?)\. Using the limit definition f'\(a\) = "
    r"lim\(h→0\) \[f\(a\+h\) - f\(a\)\] / h, find f'\((?P<a>-?\d+)\)\.$"
)


def _parse_poly_x(prose: str):
    """A generator-rendered polynomial in x ("x^3 - 2x^2 + 3x - 5",
    "-3x^3 - 2x^2 + x") back into a sympy expression: `^` -> `**`, and an
    implicit coefficient-variable juxtaposition ("3x", "-2x") gets its
    multiplication restored before `sympify`."""
    py_text = prose.replace("^", "**")
    py_text = re.sub(r"(\d)(x)", r"\1*\2", py_text)
    return sympify(py_text, locals={"x": _x})


def _oracle_derivative_definition(question) -> int:
    head = _HEAD_RE_1.search(question.text)
    assert head is not None, f"could not parse question text: {question.text!r}"
    f_expr = _parse_poly_x(head.group("fexpr"))
    a = int(head.group("a"))
    derivative_expr = diff(f_expr, _x)
    value = derivative_expr.subs(_x, a)
    assert value.is_Integer, f"oracle derivative is not an integer: {value}"
    return int(value)


@pytest.mark.parametrize("tier", TIERS)
def test_derivative_definition_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        oracle_value = _oracle_derivative_definition(q)
        assert str(oracle_value) == q.answer_text, (
            f"seed={seed}/{tier}: oracle {oracle_value} != rendered answer {q.answer_text}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_derivative_definition_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC_1, seed=seed, difficulty=tier)
        oracle_value = _oracle_derivative_definition(q)
        for d in q.distractors:
            assert int(d) != oracle_value, f"seed={seed}/{tier}: distractor {d!r} equals the oracle's own answer"


# --- Skill 2: double_integral_rectangle -------------------------------------

_HEAD_RE_2 = re.compile(
    r"f\(x, y\) = \((?P<gexpr>.+?)\)\((?P<hexpr>.+?)\) over the rectangle "
    r"(?P<a1>-?\d+) ≤ x ≤ (?P<b1>-?\d+), (?P<a2>-?\d+) ≤ y ≤ (?P<b2>-?\d+)\."
)


def _parse_poly_xy(prose: str, name: str, symbol):
    py_text = prose.replace("^", "**")
    py_text = re.sub(rf"(\d)({name})", r"\1*\2", py_text)
    return sympify(py_text, locals={name: symbol})


def _oracle_double_integral_rectangle(question) -> int:
    head = _HEAD_RE_2.search(question.text)
    assert head is not None, f"could not parse question text: {question.text!r}"
    g_expr = _parse_poly_xy(head.group("gexpr"), "x", _x2)
    h_expr = _parse_poly_xy(head.group("hexpr"), "y", _y2)
    a1, b1 = int(head.group("a1")), int(head.group("b1"))
    a2, b2 = int(head.group("a2")), int(head.group("b2"))
    f_expr = g_expr * h_expr
    value = integrate(f_expr, (_x2, a1, b1), (_y2, a2, b2))
    assert value.is_Integer, f"oracle double integral is not an integer: {value}"
    return int(value)


@pytest.mark.parametrize("tier", TIERS)
def test_double_integral_rectangle_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        oracle_value = _oracle_double_integral_rectangle(q)
        assert str(oracle_value) == q.answer_text, (
            f"seed={seed}/{tier}: oracle {oracle_value} != rendered answer {q.answer_text}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_double_integral_rectangle_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC_2, seed=seed, difficulty=tier)
        oracle_value = _oracle_double_integral_rectangle(q)
        for d in q.distractors:
            assert int(d) != oracle_value, f"seed={seed}/{tier}: distractor {d!r} equals the oracle's own answer"
