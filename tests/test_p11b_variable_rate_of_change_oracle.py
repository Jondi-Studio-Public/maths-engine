"""Reads only `question.text` / `answer_text` / `distractors` via `mathgen.make`.
Imports NOTHING from `mathgen.generators.p11b_variable_rate_of_change_build`
-- the polynomial f(x) and the interval endpoints are parsed back out of the
rendered prose with regular expressions, then the average rate of change is
re-derived with `sympy.sympify`/`.subs()` and exact `Fraction` division,
never by calling into the generator's own construction (plain Python integer
subtraction and floor division).
"""

import re
from fractions import Fraction

import pytest
from sympy import Symbol, sympify

from mathgen import make

TOPIC = "p11b_variable_rate_of_change"
TIERS = ("easy", "medium", "hard")
SEEDS = range(60)

x = Symbol("x")

_HEAD_RE = re.compile(
    r"^A (?:quadratic|cubic) function is given by f\(x\) = (?P<poly>.+?)\. Find the average rate "
    r"of change of f\(x\) over the interval from x = (?P<x1>-?\d+) to "
    r"x = (?P<x2>-?\d+)\.$"
)


def _parse_poly(poly_text: str):
    """A rendered polynomial ("x^2 + 5x - 3", "-2x^2 - 5x + 9", "x^3 - 9x - 5")
    into a sympy expression: insert an explicit `*` between a digit and an
    immediately following `x` (join_terms never writes one), then `^` -> `**`
    for sympify."""
    with_star = re.sub(r"(\d)(x)", r"\1*\2", poly_text)
    py_text = with_star.replace("^", "**")
    return sympify(py_text, locals={"x": x})


def _oracle_rate(question):
    m = _HEAD_RE.match(question.text)
    assert m is not None, f"could not parse question head from: {question.text!r}"
    f_expr = _parse_poly(m.group("poly"))
    x1 = int(m.group("x1"))
    x2 = int(m.group("x2"))
    f_x1 = f_expr.subs(x, x1)
    f_x2 = f_expr.subs(x, x2)
    rate = Fraction(int(f_x2 - f_x1), x2 - x1)
    assert rate.denominator == 1, f"non-integer rate {rate} from text: {question.text!r}"
    return int(rate)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_rendered_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        oracle_rate = _oracle_rate(q)
        assert oracle_rate == int(q.answer_text), (
            f"seed={seed}/{tier}: oracle {oracle_rate} != rendered answer {q.answer_text}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        oracle_rate = _oracle_rate(q)
        for d in q.distractors:
            assert oracle_rate != int(d), (
                f"seed={seed}/{tier}: distractor {d!r} equals oracle answer {oracle_rate}\ntext: {q.text}"
            )
