"""Verifies the artefact the learner receives, not the value the generator's own internals hold."""

import re

import pytest
import sympy
from sympy import Eq, Symbol, degree, div, solve, sympify
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make

x = Symbol("x")
SEEDS = range(60)
TIERS = ["easy", "medium", "hard"]
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


def _parse(text: str):
    """Parse rendered polynomial text (``^`` for powers, implicit
    multiplication, no ``*``/``**``) into a sympy expression."""
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMATIONS)


# ---------------------------------------------------------------------------
# zero_product_property: "Solve (factor)(factor)...(factor) = 0"

_FACTOR_RE = re.compile(r"\(([^()]+)\)")
_ROOT_RE = re.compile(r"x = (-?\d+(?:/\d+)?)")


def _oracle_zero_product_roots(question):
    factor_texts = _FACTOR_RE.findall(question.text.split(" = 0")[0])
    assert factor_texts, f"no factors found in {question.text!r}"
    product = sympy.Integer(1)
    for factor_text in factor_texts:
        product *= _parse(factor_text)
    solved = solve(Eq(sympy.expand(product), 0), x)
    return sorted(float(r) for r in solved), len(factor_texts)


def _oracle_answer_roots(answer_text):
    values = [sympify(token) for token in _ROOT_RE.findall(answer_text)]
    assert values, f"no roots parsed from {answer_text!r}"
    return sorted(float(v) for v in values)


@pytest.mark.parametrize("tier", TIERS)
def test_zero_product_property_oracle(tier):
    checked = 0
    for seed in SEEDS:
        q = make("zero_product_property", seed=seed, difficulty=tier)
        derived_roots, n_factors = _oracle_zero_product_roots(q)
        stated_roots = _oracle_answer_roots(q.answer_text)
        assert len(derived_roots) == len(stated_roots) == n_factors, (
            f"seed {seed} tier {tier}: {q.text} -> answer {q.answer_text!r}: "
            f"derived {derived_roots} vs stated {stated_roots}"
        )
        for derived, stated in zip(derived_roots, stated_roots):
            assert abs(derived - stated) < 1e-9, (
                f"seed {seed} tier {tier}: {q.text} -> answer_text says "
                f"{q.answer_text!r}, but solving the product independently "
                f"gives {derived_roots}, not {stated_roots}"
            )
        checked += 1
    assert checked == len(SEEDS)


# ---------------------------------------------------------------------------
# polynomial_long_division: "Divide f(x) = ... by g(x) = .... Give ..."
#   answer_text: "<quotient> + (<remainder>)/(<divisor>)"

_QUESTION_RE = re.compile(r"f\(x\) = (.+) by g\(x\) = (.+)\. Give")
_ANSWER_RE = re.compile(r"^(.+) \+ \((.+)\)/\((.+)\)$")


@pytest.mark.parametrize("tier", TIERS)
def test_polynomial_long_division_oracle(tier):
    checked = 0
    for seed in SEEDS:
        q = make("polynomial_long_division", seed=seed, difficulty=tier)
        qmatch = _QUESTION_RE.search(q.text)
        assert qmatch, q.text
        dividend = _parse(qmatch.group(1))
        divisor = _parse(qmatch.group(2))

        amatch = _ANSWER_RE.match(q.answer_text)
        assert amatch, f"seed {seed} tier {tier}: answer not in Q + (R)/(D) form: {q.answer_text!r}"
        stated_quotient = _parse(amatch.group(1))
        stated_remainder = _parse(amatch.group(2))
        stated_divisor = _parse(amatch.group(3))

        # Independent route: exact polynomial division over QQ (needed for a
        # non-monic divisor), never the generator's own forward construction.
        expected_q, expected_r = div(
            sympy.Poly(sympy.expand(dividend), x, domain=sympy.QQ),
            sympy.Poly(sympy.expand(divisor), x, domain=sympy.QQ),
        )
        assert sympy.expand(stated_quotient - expected_q.as_expr()) == 0, (
            f"seed {seed} tier {tier}: stated quotient {stated_quotient} != "
            f"independently divided quotient {expected_q.as_expr()}"
        )
        assert sympy.expand(stated_remainder - expected_r.as_expr()) == 0, (
            f"seed {seed} tier {tier}: stated remainder {stated_remainder} != "
            f"independently divided remainder {expected_r.as_expr()}"
        )
        assert sympy.expand(stated_divisor - divisor) == 0, (
            f"seed {seed} tier {tier}: answer's own divisor {stated_divisor} != question's divisor {divisor}"
        )
        # The remainder must actually satisfy the division algorithm's own
        # degree rule -- not merely equal what sympy.div happened to return.
        assert degree(expected_r.as_expr(), x) < degree(divisor, x) or expected_r.as_expr() == 0
        checked += 1
    assert checked == len(SEEDS)
