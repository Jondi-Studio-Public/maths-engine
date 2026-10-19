"""Reads only `question.text` / `answer_text` / `distractors` via `mathgen.make`.
Imports NOTHING from `mathgen.generators.p19f_difference_quotient_build` --
the polynomial, the point (or its two-step derivation, m -> x = 2m-1, on the
exam tier) and h are all parsed back out of the rendered prose with regular
expressions, then re-derived with sympy substitution written fresh here.
"""

import re

import pytest
from sympy import symbols
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make

SEEDS = range(200)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC = "p19f_difference_quotient_stated_h"

_x = symbols("x")


def _parse_poly(prose: str):
    # The prose writes a product by juxtaposition ("3x^2"), so parse with implicit multiplication.
    py_text = prose.replace("^", "**")
    return parse_expr(
        py_text,
        local_dict={"x": _x},
        transformations=standard_transformations + (implicit_multiplication_application,),
    )


_EASY_MEDIUM_RE = re.compile(r"^f\(x\) = (?P<f>.+?)\. Using x = (?P<p>-?\d+) and the fixed step h = (?P<h>-?\d+)")
_HARD_RE = re.compile(
    r"^f\(x\) = (?P<f>.+?)\. The interval runs from x = (?P<p>-?\d+) to x = -?\d+ "
    r"\(h = (?P<h>-?\d+) is negative"
)
_EXAM_RE = re.compile(
    r"^f\(x\) = (?P<f>.+?)\. Let x be the value satisfying x = 2m - 1 for m = (?P<m>-?\d+)\. "
    r"The interval then runs from that x to x\+h, where h = (?P<h>-?\d+) is negative"
)


def _oracle(question, tier):
    text = question.text
    if tier in ("easy", "medium"):
        m = _EASY_MEDIUM_RE.search(text)
        assert m is not None, f"could not parse {tier} text: {text!r}"
        p = int(m.group("p"))
    elif tier == "hard":
        m = _HARD_RE.search(text)
        assert m is not None, f"could not parse hard text: {text!r}"
        p = int(m.group("p"))
    else:  # exam
        m = _EXAM_RE.search(text)
        assert m is not None, f"could not parse exam text: {text!r}"
        p = 2 * int(m.group("m")) - 1
    f_expr = _parse_poly(m.group("f"))
    h = int(m.group("h"))
    value = (f_expr.subs(_x, p + h) - f_expr.subs(_x, p)) / h
    return int(value)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        oracle_answer = _oracle(q, tier)
        assert str(oracle_answer) == q.answer_text, (
            f"seed={seed}/{tier}: oracle {oracle_answer} != rendered answer {q.answer_text}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        oracle_answer = _oracle(q, tier)
        for d in q.distractors:
            assert int(d) != oracle_answer, f"seed={seed}/{tier}: distractor {d!r} equals oracle answer"
