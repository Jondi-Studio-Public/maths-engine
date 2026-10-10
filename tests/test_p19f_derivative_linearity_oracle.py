"""Reads only `question.text` / `answer_text` / `distractors` via `mathgen.make`.
Imports NOTHING from `mathgen.generators.p19f_derivative_linearity_build` --
f'(x), g'(x), a and b are all parsed back out of the rendered prose with
regular expressions, then combined with sympy written fresh here.
"""

import re

import pytest
from sympy import Symbol, expand
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from mathgen import make

SEEDS = range(200)
TIERS = ("easy", "medium", "hard", "exam")

_x = Symbol("x")
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

_HEAD_RE = re.compile(r"^f'\(x\) = (?P<f>.+?) and g'\(x\) = (?P<g>.+?)\. h\(x\)")
# Numeric-coefficient tiers (easy/medium/hard): "h(x) = <af-term>f(x) <sign> <b>g(x)."
# `aterm`/`bmag` may be a bare digit run, or empty/"-" when `fmt_signed` drops
# a unit coefficient (coefficient +-1 renders as "f(x)"/"-f(x)", "g(x)").
_NUMERIC_TAIL_RE = re.compile(r"= (?P<aterm>-?\d*)f\(x\) (?P<sign>[+-]) (?P<bmag>\d*)g\(x\)\.")
# Word tier (exam): "h(x) is <a_word> f(x) <plus|minus> <b_word> g(x)."
_WORD_TAIL_RE = re.compile(r"is (?P<aword>[a-z ]+?) f\(x\) (?P<connective>plus|minus) (?P<bword>[a-z ]+?) g\(x\)\.")

_WORD_TO_NUM = {
    "once": 1,
    "twice": 2,
    "three times": 3,
    "four times": 4,
    "five times": 5,
    "six times": 6,
}


def _parse_poly(prose: str):
    """A rendered polynomial-in-x string ("27x^2 - 4", "x + 5", "-9") back
    into a sympy expression. The generator never emits a bare multiplication
    dot for this skill (no `·`), only implicit juxtaposition, so implicit
    multiplication handles it directly once `^` is swapped for `**`."""
    return parse_expr(prose.replace("^", "**"), transformations=_TRANSFORMATIONS)


def _oracle(question):
    head = _HEAD_RE.search(question.text)
    assert head is not None, f"could not parse head from: {question.text!r}"
    f_expr = _parse_poly(head.group("f"))
    g_expr = _parse_poly(head.group("g"))

    numeric = _NUMERIC_TAIL_RE.search(question.text)
    if numeric is not None:
        aterm = numeric.group("aterm")
        a = 1 if aterm == "" else (-1 if aterm == "-" else int(aterm))
        bmag_text = numeric.group("bmag")
        b_mag = 1 if bmag_text == "" else int(bmag_text)
        b = b_mag if numeric.group("sign") == "+" else -b_mag
    else:
        word = _WORD_TAIL_RE.search(question.text)
        assert word is not None, f"could not parse tail from: {question.text!r}"
        a = _WORD_TO_NUM[word.group("aword").strip()]
        b_mag = _WORD_TO_NUM[word.group("bword").strip()]
        b = b_mag if word.group("connective") == "plus" else -b_mag

    return expand(a * f_expr + b * g_expr)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make("p19f_derivative_linearity", seed=seed, difficulty=tier)
        oracle_answer = _oracle(q)
        rendered = _parse_poly(q.answer_text)
        assert expand(oracle_answer - rendered) == 0, (
            f"seed={seed}/{tier}: oracle {oracle_answer} != rendered answer {rendered}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make("p19f_derivative_linearity", seed=seed, difficulty=tier)
        oracle_answer = _oracle(q)
        for d in q.distractors:
            parsed_d = _parse_poly(d)
            assert expand(oracle_answer - parsed_d) != 0, (
                f"seed={seed}/{tier}: distractor {d!r} equals oracle answer\ntext: {q.text}"
            )
