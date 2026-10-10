"""Reads only `question.text` / `answer_text` / `distractors` via `mathgen.make`.
Imports NOTHING from `mathgen.generators.p19f_instantaneous_rate_build` --
the polynomial, the point and (on the exam tier) the offset are all parsed
back out of the rendered prose with regular expressions, then differentiated
and evaluated with a fresh `sympy.diff` call written here.
"""

import re

import pytest
from sympy import diff, sympify, symbols

from mathgen import make

SEEDS = range(200)
TIERS = ("easy", "medium", "hard", "exam")

TOPIC = "p19f_instantaneous_rate"

_x = symbols("x")
_t = symbols("t")


def _parse_poly(prose: str, symbol, name: str):
    py_text = prose.replace("^", "**").replace("·", "*")
    return sympify(py_text, locals={name: symbol})


_EASY_MEDIUM_RE = re.compile(
    r"^f\(x\) = (?P<f>.+?)\. Find the instantaneous rate of change of f at x = (?P<p>-?\d+)\.$"
)
_HARD_RE = re.compile(r"Q\(t\) = (?P<f>.+?)\. Find the instantaneous rate of change of Q when t = (?P<p>-?\d+)\.$")
_EXAM_RE = re.compile(
    r"Q\(t\) = (?P<f>.+?)\. Find the instantaneous rate of change of Q, "
    r"(?P<offset>\d+) (?:minutes|seconds|hours) after t = (?P<p>-?\d+)\.$"
)


def _oracle(question, tier):
    text = question.text
    if tier in ("easy", "medium"):
        m = _EASY_MEDIUM_RE.search(text)
        assert m is not None, f"could not parse {tier} text: {text!r}"
        f_expr = _parse_poly(m.group("f"), _x, "x")
        point = int(m.group("p"))
        return int(diff(f_expr, _x).subs(_x, point))
    if tier == "hard":
        m = _HARD_RE.search(text)
        assert m is not None, f"could not parse hard text: {text!r}"
        f_expr = _parse_poly(m.group("f"), _t, "t")
        point = int(m.group("p"))
        return int(diff(f_expr, _t).subs(_t, point))
    m = _EXAM_RE.search(text)
    assert m is not None, f"could not parse exam text: {text!r}"
    f_expr = _parse_poly(m.group("f"), _t, "t")
    point = int(m.group("p")) + int(m.group("offset"))
    return int(diff(f_expr, _t).subs(_t, point))


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
