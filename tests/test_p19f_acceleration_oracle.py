"""Reads only `question.text`/`answer_text` via `mathgen.make`. Imports NOTHING
from `mathgen.generators.p19f_acceleration_build` -- the position function
s(t) and the evaluation time are parsed back out of the rendered prose with
regular expressions, then re-differentiated twice with `sympy.diff`, never
by calling into the generator's own construction.
"""

import re

import pytest
from sympy import diff, symbols, sympify

from mathgen import make

SEEDS = range(200)
TIERS = ("easy", "medium", "hard", "exam")
TOPIC = "p19f_acceleration_second_derivative"

t = symbols("t")

_S_RE = re.compile(r"s\(t\) = ([-+0-9t^ ]+), in metres")
_AT_RE = re.compile(r"at t = (\d+) (?:second|seconds)")
_EXAM_RE = re.compile(
    r"acceleration (\d+) (?:second|seconds) after (?:it passes |the particle passes )?"
    r"t = (\d+) (?:second|seconds)"
)


def _parse_s(text: str):
    m = _S_RE.search(text)
    assert m is not None, f"could not parse s(t) from: {text!r}"
    expr_text = m.group(1).strip().replace("^", "**")
    # "2t^2" -> "2*t**2": insert an explicit multiplication sign between a
    # digit and the variable, since sympify has no implicit-multiplication
    # transformation enabled here.
    expr_text = re.sub(r"(\d)(t)", r"\1*\2", expr_text)
    return sympify(expr_text, locals={"t": t})


def _parse_point(text: str, tier: str) -> int:
    if tier == "exam":
        m = _EXAM_RE.search(text)
        assert m is not None, f"could not parse exam time relation from: {text!r}"
        k, p = int(m.group(1)), int(m.group(2))
        return p + k
    m = _AT_RE.search(text)
    assert m is not None, f"could not parse evaluation time from: {text!r}"
    return int(m.group(1))


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_rendered_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        s_expr = _parse_s(q.text)
        point = _parse_point(q.text, tier)
        oracle_val = int(diff(s_expr, t, 2).subs(t, point))
        rendered = int(q.answer_text)
        assert oracle_val == rendered, (
            f"seed={seed}/{tier}: oracle {oracle_val} != rendered answer {rendered}\ntext: {q.text}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        s_expr = _parse_s(q.text)
        point = _parse_point(q.text, tier)
        oracle_val = int(diff(s_expr, t, 2).subs(t, point))
        for d in q.distractors:
            assert int(d) != oracle_val, f"seed={seed}/{tier}: distractor {d!r} equals the oracle answer {oracle_val}"
