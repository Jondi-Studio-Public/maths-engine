"""Reads only `question.text`/`answer`/`answer_text`/`distractors` via
`mathgen.make`. Imports NOTHING from
`mathgen.generators.p19f_concavity_inflection_build` -- f(x) is parsed back
out of the rendered prose with a regular expression, then the inflection
point and its concavity direction are re-derived from scratch with
`sympy.solve`/`sympy.diff`, never by calling into the generator's own
construction.
"""

import re

import pytest
from sympy import diff, solve, symbols, sympify

from mathgen import make

SEEDS = range(200)
TIERS = ("easy", "medium", "hard", "exam")
TOPIC = "p19f_concavity_inflection"

x = symbols("x")

_F_RE = re.compile(r"[fRQ]\(x\) = ([-+*^·0-9x() ]+)")


def _parse_f(text: str):
    m = _F_RE.search(text)
    assert m is not None, f"could not parse f(x) from: {text!r}"
    # A trailing unmatched "(" is a follow-on parenthetical in the sentence
    # ("... (not yet simplified)"), not part of the expression -- strip it
    # off along with the punctuation/whitespace immediately before it.
    expr_text = m.group(1).strip()
    while expr_text.count("(") > expr_text.count(")"):
        expr_text = expr_text.rsplit("(", 1)[0].rstrip(", ").strip()
    expr_text = expr_text.replace("^", "**").replace("·", "*")
    return sympify(expr_text, locals={"x": x})


def _oracle_verdict(f):
    """Independent classification: solve f''(x) = 0 with sympy, require
    exactly one real root, and classify direction by the sign of f''' there
    (a linear f'' means f''' is a nonzero constant that decides the sign
    change unconditionally)."""
    second = diff(f, x, 2)
    roots = solve(second, x)
    real_roots = [r for r in roots if r.is_real]
    assert len(real_roots) == 1, f"expected exactly one real root, got {roots}"
    x0 = real_roots[0]
    third = diff(f, x, 3)
    assert third != 0, "f'''(x) must not be zero"
    direction = "concave down to concave up" if third > 0 else "concave up to concave down"
    return int(x0), direction


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_rendered_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        f = _parse_f(q.text)
        oracle_x0, oracle_direction = _oracle_verdict(f)
        rendered_x0, rendered_direction = q.answer
        assert oracle_x0 == rendered_x0, (
            f"seed={seed}/{tier}: oracle x0 {oracle_x0} != rendered {rendered_x0}\ntext: {q.text}"
        )
        assert oracle_direction == rendered_direction, (
            f"seed={seed}/{tier}: oracle direction {oracle_direction!r} != rendered {rendered_direction!r}"
        )
        assert q.answer_text == f"{oracle_x0}, {oracle_direction}", (
            f"seed={seed}/{tier}: answer_text {q.answer_text!r} != oracle {oracle_x0}, {oracle_direction}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        f = _parse_f(q.text)
        oracle_x0, oracle_direction = _oracle_verdict(f)
        oracle_text = f"{oracle_x0}, {oracle_direction}"
        for d in q.distractors:
            assert d != oracle_text, f"seed={seed}/{tier}: distractor {d!r} equals the oracle answer {oracle_text!r}"
