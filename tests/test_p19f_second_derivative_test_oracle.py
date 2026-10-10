"""Reads only `question.text`/`answer`/`answer_text`/`distractors` via
`mathgen.make`. Imports NOTHING from
`mathgen.generators.p19f_second_derivative_test_build` -- f(x) is parsed
back out of the rendered prose with a regular expression, then the critical
point and its classification are re-derived from scratch with
`sympy.solve`/`sympy.diff`, never by calling into the generator's own
construction.
"""

import re

import pytest
from sympy import diff, solve, symbols, sympify

from mathgen import make

SEEDS = range(200)
TIERS = ("easy", "medium", "hard", "exam")
TOPIC = "p19f_second_derivative_test"

x = symbols("x")

_F_RE = re.compile(r"f\(x\) = ([-+*^·0-9x() ]+)")


def _parse_f(text: str):
    m = _F_RE.search(text)
    assert m is not None, f"could not parse f(x) from: {text!r}"
    expr_text = m.group(1).strip().replace("^", "**").replace("·", "*")
    return sympify(expr_text, locals={"x": x})


def _oracle_verdict(f):
    """Independent classification: solve f'(x) = 0 with sympy, require
    exactly one real root, and classify by the sign of f'' there."""
    roots = solve(diff(f, x), x)
    real_roots = [r for r in roots if r.is_real]
    assert len(real_roots) == 1, f"expected exactly one real root, got {roots}"
    x0 = real_roots[0]
    second = diff(f, x, 2).subs(x, x0)
    assert second != 0, "f''(x0) must not be zero"
    label = "local minimum" if second > 0 else "local maximum"
    return int(x0), label


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_rendered_answer(tier):
    matched = 0
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        f = _parse_f(q.text)
        oracle_x0, oracle_label = _oracle_verdict(f)
        rendered_x0, rendered_label = q.answer
        assert oracle_x0 == rendered_x0, (
            f"seed={seed}/{tier}: oracle x0 {oracle_x0} != rendered {rendered_x0}\ntext: {q.text}"
        )
        assert oracle_label == rendered_label, (
            f"seed={seed}/{tier}: oracle label {oracle_label!r} != rendered {rendered_label!r}"
        )
        # The artefact the learner actually receives, not just the
        # structured `.answer` -- the inverse_one_to_one_function lesson.
        assert q.answer_text == f"{oracle_x0}, {oracle_label}", (
            f"seed={seed}/{tier}: answer_text {q.answer_text!r} != oracle {oracle_x0}, {oracle_label}"
        )
        matched += 1
    assert matched == len(SEEDS)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_rejects_every_distractor(tier):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=tier)
        f = _parse_f(q.text)
        oracle_x0, oracle_label = _oracle_verdict(f)
        oracle_text = f"{oracle_x0}, {oracle_label}"
        for d in q.distractors:
            assert d != oracle_text, f"seed={seed}/{tier}: distractor {d!r} equals the oracle answer {oracle_text!r}"
