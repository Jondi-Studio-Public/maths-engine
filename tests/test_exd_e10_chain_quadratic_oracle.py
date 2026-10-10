"""Independent oracle for differentiate-chain-rule's chain_quadratic hard tier. Re-derives the answer from the
rendered question text alone -- does not import mathgen.generators.calculus."""

from __future__ import annotations

import re

from sympy import diff, expand, sympify, symbols

from mathgen import make

x = symbols("x")

_PATTERN = re.compile(r"\(x\*\*2 ([+-]) (\d+)\)\*\*(\d+)")


def test_oracle_recomputes_the_derivative_from_the_question_text():
    checked = 0
    for seed in range(200):
        q = make("differentiation", seed=seed, kinds=("chain_quadratic",))
        m = _PATTERN.search(q.text)
        assert m, f"unexpected question text shape: {q.text!r}"
        sign, c_str, n_str = m.group(1), m.group(2), m.group(3)
        c_val = int(c_str) if sign == "+" else -int(c_str)
        n_val = int(n_str)
        expected = expand(diff((x**2 + c_val) ** n_val, x))
        actual = sympify(q.answer_text)
        assert expand(actual - expected) == 0, (
            f"seed={seed}: question={q.text!r} answer_text={q.answer_text!r} expected={expected}"
        )
        checked += 1
    assert checked == 200
