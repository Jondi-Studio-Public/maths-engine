"""Independent oracle for solve-quadratics-formula's rearrangement-first hard tier. Re-derives from rendered
question text alone -- does not import mathgen.generators.algebra."""

from __future__ import annotations

import re

from sympy import Eq, solve, symbols

from mathgen import make

x = symbols("x")

_PATTERN = re.compile(r"Solve x\((-?\d*)x ([+-]) (\d+)\) = (-?\d+)")


def test_oracle_recomputes_the_roots_from_the_question_text():
    checked = 0
    for seed in range(300):
        q = make("quadratic", seed=seed, kind="formula_rearrange")
        m = _PATTERN.search(q.text)
        assert m, f"unexpected question text shape: {q.text!r}"
        a_str = m.group(1)
        a_val = int(a_str) if a_str not in ("", "-") else (-1 if a_str == "-" else 1)
        b_val = int(m.group(3)) if m.group(2) == "+" else -int(m.group(3))
        rhs = int(m.group(4))
        # Rearrange: x(ax + b) = rhs  ->  a*x^2 + b*x - rhs = 0
        roots = sorted(float(r) for r in solve(Eq(a_val * x**2 + b_val * x - rhs, 0), x))
        rendered = q.answer_text.replace("x = ", "").split(" or ")
        rendered_vals = sorted(float(v) for v in rendered)
        assert len(roots) == 2 and len(rendered_vals) == 2, (
            f"seed={seed}: expected two real roots, got roots={roots} rendered={rendered_vals}"
        )
        for expected, actual in zip(roots, rendered_vals):
            assert abs(expected - actual) < 0.01, f"seed={seed}: {q.text!r} -> {q.answer_text!r} expected {roots}"
        checked += 1
    assert checked == 300
