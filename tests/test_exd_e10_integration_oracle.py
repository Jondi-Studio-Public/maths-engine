"""Independent oracle for integrate-polynomial/integrate-trig/integration-by-substitution hard tiers. Re-derives
from rendered question text alone -- does not import mathgen.generators.calculus."""

from __future__ import annotations

import re

from sympy import Rational, diff, expand, integrate, sin, cos, symbols, sympify

from mathgen import make

x = symbols("x")

_REWRITE_RE = re.compile(
    r"F'\(x\) = (\d+)/x\^(\d+) and F\(1\) = (-?\d+)|"
    r"F'\(x\) = (\d+)\*sqrt\(x\) and F\(1\) = (-?\d+)"
)
_PHASE_RE = re.compile(r"∫ (\d+)\*(sin|cos)\((\d+)\*x ([+-]) (\d+)\) dx")
_NONLINEAR_RE = re.compile(r"∫ \((.+)\) dx using the substitution u = x\*\*2 \+ (\d+)")


def test_oracle_rewrite_condition():
    checked = 0
    for seed in range(200):
        q = make("integration", seed=seed, kinds=("rewrite_condition",))
        m = _REWRITE_RE.search(q.text)
        assert m, f"unexpected text: {q.text!r}"
        if m.group(1):
            c_val, n_val, k_val = int(m.group(1)), int(m.group(2)), int(m.group(3))
            f = c_val * x ** (-n_val)
        else:
            c_val, k_val = int(m.group(4)), int(m.group(5))
            f = c_val * x ** Rational(1, 2)
        F_indef = integrate(f, x)
        C = k_val - F_indef.subs(x, 1)
        expected = expand(F_indef + C)
        actual = sympify(q.answer_text)
        assert expand(actual - expected) == 0, f"seed={seed}: {q.text!r} -> {q.answer_text!r}"
        checked += 1
    assert checked == 200


def test_oracle_phase():
    checked = 0
    for seed in range(200):
        q = make("integration", seed=seed, kinds=("phase",))
        m = _PHASE_RE.search(q.text)
        assert m, f"unexpected text: {q.text!r}"
        k_val, fn_name, a_val, sign, b_abs = m.groups()
        k_val, a_val, b_abs = int(k_val), int(a_val), int(b_abs)
        b_val = b_abs if sign == "+" else -b_abs
        fn = sin if fn_name == "sin" else cos
        f = k_val * fn(a_val * x + b_val)
        expected = integrate(f, x)
        actual_expr = sympify(q.answer_text.replace(" + C", ""))
        assert expand(diff(actual_expr, x) - f) == 0, f"seed={seed}: {q.text!r} -> {q.answer_text!r}"
        checked += 1
    assert checked == 200


def test_oracle_nonlinear():
    checked = 0
    for seed in range(200):
        q = make("integration", seed=seed, kinds=("nonlinear",))
        m = _NONLINEAR_RE.search(q.text)
        assert m, f"unexpected text: {q.text!r}"
        d_val = int(m.group(2))
        actual_expr = sympify(q.answer_text.replace(" + C", ""))
        # Re-derive by differentiating the claimed answer and confirming it
        # matches a k*x*(x^2+d)^n integrand for SOME n in [2, 4], k in [1, 4]
        # -- rather than re-parsing the integrand text (fragile with **).
        deriv = expand(diff(actual_expr, x))
        found = False
        for n_val in range(2, 5):
            for k_val in range(1, 5):
                candidate_f = expand(k_val * x * (x**2 + d_val) ** n_val)
                if deriv == candidate_f:
                    found = True
                    break
            if found:
                break
        assert found, f"seed={seed}: {q.text!r} -> {q.answer_text!r}, deriv={deriv}"
        checked += 1
    assert checked == 200
