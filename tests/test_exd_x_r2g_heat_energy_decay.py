"""Tests for the multi-part exam source of ucv1-energy-estimate: decay constant, initial energy, initial energy
rate and the time for the energy to fall by a factor. Keys are re-derived from the rendered stem with sympy."""

from __future__ import annotations

import math
import re
from fractions import Fraction

import sympy

import mathgen.generators.exd_x_r2g_heat_energy_decay  # noqa: F401  (registers the topic)
from tests.exd_x_r2g_support import follow_through_check, standard_checks

TOPIC = "exd_x_r2g_heat_energy_decay"
SKILL = "ucv1-energy-estimate"
LABELS = ["a", "b", "c", "d"]
MARKS = [2, 1, 1, 2]
X = sympy.Symbol("x")


def _text(v) -> str:
    v = sympy.Rational(v)
    return str(v.p) if v.q == 1 else f"{v.p}/{v.q}"


def _facts(q):
    k = int(re.search(r"u_t = (\d+) u_xx", q.stem).group(1))
    length = int((re.search(r"\[0, (\d+)\]", q.stem) or re.search(r"length (\d+) cm", q.stem)).group(1))
    m = re.search(r"u\(x, 0\) = (\d+) sin\((\d*)pi x/(\d+)\)", q.stem)
    amp, num, den = int(m.group(1)), int(m.group(2) or 1), int(m.group(3))
    ratio = int(re.search(r"E\(0\)/(\d+)", q.text).group(1))
    return k, length, amp, sympy.Rational(num, den), ratio


def _solve(q):
    k, length, amp, kp, ratio = _facts(q)
    kappa = kp * sympy.pi
    energy0 = sympy.integrate((amp * sympy.sin(kappa * X)) ** 2, (X, 0, length))
    c = k * kp**2
    rate = -2 * c * energy0
    t = sympy.N(sympy.log(ratio) / (2 * c * sympy.pi**2), 30)
    return c, sympy.nsimplify(energy0), rate, t, (k, length, amp, kp, ratio)


def _three_dp(t) -> str:
    return f"{math.floor(Fraction(str(t)) * 1000 + Fraction(1, 2)) / 1000:.3f}"


def oracle(q):
    c, energy0, rate, t, _ = _solve(q)
    return {"a": _text(c), "b": _text(energy0), "c": _text(rate), "d": _three_dp(t)}


CHECKS = standard_checks(TOPIC, SKILL, oracle, LABELS, MARKS, {"c", "d"}, "p18a_energy_estimate")


def test_source_wired():
    CHECKS["wired"]()


def test_shape_and_text():
    CHECKS["shape"]()


def test_prompt_shapes_beyond_hard_tier():
    CHECKS["shapes"]()


def test_keys_match_independent_derivation():
    CHECKS["oracle_agrees"]()


def test_composed_answer_earns_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_rejected_and_tagged():
    CHECKS["distractors_rejected"]()


def test_answer_space():
    CHECKS["answer_space"]()


def test_deterministic():
    CHECKS["deterministic"]()


def test_end_conditions_hold_for_the_profile():
    """The stated profile satisfies the stated end conditions, so it is a genuine eigenmode."""
    from tests.exd_x_r2g_support import exam

    for s in range(120):
        q = exam(TOPIC, s)
        _k, length, _amp, kp, _ratio = _facts(q)
        kappa_l = kp * sympy.pi * length
        if "Both ends" in q.stem:
            assert sympy.simplify(sympy.sin(kappa_l)) == 0, q.stem
        else:
            assert sympy.simplify(sympy.cos(kappa_l)) == 0, q.stem


def test_follow_through_from_squared_wavenumber_slip():
    """A decay constant that forgot the diffusivity is carried through (c) and (d)."""

    def wrong_chain(q):
        c, energy0, _rate, _t, (k, _length, _amp, kp, ratio) = _solve(q)
        wrong_c = kp**2  # forgot k
        wrong_rate = -2 * wrong_c * energy0
        wrong_t = sympy.N(sympy.log(ratio) / (2 * wrong_c * sympy.pi**2), 30)
        replies = {"a": _text(wrong_c), "b": _text(energy0), "c": _text(wrong_rate), "d": _three_dp(wrong_t)}
        return "a", replies, 1 + 1 + 2

    follow_through_check(TOPIC, SKILL, wrong_chain)
