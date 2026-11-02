"""Independent-oracle tests for mathgen.subjects.exponential_g (sympy / numeric, not the module's own formulas)."""

import math
import random

import pytest
import sympy

from mathgen.subjects import exponential_g as eg


def _g(s, x):
    return s.a ** (2 * x) - s.b * s.a**x + s.c


def test_vce_2023_case():
    s = eg.ExpG(2, 4, 5)
    assert (s.b, s.c, s.m) == (9, 20, 2)
    assert s.g_at_log(3) == 2


@pytest.mark.parametrize("seed", range(300))
def test_draw_and_asks(seed):
    rng = random.Random(seed)
    s = eg.draw(rng)
    # roots of g found numerically by the independent route
    assert abs(_g(s, s.m)) < 1e-9
    assert abs(_g(s, s.log_root_value())) < 1e-6
    assert s.log_root_value() != int(s.log_root_value())  # genuinely a log
    t = sympy.symbols("t")
    assert sorted(sympy.solve(t**2 - s.b * t + s.c, t)) == sorted([s.r1, s.r2])
    k = eg.eval_k(rng, s)
    assert _g(s, math.log(k, s.a)) == pytest.approx(s.g_at_log(k), abs=1e-6)
    assert s.g_at_log(k) != 0
    key = s.g_at_log(k)
    assert len({w for w, _ in eg.eval_wrong(s, k) if w != key}) >= 3  # enough distinct wrong answers
