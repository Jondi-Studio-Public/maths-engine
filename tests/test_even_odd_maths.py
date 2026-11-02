"""Independent-oracle tests for the even/odd subject maths (no recipe glue)."""

import itertools

import pytest
from sympy import Rational, pi, simplify

from mathgen.generators import _even_odd_maths as m

TERMS = [(c, kind, p) for c in (2, -3, Rational(1, 3)) for kind in ("pow", "sin", "cos") for p in (1, 2, 3)] + [
    (5, "const", 0)
]


@pytest.mark.parametrize("a,b", list(itertools.combinations(TERMS, 2))[:400])
def test_function_parity_matches_numeric(a, b):
    terms = [a, b]
    expr = m.term_expr(a) + m.term_expr(b)
    assert m.function_parity(terms) == m.numeric_parity(expr)


@pytest.mark.parametrize("a,b", list(itertools.combinations(TERMS, 2))[:200])
def test_combine_parity_sum_and_product(a, b):
    pa, pb = m.term_parity(a[1], a[2]), m.term_parity(b[1], b[2])
    fa, fb = m.term_expr(a), m.term_expr(b)
    assert m.combine_parity(pa, pb, "*") == m.numeric_parity(fa * fb)
    if pa == pb:
        assert m.combine_parity(pa, pb, "+") == m.numeric_parity(fa + fb)


@pytest.mark.parametrize("trig", ["sin", "cos"])
@pytest.mark.parametrize("parity", [m.EVEN, m.ODD])
@pytest.mark.parametrize("b", [1, 2, 3])
@pytest.mark.parametrize("k", [-2, -1, 0, 1, 3])
def test_phase_family_members_have_the_parity(trig, parity, b, k):
    c = simplify(m.phase_family(trig, parity).subs(m.K, k))
    assert m.numeric_parity(m.shifted_expr(trig, b, c)) == parity


@pytest.mark.parametrize("trig", ["sin", "cos"])
@pytest.mark.parametrize("parity", [m.EVEN, m.ODD])
@pytest.mark.parametrize("b", [1, 2])
def test_phase_family_is_complete(trig, parity, b):
    # c off the family (shifted by a non-multiple of pi/2... pi/4 and pi/3) is never that parity
    for k in (0, 1):
        for off in (pi / 4, pi / 3, pi / 6):
            c = simplify(m.phase_family(trig, parity).subs(m.K, k) + off)
            assert m.numeric_parity(m.shifted_expr(trig, b, c)) != parity
