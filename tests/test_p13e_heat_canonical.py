"""The p13e generator skips `simplify` where it provably changes nothing; this re-checks a sample of those claims.

`mathgen/generators/p13e_residue_heat_build.py` prints heat-equation modes and exam-tier complex roots without
calling `sympy.simplify` on them, because `simplify` returns each of them unchanged and costs tens of
milliseconds. The claim was checked exhaustively when it was made (all 12,150 expressions `_mode` can build);
this test checks a spread of them, so a sympy upgrade that changes `simplify` fails here and not as a changed
answer string.
"""

import itertools

import pytest
import sympy as sp
from sympy import I

from mathgen.generators import p13e_residue_heat_build as module


def _same(simplified, original):
    return simplified == original and str(simplified) == str(original)


@pytest.mark.parametrize("family", ["dirichlet", "neumann"])
@pytest.mark.parametrize("square_n, use_alpha", [(True, True), (False, True), (True, False)])
def test_simplify_leaves_a_heat_mode_unchanged(family, square_n, use_alpha):
    pool = itertools.product([1, 7], [1, 4], [2, 9, sp.pi], [1, 5])
    for coeff, n, length, alpha in pool:
        mode = module._mode(coeff, family, n, length, alpha, square_n=square_n, use_alpha=use_alpha)
        assert _same(sp.simplify(mode), mode), mode
        assert module._heat_fmt(mode) == module._fmt(mode)
        assert module._heat_text_fmt(mode) == module._text_fmt(mode)


def test_simplify_leaves_an_integer_complex_number_unchanged():
    for p, q in itertools.product([-8, -1, 3, 8], [1, 2, 6]):
        for value in (p + q * I, p - q * I, (p + 1) + q * I):
            assert module._is_integer_gaussian(value) == isinstance(value, sp.Add)  # (-1 + 1) + q*I is just q*I
            assert _same(sp.simplify(value), value), value


def test_a_quotient_of_complex_numbers_is_still_simplified():
    value = (3 + 2 * I - 2) / (2 * I * 2)
    assert not module._is_integer_gaussian(value)
    assert module.simplify(value) == sp.simplify(value)
