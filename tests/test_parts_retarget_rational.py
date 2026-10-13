"""Follow-through into a fraction part keeps an exact rational target exact.

`sympy.nsimplify` rewrites some large rationals as products of surds (2567/795 becomes
6*2**(29/393)*...), so a fraction part's follow-through target used to be lost for those values.
"""

from __future__ import annotations

import pytest
import sympy

from mathgen.core import Part
from mathgen.grading.parts import follow_through_target


def _fraction_part(follow_through: str) -> Part:
    return Part(
        label="b",
        prompt="Find the ratio.",
        answer_kind="fraction",
        answer={"p": 1, "q": 2},
        answer_text="1/2",
        follow_through=follow_through,
    )


@pytest.mark.parametrize(("p", "q"), [(2567, 795), (4212, 361), (1, 2), (-7, 3)])
def test_rational_target_stays_exact(p, q):
    target = follow_through_target(_fraction_part("part_a/795"), {"a": sympy.Integer(p) * 795 / q})
    assert target is not None
    assert target.answer == {"p": p, "q": q}


def test_float_value_still_simplified():
    target = follow_through_target(_fraction_part("part_a"), {"a": sympy.Float("0.25")})
    assert target is not None
    assert target.answer == {"p": 1, "q": 4}
