"""Option balancing for the `w5b_*` builds: wrong options reordered so the key has no fixed place among them.

The named slips of these builds are derived from the key, so the key tends to sit among them (the middle option, the
one nearest the median). `balanced` hands the slips to `_spread_kit.mixed` with plain near-miss fillers drawn around
the key, so that the number of shown options below the key is uniform over the shown set. It registers nothing; the
generator that calls it must be wrapped in `uses_rng` so the split comes from the question's own seed.
"""

from __future__ import annotations

import sympy

from ._spread_kit import mixed

NEAR = "dg-near-miss-value"
_NO_BOUND = "expression"


def fillers_for(value, *, whole: bool = False) -> list:
    """Plain near misses of `value`, of the same shape: integers stay whole, a square root stays a square root."""
    v = sympy.sympify(value)
    out = []
    if v.is_Rational:
        steps = (
            (-3, -2, -1, 1, 2, 3, 4)
            if whole or v.is_Integer
            else (-2, -1, -sympy.Rational(1, 2), sympy.Rational(1, 2), 1, 2, 3)
        )
        out = [v + d for d in steps]
    elif v.is_Pow and v.exp == sympy.Rational(1, 2) and v.base.is_Integer:
        out = [sympy.sqrt(v.base + d) for d in (-4, -3, -2, -1, 1, 2, 3, 4) if v.base + d > 0]
    else:
        base = v / sympy.pi if v.has(sympy.pi) else None
        if base is not None and sympy.nsimplify(base).is_Rational:
            out = [sympy.pi * (base + d) for d in (-3, -2, -1, 1, 2, 3, 4)]
        else:
            out = [v * sympy.Rational(n + d, n) for n in (4, 5, 6) for d in (-1, 1)]
    return [(sympy.sympify(f), NEAR) for f in out if f != v and sympy.sympify(f).is_number]


def balanced(value, wrongs, *, count: int = 4, lo=0, hi=None, whole: bool = False) -> list:
    """`wrongs` (a list of `(value, misconception_id)`) reordered so that its first `count` hold a random number of
    values below the key. Fillers make up a side with too few named slips."""
    wrongs = list(wrongs)
    return mixed(
        sympy.sympify(value),
        wrongs,
        count,
        fillers_for(value, whole=whole),
        lo=lo,
        hi=hi,
        unbounded=_NO_BOUND if lo is None and hi is None else None,
    )
