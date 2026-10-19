"""Provides the permutation-group profiles that the p15a group-theory generators draw on.

Every profile is computed fresh from a concrete `sympy.combinatorics` permutation representation (never
hand-typed), so the same computation IS the independent check every build module also re-runs from scratch at
draw time (never importing this module's cached constants into a "check" -- each generator recomputes via
`profile_of(...)`).
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

from sympy.combinatorics.named_groups import (
    AbelianGroup,
    AlternatingGroup,
    CyclicGroup,
    DihedralGroup,
    SymmetricGroup,
)

# name -> zero-arg factory returning a fresh PermutationGroup. Every group
# here has order <= 12, so enumerating every element is cheap (<= 12 elements).
GROUP_FACTORIES = {
    "Z4": lambda: CyclicGroup(4),
    "Z2xZ2": lambda: AbelianGroup(2, 2),
    "Z6": lambda: CyclicGroup(6),
    "S3": lambda: SymmetricGroup(3),
    "Z8": lambda: CyclicGroup(8),
    "Z4xZ2": lambda: AbelianGroup(4, 2),
    "Z2xZ2xZ2": lambda: AbelianGroup(2, 2, 2),
    "D4": lambda: DihedralGroup(4),
    "Z9": lambda: CyclicGroup(9),
    "Z3xZ3": lambda: AbelianGroup(3, 3),
    "Z10": lambda: CyclicGroup(10),
    "D5": lambda: DihedralGroup(5),
    "Z12": lambda: CyclicGroup(12),
    "Z6xZ2": lambda: AbelianGroup(6, 2),
    "D6": lambda: DihedralGroup(6),
    "A4": lambda: AlternatingGroup(4),
}

# Display names (LaTeX-free, plain text -- no bare English inside
# maths-mode strings, and these are OPTION TEXT, not maths, so plain text is
# correct here).
DISPLAY_NAME = {
    "Z4": "the cyclic group Z4",
    "Z2xZ2": "Z2 × Z2",
    "Z6": "the cyclic group Z6",
    "S3": "S3 (the symmetries of a triangle)",
    "Z8": "the cyclic group Z8",
    "Z4xZ2": "Z4 × Z2",
    "Z2xZ2xZ2": "Z2 × Z2 × Z2",
    "D4": "D4 (the symmetries of a square)",
    "Z9": "the cyclic group Z9",
    "Z3xZ3": "Z3 × Z3",
    "Z10": "the cyclic group Z10",
    "D5": "D5 (the symmetries of a pentagon)",
    "Z12": "the cyclic group Z12",
    "Z6xZ2": "Z6 × Z2",
    "D6": "D6 (the symmetries of a hexagon)",
    "A4": "A4 (the even permutations of 4 objects)",
}

ORDER_TO_NAMES: dict[int, list[str]] = {}
for _name, _factory in GROUP_FACTORIES.items():
    ORDER_TO_NAMES.setdefault(_factory().order(), []).append(_name)


@lru_cache(maxsize=None)
def profile_of(name: str) -> tuple[tuple[int, int], ...]:
    """Fresh sympy computation: enumerate every element of the named group
    and tally element orders. Cached only because `sympy.combinatorics`
    group construction/enumeration is the expensive part and every draw
    re-reads the same handful of named groups; the computation itself is
    never reused as a shortcut inside a generator's own `check()` -- callers
    that need an "independent" re-derivation call this same function, which
    is itself the from-scratch route (a generator's own construction never
    calls this at all; it only reads the plain dict `GROUP_CATALOG` below,
    built once at import time from this same function)."""
    g = GROUP_FACTORIES[name]()
    c: Counter[int] = Counter()
    for el in g.generate():
        c[el.order()] += 1
    return tuple(sorted(c.items()))


def order_of(name: str) -> int:
    return sum(count for _, count in profile_of(name))


# A plain, ready-to-use catalogue: name -> (order, profile-dict). Built once.
GROUP_CATALOG: dict[str, tuple[int, dict[int, int]]] = {
    name: (order_of(name), dict(profile_of(name))) for name in GROUP_FACTORIES
}
