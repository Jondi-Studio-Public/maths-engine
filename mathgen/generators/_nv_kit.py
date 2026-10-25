"""Shared option ordering for the `nv_*` builds: wrong values chosen so the key has no fixed rank among its options.

The named slips come first. How many of the chosen ones lie below the key is drawn at random from what the slips allow;
plain near misses (`NEAR`) are used only for a side the slips cannot fill. Nothing here registers a topic.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ._spread_kit import _RNG, NEAR


def spread_order(key, cands, fillers=(), count: int = 3, *, lo=None, hi=None):
    """`cands` ([(value, tag)]) reordered so that the first `count` hold a random number below the key; fillers top up."""
    rng = _RNG.get() or random.Random(repr((str(key), [str(c[0]) for c in cands])))
    k = Fraction(key)

    def ok(v):
        return v is not None and Fraction(v) != k and (lo is None or v >= lo) and (hi is None or v <= hi)

    seen, named = set(), []
    for v, tag in cands:
        if ok(v) and Fraction(v) not in seen:
            seen.add(Fraction(v))
            named.append((v, tag))
    near = []
    for v, tag in fillers:
        if ok(v) and Fraction(v) not in seen:
            seen.add(Fraction(v))
            near.append((v, tag))
    nearer = sorted(near, key=lambda c: abs(Fraction(c[0]) - k))
    below = [c for c in named if Fraction(c[0]) < k]
    above = [c for c in named if Fraction(c[0]) > k]
    rng.shuffle(below)
    rng.shuffle(above)
    below_near = [c for c in nearer if Fraction(c[0]) < k]
    above_near = [c for c in nearer if Fraction(c[0]) > k]
    low_b = max(0, count - len(above) - len(above_near))
    high_b = min(count, len(below) + len(below_near))
    if low_b > high_b:
        return named + nearer
    b = rng.randint(low_b, high_b)

    def take(named_side, near_side, m):
        """`m` values from one side: named slips, but now and then a plain near miss, so that near misses turn up at
        every rank (a question that has them must not always have the key at an end)."""
        named_side, near_side, out = list(named_side), list(near_side), []
        for _ in range(m):
            use_near = near_side and (not named_side or rng.random() < 0.25)
            out.append(near_side.pop(0) if use_near else named_side.pop(0))
        return out

    chosen = take(below, below_near, b) + take(above, above_near, count - b)
    return chosen + [c for c in named + nearer if c not in chosen]
