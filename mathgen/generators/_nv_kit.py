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


def lone_property_fix(key, ordered, props, count: int = 3, view=lambda v: v):
    """`ordered` ([(value, tag)], the first `count` are the options shown) changed so the key is never the ONLY option
    with a property: for each `(pred, pool)` in `props`, when `pred(key)` holds and none of the shown wrong values has it,
    a wrong value that has it (from the rest of `ordered`, else from `pool`, a list of `(value, tag)`) takes the place of
    one shown, never of the last shown value that gives another property its second holder.

    `view` turns a value into what the learner sees (rounded to the stated places) before a property is tested.

    Raises `TooFewDistractors` when no such value exists, so the caller's draw is made again."""
    from ._w6a_kit import TooFewDistractors

    k = Fraction(key)
    pv = lambda v: view(Fraction(v))  # noqa: E731
    shown, rest = list(ordered[:count]), list(ordered[count:])
    for pred, pool in props:
        if not pred(view(k)) or any(pred(pv(v)) for v, _ in shown):
            continue
        have = {pv(v) for v, _ in shown} | {view(k)}
        options = [c for c in rest + list(pool) if pred(pv(c[0])) and pv(c[0]) not in have]
        if not options:
            raise TooFewDistractors("no wrong value shares the key's property")
        pick = _RNG.get().choice(options) if _RNG.get() else options[0]  # any of them, so a side is not favoured
        sole = [
            i
            for i, (v, _) in enumerate(shown)
            if any(
                pred2(k) and sum(pred2(Fraction(w)) for w, _ in shown) == 1 and pred2(Fraction(v)) for pred2, _ in props
            )
        ]
        free = [i for i in range(len(shown)) if i not in sole] or list(range(len(shown)))
        victim = _RNG.get().choice(free) if _RNG.get() else free[-1]
        shown[victim] = pick
    return shown + [c for c in ordered[count:] if c not in shown]


def is_power_of_two(v: Fraction) -> bool:
    return v.denominator == 1 and v >= 1 and (int(v) & (int(v) - 1)) == 0


def is_multiple_of_ten(v: Fraction) -> bool:
    return v.denominator == 1 and v != 0 and int(v) % 10 == 0
