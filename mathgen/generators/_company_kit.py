"""Keep the key from being the only option of its kind.

A learner who sees one power of two, one odd number, one number with a single digit, one standard angle or the only
number of one sign among the four options picks it without working. `keep_company` takes the options a builder has
already ordered (the first `count` are shown) and, only when the key holds such a feature value alone, swaps shown
wrong options for others from a pool of plain near misses so that the key has company in every feature it can.
Swaps are the fewest that work and keep the number of shown options below the key, so the key's rank is not disturbed.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction


STD_DEGREES = (30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330, 360)
_MAX_POOL = 40
WEIGHT = {"sign": 10}  # a key alone on its side of zero is the one tell worth moving its rank for
SHIFT_COST = 4.0  # a swap that moves the key's rank costs nearly as much as a feature the key holds alone


def _tuple(v) -> tuple:
    return tuple(v) if isinstance(v, (tuple, list)) else (v,)


def _text(v) -> str:
    return ", ".join(str(Fraction(x)) if Fraction(x).denominator != 1 else str(int(x)) for x in _tuple(v))


def _sign(x) -> int:
    return (x > 0) - (x < 0)


def features(v) -> dict:
    """What a learner can notice about one option: sign, whole or not, parity, power of two, standard angle, digits."""
    items = [Fraction(x) for x in _tuple(v)]
    f = {
        "ndigits": min(sum(c.isdigit() for c in _text(v)), 8),
        "sign": tuple(_sign(x) for x in items),
        "whole": tuple(x.denominator == 1 for x in items),
    }
    if all(x.denominator == 1 for x in items):
        ints = [int(x) for x in items]
        f["parity"] = tuple(x % 2 for x in ints)
        f["pow2"] = tuple(x > 0 and x & (x - 1) == 0 for x in ints)
        f["ten"] = tuple(x != 0 and x % 10 == 0 for x in ints)
        f["std"] = tuple(abs(x) in STD_DEGREES for x in ints)
    return f


def lone_features(key, shown) -> set:
    """Feature names whose value on the key no shown option shares (a feature counts when every option has it)."""
    fk = features(key)
    others = [features(v) for v in shown]
    names = set(fk).intersection(*(set(o) for o in others)) if others else set(fk)
    return {n for n in names if all(o[n] != fk[n] for o in others)}


def keep_company(key, ordered, pool=(), count: int = 3, shift_cost: float = SHIFT_COST):
    """`ordered` ([(value, tag)]; the first `count` are shown) changed so the key has company in as many features as
    possible. Candidates are the rest of `ordered` and `pool`; a swap keeps the number below the key where it can."""
    shown = list(ordered[:count])
    plain = lambda v: all(isinstance(x, (int, Fraction)) for x in _tuple(v))  # noqa: E731
    if not plain(key) or not all(plain(v) for v, _ in ordered):
        return ordered  # not whole numbers or fractions (a word, a decimal, an expression): nothing to compare
    if len(shown) < count or not lone_features(key, [v for v, _ in shown]):
        return ordered
    k = _tuple(key)
    seen = {k} | {_tuple(v) for v, _ in shown}
    cands = list(shown)
    for v, t in list(ordered[count:]) + list(pool):
        if _tuple(v) not in seen:
            seen.add(_tuple(v))
            cands.append((v, t))
    # its own generator, seeded from the options: the question's random stream is left as it was
    rng = random.Random(repr((str(key), [str(v) for v, _ in ordered])))
    cands = cands[:_MAX_POOL]

    def below(v):
        return _tuple(v) < k

    base_below = sum(1 for v, _ in shown if below(v))
    best, best_score = [], None
    for n_new in range(count + 1):  # the fewest swaps that leave the key with company
        for old, new in itertools.product(
            itertools.combinations(range(count), count - n_new), itertools.combinations(range(count, len(cands)), n_new)
        ):
            pick = [cands[i] for i in (*old, *new)]
            score = (
                sum(WEIGHT.get(n, 1) for n in lone_features(key, [v for v, _ in pick]))
                + shift_cost * abs(sum(1 for v, _ in pick if below(v)) - base_below),
            )
            if best_score is None or score < best_score:
                best, best_score = [pick], score
            elif score == best_score:
                best.append(pick)
        if best_score is not None and best_score[0] < 1:
            break
    pick = rng.choice(best) if rng else best[0]
    chosen = list(pick)
    if rng:
        rng.shuffle(chosen)
    return chosen + [c for c in ordered if not any(c is x or c == x for x in chosen)]


def numeric_pool(key, step=1, lo=None, hi=None, ok=lambda v: True):
    """Plain near misses for a number: nearby values, doubles and halves (powers of two keep company), standard angles."""
    k = Fraction(key)
    out = {}
    for d in range(1, 9):
        for s in (-1, 1):
            out[k + s * d * step] = None
    if k.denominator == 1 and k % 10 == 0:  # a round key keeps a round option for company
        for d in (-20, -10, 10, 20):
            out[k + d] = None
    for e in (-3, -2, -1, 1, 2, 3):
        out[k * Fraction(2) ** e] = None
    if k.denominator == 1:
        for g in STD_DEGREES:
            out[Fraction(g) * (1 if k >= 0 else -1)] = None
    keep = []
    for v in out:
        if (
            v == k
            or (k.denominator == 1 and step == 1 and v.denominator != 1)
            or (lo is not None and v < lo)
            or (hi is not None and v > hi)
            or not ok(v)
        ):
            continue
        keep.append((int(v) if isinstance(key, int) and v.denominator == 1 else v, "near-miss-value"))
    keep.sort(key=lambda c: abs(Fraction(c[0]) - k))
    return keep


def nudge_rank(key, ordered, pool=(), count: int = 3, *, p: float = 0.2, from_below: int = 1):
    """Companion swaps move a small key down a rank, which crowds the middle ranks. When exactly `from_below` shown wrong
    options lie below the key, with probability `p` one shown option above it gives its place to a plain near miss below
    it (one that leaves the key no more alone in any feature than before), so the key's rank is spread back out."""
    shown = list(ordered[:count])
    plain = lambda v: all(isinstance(x, (int, Fraction)) for x in _tuple(v))  # noqa: E731
    if len(shown) < count or not plain(key) or not all(plain(v) for v, _ in shown):
        return ordered
    rng = random.Random(repr(("nudge", str(key), [str(v) for v, _ in ordered])))
    k = _tuple(key)
    if sum(1 for v, _ in shown if _tuple(v) < k) != from_below or rng.random() >= p:
        return ordered
    taken = {k} | {_tuple(v) for v, _ in shown}
    below = [c for c in list(ordered[count:]) + list(pool) if _tuple(c[0]) < k and _tuple(c[0]) not in taken]
    above = [i for i, (v, _) in enumerate(shown) if _tuple(v) > k]
    if not below or not above:
        return ordered
    before = len(lone_features(key, [v for v, _ in shown]))
    rng.shuffle(below)
    rng.shuffle(above)
    for cand in below:
        for i in above:
            new = shown[:i] + [cand] + shown[i + 1 :]
            if len(lone_features(key, [v for v, _ in new])) <= before:
                return new + [c for c in ordered if not any(c is x or c == x for x in new)]
    return ordered
