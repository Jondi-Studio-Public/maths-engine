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
_MAX_POOL = 30
WRONG_LONE = 0.4  # a wrong option that is the odd one out is a tell too: the key should not be the only one that is not
WEIGHT = {"sign": 10, "pow2": 2}  # a key alone on its side of zero is the one tell worth moving its rank for
SHIFT_COST = 4.0  # a swap that moves the key's rank costs nearly as much as a feature the key holds alone


def _tuple(v) -> tuple:
    return tuple(v) if isinstance(v, (tuple, list)) else (v,)


# decimal places the options are printed to, while a builder that prints decimals is being served
_PLACES: list = [None]
# whether the distance between options counts (not for labels such as quadrant numbers)
_NEAR: list = [True]
# what a caller asks beyond that: view (value -> what the learner reads), custom (name -> function of a value), balance
_VIEW: list = [None]
_CUSTOM: list = [None]
_BALANCE: list = [True]
NEAR_FEATURES = ("near1", "near2", "mult5")  # always measured; kept so a caller can name them


def _items(v) -> list:
    """The numbers of an option as the learner reads them (through the caller's `view`, if any)."""
    view = _VIEW[0] or (lambda x: x)
    return [view(Fraction(x)) for x in _tuple(v)]


def _text(v) -> str:
    places = _PLACES[0]
    if places is not None:
        return ", ".join(f"{float(x):.{places}f}" for x in _items(v))
    return ", ".join(str(x) if x.denominator != 1 else str(int(x)) for x in _items(v))


def _sign(x) -> int:
    return (x > 0) - (x < 0)


def features(v) -> dict:
    """What a learner can notice about one option: sign, whole or not, parity, power of two, standard angle, digits."""
    items = _items(v)
    text = _text(v)
    f = {
        "ndigits": min(sum(c.isdigit() for c in text), 8),
        "sign": tuple(_sign(x) for x in items),
        "whole": tuple(x.denominator == 1 for x in items),
    }
    if _PLACES[0] is not None and len(items) == 1:
        decimals = text.partition(".")[2]
        f["decimal_style"] = "none" if not decimals else "zeros" if not decimals.strip("0") else "nonzero"
    if all(x.denominator == 1 for x in items):
        ints = [int(x) for x in items]
        f["parity"] = tuple(x % 2 for x in ints)
        f["pow2"] = tuple(x > 0 and x & (x - 1) == 0 for x in ints)
        f["mult5"] = tuple(x % 5 == 0 for x in ints)
        f["ten"] = tuple(x != 0 and x % 10 == 0 for x in ints)
        f["std"] = tuple(abs(x) in STD_DEGREES for x in ints)
    return f


_CACHE: dict = {}


def _cached(v):
    """(features, float value or None) of one option, remembered."""
    key = (_PLACES[0], *_tuple(v))
    if _VIEW[0] is not None:  # a view is a function: its results are not remembered
        return features(v), float(_items(v)[0]) if len(key) == 2 else None
    hit = _CACHE.get(key)
    if hit is None:
        if len(_CACHE) > 5000:
            _CACHE.clear()
        hit = _CACHE[key] = (features(v), float(Fraction(key[1])) if len(key) == 2 else None)
    return hit


def lone_sets(options) -> list:
    """For each option (the key first), the names of the features on which it is the only option with its value. Besides
    the features of one option, `near1` and `near2` say how many other options lie within 1 and within 2 of it: a key
    in a cluster of near misses, or clear of them, is picked out by that count."""
    hits = [_cached(v) for v in options]
    feats = [h[0] for h in hits]
    names = set(feats[0])
    for f in feats[1:]:
        names &= f.keys()
    columns = {n: [f[n] for f in feats] for n in names}
    values = [h[1] for h in hits]
    if None not in values and _NEAR[0]:
        columns["near1"] = [min(sum(abs(v - w) <= 1 + 1e-9 for w in values) - 1, 3) for v in values]
        columns["near2"] = [min(sum(abs(v - w) <= 2 + 1e-9 for w in values) - 1, 3) for v in values]
    for name, fn in (_CUSTOM[0] or {}).items():  # features the caller defines for every option
        columns[name] = [fn(v) for v in options]
    out = [set() for _ in options]
    for n, col in columns.items():
        for i, v in enumerate(col):
            if col.count(v) == 1:
                out[i].add(n)
    return out


def lone_features(key, shown) -> set:
    """Feature names on which no shown option shares the key's value (`lone_sets` for the key alone)."""
    return lone_sets([key, *shown])[0]


def keep_company(
    key,
    ordered,
    pool=(),
    count: int = 3,
    shift_cost: float = SHIFT_COST,
    places=None,
    near: bool = True,
    *,
    extra=(),
    view=None,
    custom=None,
    ok=None,
    must=(),
    balance: bool = True,
    max_pool: int = _MAX_POOL,
):
    """`_keep_company` for options printed to `places` decimal places (None: as whole numbers or fractions).

    `view` turns a value into what the learner reads of it (a rounded decimal) before anything is measured; `custom`
    maps a name to a function of a value, a feature of every option; `ok(value)` rules a candidate out (the shown options
    are never ruled out); `balance` (default) also counts the wrong options that are alone in a feature, False only the
    key; `max_pool` caps the candidates; when the key still has no company in a feature named in `must`,
    `TooFewDistractors` is raised so the caller's draw is made again. `extra` names features (``NEAR_FEATURES``) a caller
    protects: they are always measured, so it changes nothing."""
    _PLACES[0], _NEAR[0], _VIEW[0], _CUSTOM[0], _BALANCE[0] = places, near, view, custom, balance
    try:
        clean = [c for c in ordered if c[0] is not None]  # a slip that did not come out (None) is not an option
        if len(clean) == len(ordered):
            return _keep_company(key, ordered, pool, count, shift_cost, ok, must, max_pool)
        return _keep_company(key, clean, pool, count, shift_cost, ok, must, max_pool) + [
            c for c in ordered if c[0] is None
        ]
    finally:
        _PLACES[0], _NEAR[0], _VIEW[0], _CUSTOM[0], _BALANCE[0] = None, True, None, None, True


def _keep_company(key, ordered, pool, count, shift_cost, ok=None, must=(), max_pool=_MAX_POOL):
    """`ordered` ([(value, tag)]; the first `count` are shown) changed so the key has company in as many features as
    possible. Candidates are the rest of `ordered` and `pool`; a swap keeps the number below the key where it can."""
    shown = list(ordered[:count])
    plain = lambda v: all(isinstance(x, (int, Fraction)) for x in _tuple(v))  # noqa: E731
    if not plain(key) or not all(plain(v) for v, _ in ordered):
        return ordered  # not whole numbers or fractions (a word, a decimal, an expression): nothing to compare
    if len(shown) < count:
        return ordered
    sets0 = lone_sets([key, *(v for v, _ in shown)])
    if not sets0[0] and not (_BALANCE[0] and any(sets0[1:])):
        return ordered
    k = _tuple(key)
    seen = {k} | {_tuple(v) for v, _ in shown}
    cands = list(shown)
    for v, t in list(ordered[count:]) + list(pool):
        if _tuple(v) not in seen and (ok is None or ok(v)):
            seen.add(_tuple(v))
            cands.append((v, t))
    # its own generator, seeded from the options: the question's random stream is left as it was
    rng = random.Random(repr((str(key), [str(v) for v, _ in ordered])))
    cands = cands[:max_pool]

    def below(v):
        return _tuple(v) < k

    base_below = sum(1 for v, _ in shown if below(v))

    def score(pick):
        sets = lone_sets([key, *(v for v, _ in pick)])
        return (
            sum(WEIGHT.get(n, 1) for n in sets[0])
            + (WRONG_LONE if _BALANCE[0] else 0) * sum(len(x) for x in sets[1:])
            + shift_cost * abs(sum(1 for v, _ in pick if below(v)) - base_below)
        )

    # a swap at a time, the best each round, so the search stays cheap: stop when no swap helps
    current, current_score = list(shown), score(shown)
    for _ in range(count + 1):
        options = []
        for i in range(count):
            for c in cands:
                if any(c is x or c == x for x in current):
                    continue
                trial = current[:i] + [c] + current[i + 1 :]
                options.append((score(trial), i, trial))
        if not options:
            break
        top = min(o[0] for o in options)
        if top >= current_score - 1e-9:
            break
        current_score = top
        current = rng.choice([o[2] for o in options if o[0] == top])
    if lone_sets([key, *(v for v, _ in current)])[0]:  # the key is still alone in something: try two swaps at once
        best_pair, best_pair_score = None, current_score
        extras = [c for c in cands if not any(c is x or c == x for x in current)]
        for i, j in itertools.combinations(range(count), 2):
            for a, b in itertools.combinations(extras, 2):
                trial = list(current)
                trial[i], trial[j] = a, b
                sc = score(trial)
                if sc < best_pair_score - 1e-9:
                    best_pair, best_pair_score = trial, sc
        if best_pair:
            current = best_pair
    chosen = list(current)
    if must and set(must) & lone_sets([key, *(v for v, _ in chosen)])[0]:
        from ._w6a_kit import TooFewDistractors

        raise TooFewDistractors("no wrong option keeps the key company in " + ", ".join(sorted(must)))
    rng.shuffle(chosen)
    return chosen + [c for c in ordered if not any(c is x or c == x for x in chosen)]


def numeric_pool(key, step=1, lo=None, hi=None, ok=lambda v: True):
    """Plain near misses for a number: nearby values, doubles and halves (powers of two keep company), standard angles."""
    k = Fraction(key)
    out = {}
    for d in range(1, 9):
        for s in (-1, 1):
            out[k + s * d * step] = None
    if k.denominator == 1 and step != 1:  # whole numbers beside a whole key printed to decimal places
        for d in (-3, -2, -1, 1, 2, 3):
            out[k + d] = None
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
