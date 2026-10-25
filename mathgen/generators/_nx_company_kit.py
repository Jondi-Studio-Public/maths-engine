"""Give the key company: choose among spare candidates the set of options in which the key is never alone.

A learner with no maths can still read a few things off the options: digit count, a minus sign, a square root, a leading
coefficient, a multiple of 5, how many other options sit within 1 or 2 of it. If the key is the only option with some
value of such a feature, or the wrong options are lone in many of them (so the learner can eliminate them), the key is
easier to find than chance. `company` swaps wrong options for spare candidates, one at a time, while that count falls.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from collections import Counter

_STD_DEG = {30, 45, 60, 90, 120, 135, 150, 180}
SHIFT_COST = 9  # moving the key's rank costs nearly as much as a feature the key holds alone
KEY_WEIGHT = 10  # being the lone holder of a feature value costs the key this much, a wrong option 1


def _pi_kind(t, v):
    if "pi" not in t or v is None:
        return "no-pi"
    r = Fraction(v / math.pi).limit_denominator(1000)
    if abs(float(r) - v / math.pi) > 1e-9:
        return "pi-mixed"
    return "pi-std" if r.denominator in (1, 2, 3, 4, 6) else "pi-other"


def _features(texts: list, values: list) -> list[dict]:
    plain = all(v is not None for v in values)
    feats = []
    for i, (t, v) in enumerate(zip(texts, values)):
        t = t.strip()
        f = {
            "ndigits": min(sum(c.isdigit() for c in t), 8),
            "lead": "numeric"
            if re.fullmatch(r"-?[\d.,]+(/[\d.]+)?", t)
            else "coef"
            if re.match(r"-?\(?\d", t)
            else "bare",
            "slash": "/" in t,
            "sqrt": "sqrt" in t,
            "minus": "-" in t,
            "pi": _pi_kind(t, v),
        }
        if v is not None and abs(v - round(v)) < 1e-9:
            n = round(v)
            f["mult5"] = n % 5 == 0
            f["parity"] = n % 2
            f["std"] = abs(n) in _STD_DEG
            f["pow2"] = n > 0 and n & (n - 1) == 0
        if plain:
            f["near1"] = min(sum(abs(w - v) <= 1 for j, w in enumerate(values) if j != i), 3)
            f["near2"] = min(sum(abs(w - v) <= 2 for j, w in enumerate(values) if j != i), 3)
        feats.append(f)
    if plain:
        spread = [round(sum(abs(v - w) for w in values), 9) for v in values]
        lo, hi = min(spread), max(spread)
        for f, s in zip(feats, spread):
            f["centrality"] = (
                "central"
                if s == lo and spread.count(lo) == 1
                else "outlier"
                if s == hi and spread.count(hi) == 1
                else "mid"
            )
    common = set.intersection(*(set(f) for f in feats))
    return [{k: f[k] for k in common} for f in feats]


def lone_cost(texts: list, values: list) -> int:
    """KEY_WEIGHT per feature value only the key (first) holds, plus 1 per feature value only a wrong option holds."""
    feats = _features(texts, values)
    cost = 0
    for name in feats[0]:
        column = [f[name] for f in feats]
        counts = Counter(column)
        if len(counts) == 1:
            continue
        for i, v in enumerate(column):
            if counts[v] == 1:
                cost += (2 if name == "centrality" else KEY_WEIGHT) if i == 0 else 1
    return cost


def company(key_text, key_value, options, pool, rng, value_of, passes: int = 3):
    """`options` (objects with `.text`) with wrong options swapped for ones in `pool` while the lone cost falls."""
    options, pool = list(options), list(pool)

    def cost(opts):
        texts = [key_text, *(o.text for o in opts)]
        values = [value_of(o.text) for o in opts]
        moved = abs(sum(v < key_value for v in values) - start_below)  # the key's drawn rank is kept where it can be
        return lone_cost(texts, [key_value, *values]) + SHIFT_COST * moved

    start_below = sum(value_of(o.text) < key_value for o in options)
    best = cost(options)
    for _ in range(passes):
        improved = False
        for i in rng.sample(range(len(options)), len(options)):
            for j in rng.sample(range(len(pool)), len(pool)):
                trial = options[:i] + [pool[j]] + options[i + 1 :]
                c = cost(trial)
                if c < best:
                    best, improved = c, True
                    options[i], pool[j] = pool[j], options[i]
        if not improved or best == 0:
            break
    return options
