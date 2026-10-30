"""Choose the shown wrong options so that no no-maths strategy finds the key more often than chance.

`balance` takes a pool of numeric wrong options (more than are shown) and picks the subset to show. The key's rank among
the options is drawn from the generator's own `rng`; within that rank the subset is the one in which the key is the lone
holder of the fewest visible features, and is neither the most central, the most outlying, nor the one nearest the median.
Nothing here registers a topic.
"""

from __future__ import annotations

import functools
import itertools
import statistics
from fractions import Fraction

from ._nx_company_kit import lone_cost

_STRATEGY_COST = 7  # a strategy that picks the key outright costs about as much as a feature the key holds alone
_RANK_COST = 3


def _num(text: str) -> Fraction:
    return Fraction(text.strip().replace(",", ""))


def _tells(key: Fraction, values: list) -> int:
    allv = [key, *values]
    spreads = [sum(abs(v - w) for w in allv) for v in allv]
    tells = 0
    if spreads[0] == min(spreads) and spreads.count(spreads[0]) == 1:
        tells += 1
    if spreads[0] == max(spreads) and spreads.count(spreads[0]) == 1:
        tells += 1
    med = statistics.median(allv)
    dist = [abs(v - med) for v in allv]
    if dist[0] == min(dist):
        tells += 1
    return tells


def balance(answer_text: str, pool: list, n: int, rng, max_pool: int = 9) -> list:
    """`n` of the `pool` (objects with `.text`) to show beside the key `answer_text`."""
    if len(pool) <= n:
        return list(pool)
    pool = list(pool)[:max_pool]
    key = _num(answer_text)
    vals = [_num(d.text) for d in pool]
    target = rng.randrange(n + 1)
    best, best_cost = [], None
    for combo in itertools.combinations(range(len(pool)), n):
        chosen = [vals[i] for i in combo]
        below = sum(v < key for v in chosen)
        cost = (
            lone_cost([answer_text, *(pool[i].text for i in combo)], [key, *chosen])
            + _STRATEGY_COST * _tells(key, chosen)
            + _RANK_COST * abs(below - target)
        )
        if best_cost is None or cost < best_cost:
            best, best_cost = [combo], cost
        elif cost == best_cost:
            best.append(combo)
    pick = best[rng.randrange(len(best))]
    return [pool[i] for i in pick]


def balance_text(answer_text: str, pool: list, n: int, rng, max_pool: int = 9) -> list:
    """`n` of the `pool` (objects with `.text`) to show beside an expression key, ordered by text features alone: the
    subset in which the key and the wrong options are lone holders of the fewest visible features."""
    pool = list(pool)[:max_pool]
    if len(pool) <= n:
        return pool
    best, best_cost = [], None
    for combo in itertools.combinations(range(len(pool)), n):
        cost = lone_cost([answer_text, *(pool[i].text for i in combo)], [None] * (n + 1))
        if best_cost is None or cost < best_cost:
            best, best_cost = [combo], cost
        elif cost == best_cost:
            best.append(combo)
    return [pool[i] for i in best[rng.randrange(len(best))]]


# -- a learner-eye cost that mirrors what a person can read off the options -----------------------------------------------

import math  # noqa: E402
import re  # noqa: E402
from collections import Counter  # noqa: E402

_SIGN = lambda x: "neg" if x < -1e-12 else "zero" if abs(x) <= 1e-12 else "pos"  # noqa: E731


@functools.lru_cache(maxsize=None)
def _complex_value(text: str):
    t = text.strip().replace("−", "-")
    if "pi" in t or "π" in t or "," in t or not re.fullmatch(r"[0-9i()*/+\-. ]+", t):
        try:
            return None if not re.fullmatch(r"-?\d+\.?\d*", t) else complex(float(t))
        except ValueError:
            return None
    t = re.sub(r"(?<=[\d)])i\b", "*I", t)
    t = re.sub(r"(?<![\w*])i\b", "I", t)
    try:
        import sympy

        return complex(sympy.sympify(t, locals={"I": sympy.I}))
    except Exception:  # noqa: BLE001 - text that does not parse is simply not a number
        return None


@functools.lru_cache(maxsize=None)
def _base(text: str) -> tuple[dict, complex | None]:
    t = text.strip()
    f = {
        "ndigits": min(len(re.findall(r"\d", t)), 8),
        "lead": "numeric" if re.fullmatch(r"-?[\d.,]+(/[\d.]+)?", t) else "coef" if re.match(r"-?\(?\d", t) else "bare",
        "dec": "none"
        if not re.fullmatch(r"-?\d+\.\d+", t)
        else "zeros"
        if not t.split(".")[1].strip("0")
        else "nonzero",
        "slash": "/" in t,
        "sqrt": "sqrt" in t or "√" in t,
        "minus": "-" in t or "−" in t,
        "mult10": bool(re.fullmatch(r"-?\d+", t)) and int(t) != 0 and int(t) % 10 == 0,
        "pi": ("pi-" if ("pi" in t or "π" in t) else "no-pi"),
    }
    z = _complex_value(t)
    if z is not None:
        f["kind"] = "real" if abs(z.imag) < 1e-9 else "imag" if abs(z.real) < 1e-9 else "mixed"
        f["re_sign"], f["im_sign"] = _SIGN(z.real), _SIGN(z.imag)
        if abs(z.imag) < 1e-9:
            x = z.real
            f["whole"] = abs(x - round(x)) < 1e-9
            if f["whole"]:
                n = round(x)
                f["parity"], f["mult5"], f["pow2"] = n % 2, n % 5 == 0, n > 0 and n & (n - 1) == 0
    return f, z


def _gains(texts: list) -> tuple[float, float, int]:
    """(the strategies' best gain over chance, the number of features the key alone holds) for options with the key first."""
    base = [_base(t) for t in texts]
    feats = [dict(b[0]) for b in base]
    n = len(texts)
    zs = [b[1] for b in base]
    plain = all(z is not None and abs(z.imag) < 1e-9 for z in zs)
    gains = []
    if plain:
        vals = [z.real for z in zs]
        for i, f in enumerate(feats):
            f["near1"] = min(sum(abs(v - vals[i]) <= 1 for j, v in enumerate(vals) if j != i), 3)
            f["near2"] = min(sum(abs(v - vals[i]) <= 2 for j, v in enumerate(vals) if j != i), 3)
        spread = [round(sum(abs(v - w) for w in vals), 9) for v in vals]
        lo, hi = min(spread), max(spread)
        for f, s in zip(feats, spread):
            f["cent"] = (
                "central"
                if s == lo and spread.count(lo) == 1
                else "outlier"
                if s == hi and spread.count(hi) == 1
                else "mid"
            )
        for tag in ("central", "outlier"):
            pick = [i for i, f in enumerate(feats) if f["cent"] == tag]
            gains.append(((1 / len(pick) if 0 in pick else 0.0) if pick else 1 / n) - 1 / n)
        ordered = sorted(vals)
        mid = n // 2
        median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        dist = [round(abs(v - median), 9) for v in vals]
        pick = [i for i, d in enumerate(dist) if d == min(dist)]
        gains.append((1 / len(pick) if 0 in pick else 0.0) - 1 / n)
    common = set.intersection(*(set(f) for f in feats))
    lone = [False] * n
    key_alone = 0
    for name in common:
        col = [f[name] for f in feats]
        counts = Counter(col)
        if len(counts) == 1:
            continue
        for i, v in enumerate(col):
            if counts[v] == 1:
                lone[i] = True
                key_alone += i == 0
    alive = [i for i in range(n) if not lone[i]] or list(range(n))
    gains.append((1 / len(alive) if 0 in alive else 0.0) - 1 / n)
    return max(gains), sum(lone[1:]), key_alone


def fair(answer_text: str, pool: list, n: int, rng, max_pool: int = 10) -> list:
    """`n` of `pool` (objects with `.text`) to show beside `answer_text`, chosen by what a learner can read off the options:
    least gain for eliminating odd ones out, picking the most central, the most outlying or the nearest the median."""
    pool = list(pool)[:max_pool]
    if len(pool) <= n:
        return pool
    best, best_cost = [], None
    for combo in itertools.combinations(range(len(pool)), n):
        gain, wrong_lone, key_alone = _gains([answer_text, *(pool[i].text for i in combo)])
        cost = round(10 * max(gain, 0) + 0.3 * wrong_lone + 0.5 * key_alone, 6)
        if best_cost is None or cost < best_cost:
            best, best_cost = [combo], cost
        elif cost == best_cost:
            best.append(combo)
    return [pool[i] for i in best[rng.randrange(len(best))]]
