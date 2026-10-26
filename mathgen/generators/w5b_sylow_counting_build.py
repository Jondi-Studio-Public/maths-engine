"""Generates `w5b_sylow_counting` questions (easy, medium, hard) and the exam chain `exd_x_w5b_sylow_counting`.

``uc2-sylow-counting`` (``abstract-algebra``, answer ``number``, some tiers ``set``): Sylow's theorems for a finite
group of stated order n, with the arithmetic done from the divisors and congruences of n.

* easy: the order of a Sylow p-subgroup (the largest power of p dividing n), the sum of the possible values of n_p
  (n_p = 1 mod p and n_p divides n / p^k), or the sum of the Sylow orders, one for each prime dividing n.
* medium: how many values n_p may take, the largest of them, the number of elements of order p when n_p is given
  (Sylow p-subgroups of prime order meet only in the identity, so n_p (p - 1)), and for how many primes the Sylow
  subgroup is forced to be normal by counting alone.
* hard: the elements left over once two prime-order Sylow families are counted (|G| = p q r), the surplus that makes
  a pair of Sylow counts impossible, the least number of fixed points of a p-group acting on N points (N mod p), and
  the elements of G = P x_ Q-type orders that are not of order q.
* exam: typed three-part chains: (a) the set of possible n_q, (b) the number of elements of order q for the largest
  n_q, (c) the elements left over (follows through from (b)).

Every n comes from a curated pool of orders whose Sylow subgroups for the primes asked about have prime order when an
element count is asked, so the counts are exact; each answer is recomputed by a second route (divisors of n tested
for the congruence, a permutation of the right cycle type for the fixed points).
"""

from __future__ import annotations

import itertools
import math
import re
from functools import lru_cache

from ..answers import exact_value
from ..core import attempts, check, generator, phrase
from ._r4a_alg_kit import int_options, number_part, set_part, with_latex

PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)


# -- number theory ---------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def factor(n):
    """{prime: exponent} by trial division."""
    out = {}
    for d in range(2, n + 1):
        if d * d > n:
            break
        for _ in range(n.bit_length()):
            if n % d:
                break
            out[d] = out.get(d, 0) + 1
            n //= d
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def divisors(m):
    return [d for d in range(1, m + 1) if m % d == 0]


def sylow_order(n, p):
    return p ** factor(n).get(p, 0)


def np_values(n, p):
    """Possible n_p: divisors of n / p^k that are 1 mod p, ascending."""
    return [d for d in divisors(n // sylow_order(n, p)) if d % p == 1 % p]


def np_values_by_scan(n, p):
    """The same set by a second route: every d dividing n with d = 1 (mod p) (a divisor of n that is 1 mod p cannot
    contain p, so it divides n / p^k)."""
    return [d for d in range(1, n + 1) if n % d == 0 and d % p == 1 % p]


def checked_np(n, p):
    values = np_values(n, p)
    check(values == np_values_by_scan(n, p), "the two routes to the possible n_p disagree")
    return values


@lru_cache(maxsize=None)
def _pool_orders():
    """Orders 12..720 with at least two prime divisors."""
    return tuple(n for n in range(12, 721) if len(factor(n)) >= 2)


@lru_cache(maxsize=None)
def _pool_squarefree3():
    """Products of three distinct primes up to 1100: (n, (p, q, r))."""
    out = []
    for i, a in enumerate(PRIMES[:9]):
        for j in range(i + 1, 9):
            for k in range(j + 1, 9):
                n = a * PRIMES[j] * PRIMES[k]
                if n <= 1100:
                    out.append((n, (a, PRIMES[j], PRIMES[k])))
    return tuple(sorted(out))


@lru_cache(maxsize=None)
def _pool_pq():
    """Orders p q with p < q primes and p dividing q - 1: (p, q)."""
    return tuple((p, q) for p in PRIMES for q in PRIMES if p < q and (q - 1) % p == 0 and p * q <= 600)


@lru_cache(maxsize=None)
def _pool_pkq():
    """Orders p^k q (k = 2 or 3, q a different prime) whose n_q can be more than 1: (n, p, q)."""
    out = []
    for p in (2, 3, 5, 7, 11):
        for q in PRIMES:
            for k in (2, 3, 4):
                n = p**k * q
                if q != p and n <= 1000 and len(np_values(n, q)) >= 2:
                    out.append((n, p, q))
    return tuple(sorted(out))


@lru_cache(maxsize=None)
def _pool_squarefree34():
    """Products of three or four distinct primes up to 2400: (n, primes)."""
    out = []
    for size in (3, 4):
        for combo in itertools.combinations(PRIMES[:10], size):
            n = math.prod(combo)
            if n <= 2400:
                out.append((n, combo))
    return tuple(sorted(out))


# -- answer plumbing -------------------------------------------------------------------------------------------


def _fillers(value, lo=0):
    """Untagged near misses used to top up an option list."""
    return [(value + 1, None), (value - 1, None), (value + 2, None), (2 * value, None), (value + 10, None)]


def number_q(text, value, working, slips, *, keep=4, lo=0):
    options = int_options(value, [*slips, *_fillers(value)], keep, lo=lo)
    check(len(options) >= 2, "too few distractors")
    return dict(
        text=text,
        answer_kind="number",
        answer=exact_value(value),
        answer_text=str(value),
        answer_latex=str(value),
        working=working,
        distractors=options,
    )


def _set_candidates(n, p):
    """Wrong sets for 'the possible values of n_p', each with the error behind it."""
    k_part = sylow_order(n, p)
    m = n // k_part
    right = np_values(n, p)
    all_div = divisors(m)
    negative = [d for d in all_div if d % p == (p - 1) % p]
    mod_pk = [d for d in all_div if d % k_part == 1 % k_part]
    return [
        (all_div, "sylow-np-skips-congruence"),
        (right[1:], "sylow-np-omits-one"),
        (negative, "sylow-np-wrong-residue"),
        (mod_pk, "sylow-np-congruent-mod-pk"),
        ([d for d in divisors(n) if d % p == 1 % p and d != 1 and d < m] or None, None),
        (sorted({*right, p}), None),
        ([d for d in all_div if d != 1 and d not in right], "sylow-np-skips-congruence"),
    ]


def _nonempty(candidates):
    return [(v, tag) for v, tag in candidates if v]


def _np_text(p):
    return f"n_{p}"


# -- easy ------------------------------------------------------------------------------------------------------

_E1_STEMS = (
    "A finite group G has order {n}. What is the order of a Sylow {p}-subgroup of G?",
    "Let |G| = {n}. Find the order of a Sylow {p}-subgroup P of G.",
    "By Sylow's first theorem a group of order {n} has a subgroup whose order is the largest power of {p} dividing {n}. "
    "What is that order?",
)
_E2_STEMS = (
    "A group G has order {n}, and n_{p} is the number of its Sylow {p}-subgroups. Add up every value that n_{p} could "
    "take by Sylow's theorems. What is the sum?",
    "Let |G| = {n}. Using Sylow's theorems, find every value that the number n_{p} of Sylow {p}-subgroups of G could "
    "have, and give the sum of all of them.",
)
_E3_STEMS = (
    "A group G has order {n}. For each prime dividing {n} take the order of a Sylow subgroup for that prime. What is "
    "the sum of these orders?",
    "G is a finite group of order {n}. Add together the orders of its Sylow subgroups, one for each prime dividing "
    "{n}. What is the total?",
)


def _easy(rng):
    kind = rng.choice(("order", "np_sum", "orders_sum"))
    if kind == "order":
        for _ in attempts(500, "an order with a Sylow subgroup of non-prime order"):
            n = rng.choice(_pool_orders())
            p = rng.choice(sorted(factor(n)))
            if p <= 13 and (factor(n)[p] >= 2 or rng.random() < 0.2):
                break
        k = factor(n)[p]
        key = p**k
        check(n % key == 0 and (n // key) % p != 0, "p^k is not the exact power of p dividing n")
        m = n // key
        return number_q(
            phrase(rng, _E1_STEMS, n=n, p=p),
            key,
            [f"{n} = {p}^{k} × {m} with {p} not dividing {m}, so |P| = {p}^{k} = {key}."],
            [
                (p, "sylow-order-prime-only"),
                (m, "sylow-order-cofactor"),
                (k, "sylow-order-exponent-only"),
                (n // p, "sylow-order-n-over-p"),
            ],
            lo=1,
        )
    if kind == "np_sum":
        for _ in attempts(500, "an order with at least two possible n_p"):
            n = rng.choice(_pool_orders())
            p = rng.choice(sorted(factor(n)))
            values = checked_np(n, p)
            if p <= 13 and 2 <= len(values) <= 6:
                break
        m = n // sylow_order(n, p)
        return number_q(
            phrase(rng, _E2_STEMS, n=n, p=p),
            sum(values),
            [
                f"n_{p} divides {m} and n_{p} = 1 mod {p}.",
                f"Divisors of {m}: {', '.join(map(str, divisors(m)))}; those that are 1 mod {p}: {', '.join(map(str, values))}; "
                f"their sum is {sum(values)}.",
            ],
            [
                (sum(divisors(m)), "sylow-np-skips-congruence"),
                (sum(values) - 1, "sylow-np-omits-one"),
                (sum(d for d in divisors(m) if d % p == (p - 1) % p), "sylow-np-wrong-residue"),
                (
                    sum(d for d in divisors(m) if d % sylow_order(n, p) == 1 % sylow_order(n, p)),
                    "sylow-np-congruent-mod-pk",
                ),
            ],
            lo=1,
        )
    for _ in attempts(500, "an order with several Sylow orders"):
        n = rng.choice(_pool_orders())
        if len(factor(n)) >= 2 and any(e >= 2 for e in factor(n).values()):
            break
    primes = sorted(factor(n))
    orders = [sylow_order(n, p) for p in primes]
    check(math.prod(orders) == n, "the Sylow orders do not multiply to n")
    return number_q(
        phrase(rng, _E3_STEMS, n=n),
        sum(orders),
        [
            f"{n} = "
            + " × ".join(f"{p}^{factor(n)[p]}" for p in primes)
            + f", so the orders are {', '.join(map(str, orders))}.",
            f"Their sum is {sum(orders)}.",
        ],
        [
            (sum(primes), "sylow-order-prime-only"),
            (sum(factor(n)[p] for p in primes), "sylow-order-exponent-only"),
            (sum(n // o for o in orders), "sylow-order-cofactor"),
            (sum(o * p for o, p in zip(orders, primes)), None),
        ],
        lo=1,
    )


# -- medium ----------------------------------------------------------------------------------------------------

_M1_STEMS = (
    "A group G has order {n}. By Sylow's theorems, how many different values can n_{p}, the number of Sylow "
    "{p}-subgroups of G, take?",
    "Let |G| = {n}. Counting every value that Sylow's theorems allow for the number n_{p} of Sylow {p}-subgroups, how "
    "many possibilities are there?",
)
_M2_STEMS = (
    "G is a group of order {n}. What is the largest value that n_{p}, the number of Sylow {p}-subgroups, could have?",
    "Sylow's theorems restrict the number n_{p} of Sylow {p}-subgroups of a group of order {n}. Find the greatest "
    "value they allow.",
)
_M3_STEMS = (
    "A group G of order {n} has exactly {x} Sylow {p}-subgroups, each of prime order {p}. How many elements of G "
    "have order {p}?",
    "Suppose |G| = {n} and G has n_{p} = {x}. Each Sylow {p}-subgroup has order {p}. Find the number of elements of "
    "G of order {p}.",
)
_M4_STEMS = (
    "Take a group of order {n}. For how many of the primes dividing {n} does Sylow's theorem force the number of "
    "Sylow subgroups to be 1, so that the Sylow subgroup is normal?",
    "G has order {n}. Count the primes p dividing {n} for which the only value of n_p allowed by Sylow's theorems "
    "is 1.",
)


def _medium(rng):
    kind = rng.choice(("count", "largest", "elements", "forced"))
    if kind == "count":
        for _ in attempts(500, "an order with several n_p values"):
            n = rng.choice(_pool_orders())
            p = rng.choice(sorted(factor(n)))
            values = checked_np(n, p)
            if p <= 13 and 3 <= len(values) <= 8:
                break
        m = n // sylow_order(n, p)
        return number_q(
            phrase(rng, _M1_STEMS, n=n, p=p),
            len(values),
            [f"n_{p} divides {m} and is 1 mod {p}: {', '.join(map(str, values))}, which is {len(values)} values."],
            [
                (len(divisors(m)), "sylow-np-skips-congruence"),
                (len(values) - 1, "sylow-np-omits-one"),
                (len(values) + 1, None),
                (len([d for d in divisors(m) if d % p == (p - 1) % p]), "sylow-np-wrong-residue"),
            ],
            lo=1,
        )
    if kind == "largest":
        for _ in attempts(500, "an order whose largest n_p differs from the cofactor"):
            n = rng.choice(_pool_orders())
            p = rng.choice(sorted(factor(n)))
            values = checked_np(n, p)
            m = n // sylow_order(n, p)
            if p <= 13 and len(values) >= 3 and values[-1] != m:
                break
        return number_q(
            phrase(rng, _M2_STEMS, n=n, p=p),
            values[-1],
            [f"The allowed values are {', '.join(map(str, values))}; the largest is {values[-1]}."],
            [
                (m, "sylow-np-largest-divisor"),
                (values[-2], "sylow-np-second-largest"),
                (n // p, None),
                (values[-1] + p, None),
            ],
            lo=1,
        )
    if kind == "elements":
        for _ in attempts(500, "an order with p exactly dividing n and a nontrivial n_p"):
            n = rng.choice(_pool_orders())
            primes = [p for p in factor(n) if 3 <= p <= 31 and factor(n)[p] == 1 and len(np_values(n, p)) >= 2]
            if primes:
                p = rng.choice(primes)
                values = checked_np(n, p)
                x = rng.choice(values[1:])
                break
        key = x * (p - 1)
        # two independent routes: x subgroups of order p, pairwise meeting only in the identity
        check(key == x * p - x, "n_p(p - 1) and n_p p - n_p disagree")
        return number_q(
            phrase(rng, _M3_STEMS, n=n, p=p, x=x),
            key,
            [
                f"Each Sylow {p}-subgroup has {p - 1} elements of order {p}, and two of them share only the identity.",
                f"{x} × {p - 1} = {key}.",
            ],
            [
                (x * p, "sylow-elements-include-identity"),
                (x, "sylow-elements-one-per-subgroup"),
                (key + 1, "sylow-elements-count-identity"),
                (n - key, "sylow-elements-complement"),
            ],
            lo=1,
        )
    for _ in attempts(500, "an order with at least one forced and one free prime"):
        n = rng.choice(_pool_orders())
        forced = sum(1 for p in factor(n) if checked_np(n, p) == [1])
        if len(factor(n)) >= 2 and forced <= len(factor(n)):
            if rng.random() < 0.7 or forced in (0, 1, 2):
                break
    primes = sorted(factor(n))
    return number_q(
        phrase(rng, _M4_STEMS, n=n),
        forced,
        [
            "; ".join(f"n_{p} in {{{', '.join(map(str, np_values(n, p)))}}}" for p in primes)
            + f". Forced to be 1 for {forced} of the {len(primes)} primes."
        ],
        [
            (len(primes), "sylow-forced-normal-all"),
            (forced + 1, None),
            (max(forced - 1, 0), None),
            (sum(1 for p in primes if len(np_values(n, p)) == 2), "sylow-forced-normal-two-values"),
        ],
    )


# -- hard ------------------------------------------------------------------------------------------------------

_H1_STEMS = (
    "A group G of order {n} = {p} × {q} × {r} has n_{p} = {xp} and n_{q} = {xq}, and each Sylow subgroup has prime "
    "order. How many elements of G have an order other than 1, {p} or {q}?",
    "Suppose |G| = {n} = {p} × {q} × {r}, with n_{p} = {xp} and n_{q} = {xq} and prime-order Sylow subgroups. Count "
    "the elements of G whose order is not 1, {p} or {q}.",
)
_H2_STEMS = (
    "Suppose a group G of order {n} = {p} × {q} × {r} had n_{p} = {xp} and n_{q} = {xq}. Sylow subgroups here have "
    "prime order, so they meet only in the identity. Counting the identity once, how many more elements than "
    "|G| would these subgroups force G to have?",
    "A group G has order {n} = {p} × {q} × {r}. Assume n_{p} = {xp} and n_{q} = {xq}. By how much does the count of "
    "the identity together with every element of order {p} or {q} exceed |G|?",
)
_H3_STEMS = (
    "A group of order {g} acts on a set X with {N} elements. What is the smallest number of points of X that can be "
    "fixed by every element of the group?",
    "Let a {p}-group of order {g} act on a finite set X with |X| = {N}. Find the least possible number of elements "
    "of X fixed by the whole group.",
)
_H4_STEMS = (
    "A group G of order {n} has exactly {x} Sylow {q}-subgroups, each of prime order {q}. How many elements of G do "
    "not have order {q}? (The identity counts.)",
    "Let |G| = {n} and n_{q} = {x}, where the Sylow {q}-subgroups have order {q}. Find the number of elements of G "
    "that are not of order {q}.",
)


def _squarefree3_draw(rng):
    for _ in attempts(2000, "a triple order with usable n values"):
        n, (a, b, c) = rng.choice(_pool_squarefree3())
        trio = [a, b, c]
        rng.shuffle(trio)
        p, q, r = trio
        sp, sq = checked_np(n, p), checked_np(n, q)
        xp, xq = rng.choice(sp), rng.choice(sq)
        total = 1 + xp * (p - 1) + xq * (q - 1)
        if len(sp) + len(sq) > 2 and xp * xq > 1:
            return n, p, q, r, xp, xq, total
    raise AssertionError  # pragma: no cover


def _hard(rng):
    kind = rng.choice(("leftover", "surplus", "fixed", "notq"))
    if kind == "leftover":
        for _ in attempts(2000, "a consistent leftover"):
            n, p, q, r, xp, xq, total = _squarefree3_draw(rng)
            left = n - total
            if left >= r - 1 and (xp > 1 or xq > 1):
                break
        return number_q(
            phrase(rng, _H1_STEMS, n=n, p=p, q=q, r=r, xp=xp, xq=xq),
            left,
            [
                f"Elements of order {p}: {xp} × {p - 1} = {xp * (p - 1)}; of order {q}: {xq} × {q - 1} = {xq * (q - 1)}.",
                f"{n} - 1 - {xp * (p - 1)} - {xq * (q - 1)} = {left}.",
            ],
            [
                (left + 1, "sylow-leftover-forgets-identity"),
                (n - xp * p - xq * q, "sylow-leftover-counts-np-times-p"),
                (n - xp - xq, "sylow-leftover-one-per-subgroup"),
                (n - 1 - xp * (p - 1), "sylow-leftover-ignores-second-prime"),
            ],
            lo=0,
        )
    if kind == "surplus":
        for _ in attempts(2000, "a pair of counts that cannot coexist"):
            n, p, q, r, xp, xq, total = _squarefree3_draw(rng)
            if total > n:
                break
        key = total - n
        return number_q(
            phrase(rng, _H2_STEMS, n=n, p=p, q=q, r=r, xp=xp, xq=xq),
            key,
            [
                f"1 + {xp} × {p - 1} + {xq} × {q - 1} = {total}, which is {total} - {n} = {key} more than |G|.",
            ],
            [
                (key - 1, "sylow-leftover-forgets-identity"),
                (total - n + xp + xq, None),
                (xp * p + xq * q - n, "sylow-leftover-counts-np-times-p"),
                (total, "sylow-surplus-total-not-excess"),
            ],
            lo=1,
        )
    if kind == "fixed":
        p = rng.choice((3, 5, 7, 11, 13))
        k = rng.randint(1, 3 if p <= 5 else 2)
        N = rng.randint(40, 4000)
        key = N % p
        # a second route: Z_p acting by (N - key)/p disjoint p-cycles realises exactly `key` fixed points
        moved = N - key
        check(moved % p == 0 and key < p, "the realising action is not a union of p-cycles")
        slips = [
            (N // p, "sylow-fixed-points-quotient"),
            (N % (p**k) if k >= 2 else None, "sylow-fixed-points-mod-group-order"),
            ((p - key) if key else p, "sylow-fixed-points-complement"),
            (key + p, "sylow-fixed-points-adds-p"),
        ]
        return number_q(
            phrase(rng, _H3_STEMS, g=p**k, p=p, N=N),
            key,
            [
                f"Every orbit has {p}-power size, so the points not fixed come in orbits whose sizes are multiples of {p}.",
                f"So the number of fixed points is {N} mod {p} = {key} at least, and {key} is reached.",
            ],
            slips,
            lo=0,
        )
    for _ in attempts(1000, "a p^k q order with n_q more than 1"):
        n, p, q = rng.choice(_pool_pkq())
        values = checked_np(n, q)
        if q >= 3 and len(values) >= 2:
            x = rng.choice(values[1:])
            if n - x * (q - 1) not in (n, x, q, p):
                break
    key = n - x * (q - 1)
    return number_q(
        phrase(rng, _H4_STEMS, n=n, q=q, x=x),
        key,
        [f"Elements of order {q}: {x} × {q - 1} = {x * (q - 1)}; the rest is {n} - {x * (q - 1)} = {key}."],
        [
            (x * (q - 1), "sylow-elements-count-complement"),
            (n - x * q, "sylow-elements-include-identity"),
            (n - x, "sylow-elements-one-per-subgroup"),
            (key - 1, "sylow-leftover-forgets-identity"),
        ],
        lo=0,
    )


@generator(
    "w5b_sylow_counting",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@with_latex
def w5b_sylow_counting(rng, tier="easy"):
    """Sylow counting; the tiers are described in the module docstring."""
    build = {"easy": _easy, "medium": _medium, "hard": _hard}.get(tier)
    if build is None:
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(50, "a question whose key is not printed in its stem"):
        d = build(rng)
        # a key that is also a number in the stem can be guessed from the text alone
        if int(d["answer_text"]) < 4 or d["answer_text"] not in re.findall(r"\d+", d["text"]):
            return d


# -- exam ------------------------------------------------------------------------------------------------------


_EXAM_A = (
    "G is a group of order {n} = {fac}, so every Sylow subgroup has prime order.",
    "Let G be a group whose order is {n} = {fac}; each of its Sylow subgroups has prime order.",
)
_EXAM_C = (
    "A group G has order {n} = {pk} × {q}, where {p} and {q} are primes. The Sylow {q}-subgroups have prime order {q}.",
    "G is a finite group of order {n}, which is {pk} × {q} for the primes {p} and {q}; a Sylow {q}-subgroup has "
    "order {q}.",
)
_EXAM_D = (
    "G is a group of order {n} = {pk} × {q} × {r}, where {p}, {q} and {r} are distinct primes. The Sylow "
    "{q}-subgroups and Sylow {r}-subgroups have prime order.",
    "Let |G| = {n} = {pk} × {q} × {r} with {p}, {q}, {r} distinct primes; the Sylow {q}-subgroups and the Sylow "
    "{r}-subgroups have prime order.",
)

_SET_PROMPTS = (
    "Find the set of possible values of n_{q}, the number of Sylow {q}-subgroups, typed like {{a, b, c}}.",
    "List every value that n_{q}, the number of Sylow {q}-subgroups of G, could take, as a set typed like {{a, b, c}}.",
)


def _elements_part(q, xq, n, marks=2):
    elems = xq * (q - 1)
    return number_part(
        f"Suppose n_{q} = {xq}. How many elements of G have order {q}?",
        elems,
        marks,
        [f"{xq} subgroups of order {q} meeting only in the identity give {xq} × {q - 1} = {elems} elements."],
        [
            (xq * q, "sylow-elements-include-identity"),
            (xq, "sylow-elements-one-per-subgroup"),
            (elems + 1, "sylow-elements-count-identity"),
            (n - elems, "sylow-elements-complement"),
        ],
        sensible_range=(0, None),
        lo=0,
    )


def _leftover_part(n, p, q, xp, elems, min_left, word):
    left = n - 1 - elems - xp * (p - 1)
    return number_part(
        f"Suppose also that n_{p} = {xp}. How many elements of G then have an order other than 1, {p} or {q}?{word}",
        left,
        3,
        [f"{n} - 1 - {elems} - {xp} × {p - 1} = {left}."],
        [
            (left + 1, "sylow-leftover-forgets-identity"),
            (n - elems - xp * p, "sylow-leftover-counts-np-times-p"),
            (n - elems - xp, "sylow-leftover-one-per-subgroup"),
            (n - 1 - xp * (p - 1), "sylow-leftover-ignores-second-prime"),
        ],
        follow_through=f"{n} - 1 - part_b - {xp * (p - 1)}",
        sensible_range=(min_left - 1, None),
        lo=0,
    )


def _exam_a(rng):
    for _ in attempts(2000, "a squarefree order with a usable prime pair"):
        n, combo = rng.choice(_pool_squarefree34())
        trio = list(combo)
        rng.shuffle(trio)
        q, p, *others = trio
        sq, sp = checked_np(n, q), checked_np(n, p)
        if q >= 3 and len(sq) >= 2 and len(sp) >= 2:
            xq = rng.choice(sq[1:])
            xp = rng.choice(sp[1:])
            min_left = sum(o - 1 for o in others)
            if n - 1 - xq * (q - 1) - xp * (p - 1) >= min_left:
                break
    return dict(
        text=phrase(rng, _EXAM_A, n=n, fac=" × ".join(map(str, sorted(combo)))),
        parts=[
            set_part_for(rng, n, q),
            _elements_part(q, xq, n),
            _leftover_part(n, p, q, xp, xq * (q - 1), min_left, ""),
        ],
    )


def _exam_c(rng):
    for _ in attempts(500, "a p^k q order with q > 2 whose leftover holds a Sylow p-subgroup"):
        n, p, q = rng.choice(_pool_pkq())
        xq = rng.choice(checked_np(n, q)[1:])
        if q >= 3 and n - xq * (q - 1) >= n // q:
            break
    elems = xq * (q - 1)
    left = n - elems
    return dict(
        text=phrase(rng, _EXAM_C, n=n, p=p, q=q, pk=n // q),
        parts=[
            set_part_for(rng, n, q),
            _elements_part(q, xq, n),
            number_part(
                f"How many elements of G are then not of order {q}? These include a whole Sylow {p}-subgroup.",
                left,
                3,
                [f"{n} - {elems} = {left}."],
                [
                    (elems, "sylow-elements-count-complement"),
                    (left - 1, "sylow-leftover-forgets-identity"),
                    (n - xq, "sylow-elements-one-per-subgroup"),
                    (n - xq * q, "sylow-elements-include-identity"),
                ],
                follow_through=f"{n} - part_b",
                sensible_range=(0, None),
                lo=0,
            ),
        ],
    )


@lru_cache(maxsize=None)
def _pool_p2qr():
    """Orders p^2 q r with p, q, r distinct primes (n up to 1000): (n, p, q, r)."""
    out = []
    for p in (2, 3, 5, 7):
        for q in PRIMES[:10]:
            for r in PRIMES[:10]:
                n = p * p * q * r
                if len({p, q, r}) == 3 and q < r and n <= 2500:
                    out.append((n, p, q, r))
    return tuple(sorted(out))


def _exam_d(rng):
    for _ in attempts(2000, "a p^2 q r order with a usable prime pair"):
        n, p, a, b = rng.choice(_pool_p2qr())
        q, r = (a, b) if rng.random() < 0.5 else (b, a)
        sq, sr = checked_np(n, q), checked_np(n, r)
        if q >= 3 and len(sq) >= 2 and len(sr) >= 2:
            xq = rng.choice(sq[1:])
            xr = rng.choice(sr[1:])
            if n - 1 - xq * (q - 1) - xr * (r - 1) >= p * p - 1:
                break
    return dict(
        text=phrase(rng, _EXAM_D, n=n, p=p, q=q, r=r, pk=p * p),
        parts=[
            set_part_for(rng, n, q),
            _elements_part(q, xq, n),
            _leftover_part(n, r, q, xr, xq * (q - 1), p * p - 1, " They include a Sylow {p}-subgroup.".format(p=p)),
        ],
    )


def set_part_for(rng, n, q):
    values = checked_np(n, q)
    m = n // sylow_order(n, q)
    return set_part(
        phrase(rng, _SET_PROMPTS, q=q),
        values,
        2,
        [f"n_{q} divides {m} and n_{q} = 1 mod {q}, so n_{q} is one of {', '.join(map(str, values))}."],
        _nonempty(_set_candidates(n, q)),
    )


@generator("exd_x_w5b_sylow_counting", level=19, presets={"exam": dict(tier="exam")})
@with_latex
def exd_x_w5b_sylow_counting(rng, tier="exam"):
    """Exam chain: possible n_q, elements of order q for the largest n_q, then the elements left over."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    shape = rng.choice(("three_primes", "three_primes", "three_primes", "p_k_q", "p2_q_r", "p2_q_r"))
    if shape == "three_primes":
        return _exam_a(rng)
    if shape == "p_k_q":
        return _exam_c(rng)
    return _exam_d(rng)
