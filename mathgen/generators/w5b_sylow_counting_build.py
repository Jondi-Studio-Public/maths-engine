"""Generates `w5b_sylow_counting` questions (easy, medium, hard) and the exam chain `exd_x_w5b_sylow_counting`.

``uc2-sylow-counting`` (``abstract-algebra``, answer ``number``, some tiers ``set``): Sylow's theorems for a finite
group of stated order n, with the arithmetic done from the divisors and congruences of n.

* easy: the order of a Sylow p-subgroup (the largest power of p dividing n), its index in the group, or the sum of the
  Sylow orders, one for each prime dividing n.
* medium: Sylow's counting conditions (n_p divides n / p^k and is 1 mod p): how many values n_p may take, the largest
  of them, their sum, the number of elements of order p for a supposed n_p (Sylow subgroups of prime order meet only in
  the identity, so n_p (p - 1)), and for how many primes the Sylow subgroup is forced to be normal by counting alone.
* hard: for a supposed pair of counts in a group of order p q r, the elements left over or the surplus that makes the
  pair impossible; the least number of fixed points of a p-group acting on N points (N mod p); and the elements not of
  order q for a supposed n_q.
* exam: typed three-part chains: (a) the set of possible n_q (typed with a space after each comma, and no possible
  value has exactly three digits, so the comma-less spelling is not read as a thousands-grouped number), (b) the number
  of elements of order q for a supposed n_q, (c) the elements left over (follows through from (b)).

A stated n_p = x need not occur in any group of that order, so every stem that states one supposes it ("Suppose a group
of order N had n_p = x"): the element counts are a counting exercise, not a claim that the group exists.

Every n comes from a curated pool of orders whose Sylow subgroups for the primes asked about have prime order when an
element count is asked, so the counts are exact; each answer is recomputed by a second route (divisors of n tested
for the congruence, a permutation of the right cycle type for the fixed points).
"""

from __future__ import annotations

import itertools
import math
import re
from fractions import Fraction
from functools import lru_cache

from ..answers import exact_value
from ..core import attempts, check, generator, phrase
from ._r4a_alg_kit import set_part, with_latex
from ._spread_kit import uses_rng
from ._w5b_uniform_rank_kit import balanced

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
    """Divisors of m ascending, by pairing d with m / d for d up to the square root."""
    small = [d for d in range(1, math.isqrt(m) + 1) if m % d == 0]
    return sorted({*small, *(m // d for d in small)})


def sylow_order(n, p):
    return p ** factor(n).get(p, 0)


def np_values(n, p):
    """Possible n_p: divisors of n / p^k that are 1 mod p, ascending."""
    return [d for d in divisors(n // sylow_order(n, p)) if d % p == 1 % p]


def np_values_by_scan(n, p):
    """The same set by a second route: every divisor of n built from its prime factorisation that is 1 mod p (such a
    divisor cannot contain p, so it divides n / p^k)."""
    built = [1]
    for prime, exponent in factor(n).items():
        built = [d * prime**e for d in built for e in range(exponent + 1)]
    return sorted(d for d in built if d % p == 1 % p)


def checked_np(n, p):
    values = np_values(n, p)
    check(values == np_values_by_scan(n, p), "the two routes to the possible n_p disagree")
    return values


@lru_cache(maxsize=None)
def _pool_orders():
    """Orders 12..1500 with at least two prime divisors."""
    return tuple(n for n in range(12, 1501) if len(factor(n)) >= 2)


@lru_cache(maxsize=None)
def _pool_orders3():
    """Orders 30..15000 with at least three distinct prime divisors."""
    return tuple(n for n in range(30, 15001) if len(factor(n)) >= 3)


@lru_cache(maxsize=None)
def _pool_squarefree3():
    """Products of three distinct primes up to 1100: (n, (p, q, r))."""
    out = []
    for i, a in enumerate(PRIMES[:11]):
        for j in range(i + 1, 11):
            for k in range(j + 1, 11):
                n = a * PRIMES[j] * PRIMES[k]
                if n <= 3000:
                    out.append((n, (a, PRIMES[j], PRIMES[k])))
    return tuple(sorted(out))


@lru_cache(maxsize=None)
def _pool_pq():
    """Orders p q with p < q primes and p dividing q - 1: (p, q)."""
    return tuple((p, q) for p in PRIMES for q in PRIMES if p < q and (q - 1) % p == 0 and p * q <= 600)


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


def number_q(text, value, working, slips, *, keep=4, lo=0):
    options = balanced(Fraction(value), slips, keep, kind="number", lo=lo)
    return dict(
        text=text,
        answer_kind="number",
        answer=exact_value(value),
        answer_text=str(value),
        answer_latex=str(value),
        working=working,
        distractors=options,
    )


def number_part(prompt, value, marks, working, slips, *, follow_through=None, sensible_range=None, keep=3, lo=0):
    part = dict(
        prompt=prompt,
        answer_kind="number",
        answer=exact_value(value),
        answer_text=str(value),
        marks=marks,
        working=working,
        distractors=balanced(Fraction(value), slips, keep, kind="number", lo=lo),
    )
    if follow_through:
        part["follow_through"] = follow_through
    if sensible_range is not None:
        part["sensible_range"] = sensible_range
    return part


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
        ([d for d in all_div if d != 1 and d not in right], None),
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
_E4_STEMS = (
    "A finite group G has order {n}. What is the index [G : P] of a Sylow {p}-subgroup P in G?",
    "Let |G| = {n} and let P be a Sylow {p}-subgroup of G. How many left cosets of P are there in G?",
)
_E3_STEMS = (
    "A group G has order {n}. For each prime dividing {n} take the order of a Sylow subgroup for that prime. What is "
    "the sum of these orders?",
    "G is a finite group of order {n}. Add together the orders of its Sylow subgroups, one for each prime dividing "
    "{n}. What is the total?",
)


def _easy(rng):
    kind = rng.choice(("order", "orders_sum", "index"))
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
    if kind == "index":
        for _ in attempts(500, "an order with a Sylow subgroup of index more than 1"):
            n = rng.choice(_pool_orders())
            p = rng.choice(sorted(factor(n)))
            if p <= 13 and factor(n)[p] >= 1:
                break
        k = factor(n)[p]
        m = n // p**k
        check(m * p**k == n and m % p != 0, "the index is not n over the full power of p")
        return number_q(
            phrase(rng, _E4_STEMS, n=n, p=p),
            m,
            [f"{n} = {p}^{k} × {m} with {p} not dividing {m}, so |P| = {p**k} and [G : P] = {n}/{p**k} = {m}."],
            [
                (p**k, "sylow-index-as-order"),
                (n // p, "sylow-index-divides-by-p-once"),
                (n - p**k, None),
                (m + p, None),
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
    "Sylow's theorems restrict n_{p}, the number of Sylow {p}-subgroups, for a group of order {n}. How many different "
    "values does n_{p} have to choose from?",
)
_M2_STEMS = (
    "G is a group of order {n}. What is the largest value that n_{p}, the number of Sylow {p}-subgroups, could have?",
    "Sylow's theorems restrict the number n_{p} of Sylow {p}-subgroups of a group of order {n}. Find the greatest "
    "value they allow.",
    "Of the values that Sylow's theorems allow for the number n_{p} of Sylow {p}-subgroups of a group of order {n}, "
    "which is the biggest?",
)
_M3_STEMS = (
    "Suppose a group G of order {n} had n_{p} = {x} Sylow {p}-subgroups, each of prime order {p}. How many elements "
    "of G would have order {p}?",
    "Suppose |G| = {n} and n_{p} = {x}, and every Sylow {p}-subgroup has order {p}. Find the number of elements of "
    "G of order {p}.",
    "Imagine a group of order {n} with n_{p} = {x}, where each Sylow {p}-subgroup has prime order {p}. Counting "
    "subgroup by subgroup, how many elements of order {p} would it contain?",
)
_M4_STEMS = (
    "Take a group of order {n}. For how many of the primes dividing {n} does Sylow's theorem force the number of "
    "Sylow subgroups to be 1, so that the Sylow subgroup is normal?",
    "G has order {n}. Count the primes p dividing {n} for which the only value of n_p allowed by Sylow's theorems "
    "is 1.",
    "For a group of order {n}, Sylow's theorems restrict n_p for each prime p dividing {n}. For how many of these "
    "primes is n_p = 1 the only possibility?",
)
_M5_STEMS = (
    "A group G has order {n}, and n_{p} is the number of its Sylow {p}-subgroups. Add up every value that n_{p} could "
    "take by Sylow's theorems. What is the sum?",
    "Let |G| = {n}. Using Sylow's theorems, find every value that the number n_{p} of Sylow {p}-subgroups of G could "
    "have, and give the sum of all of them.",
    "Sylow's theorems allow several values for n_{p}, the number of Sylow {p}-subgroups of a group of order {n}. What "
    "is their total?",
)


def _medium(rng):
    kind = rng.choice(("count", "largest", "sum", "elements", "forced"))
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
    if kind == "sum":
        for _ in attempts(500, "an order with at least two possible n_p"):
            n = rng.choice(_pool_orders())
            p = rng.choice(sorted(factor(n)))
            values = checked_np(n, p)
            if p <= 13 and 3 <= len(values) <= 7:
                break
        m = n // sylow_order(n, p)
        return number_q(
            phrase(rng, _M5_STEMS, n=n, p=p),
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
    # how many primes force n_p = 1: three or more primes, and a target count drawn first (1, 2 or 3) so the key spreads
    target = rng.choice((1, 2, 3))
    for attempt in attempts(6000, "an order with at least three primes, some forced and some free"):
        n = rng.choice(_pool_orders3())
        primes = sorted(factor(n))
        forced = sum(1 for p in primes if checked_np(n, p) == [1])
        if 1 <= forced <= len(primes) - 1 and (forced == target or attempt > 2500):
            break
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
            (forced - 1, None),
            (sum(1 for p in primes if len(np_values(n, p)) == 2), "sylow-forced-normal-two-values"),
        ],
        lo=0,
    )


# -- hard ------------------------------------------------------------------------------------------------------

_H1_STEMS = (
    "Suppose a group G of order {n} = {p} × {q} × {r} had n_{p} = {xp} and n_{q} = {xq}, with each Sylow subgroup of "
    "prime order. How many elements of G would have an order other than 1, {p} or {q}?",
    "Suppose |G| = {n} = {p} × {q} × {r}, with n_{p} = {xp} and n_{q} = {xq} and prime-order Sylow subgroups. Count "
    "the elements of G whose order would not be 1, {p} or {q}.",
    "Imagine a group G of order {n} = {p} × {q} × {r} with n_{p} = {xp} and n_{q} = {xq}, all Sylow subgroups having "
    "prime order. How many of its elements would have an order other than 1, {p} or {q}?",
)
_H2_STEMS = (
    "Suppose a group G of order {n} = {p} × {q} × {r} had n_{p} = {xp} and n_{q} = {xq}. Sylow subgroups here have "
    "prime order, so they meet only in the identity. Counting the identity once, how many more elements than "
    "|G| would these subgroups force G to have?",
    "Suppose a group G has order {n} = {p} × {q} × {r}, and suppose n_{p} = {xp} and n_{q} = {xq}. By how much does "
    "the count of the identity together with every element of order {p} or {q} exceed |G|?",
    "Take n_{p} = {xp} and n_{q} = {xq} for a hypothetical group G of order {n} = {p} × {q} × {r}, with prime-order "
    "Sylow subgroups. By how many does the identity plus the elements of order {p} or {q} outnumber the {n} elements "
    "available?",
)
_H3_STEMS = (
    "A group of order {g} acts on a set X with {N} elements. What is the smallest number of points of X that can be "
    "fixed by every element of the group?",
    "Let a group of order {g}, a power of the prime {p}, act on a finite set X with |X| = {N}. Find the least possible "
    "number of elements of X fixed by the whole group.",
    "A group whose order {g} is a power of {p} permutes the points of a set X with {N} elements. At least how many "
    "points does the whole group fix?",
)
_H4_STEMS = (
    "Suppose a group G of order {n} had exactly {x} Sylow {q}-subgroups, each of prime order {q}. How many elements "
    "of G would not have order {q}? (The identity counts.)",
    "Let |G| = {n} and suppose n_{q} = {x}, where the Sylow {q}-subgroups have order {q}. Find the number of elements "
    "of G that would not be of order {q}.",
    "Suppose a group of order {n} contained n_{q} = {x} Sylow {q}-subgroups, each of prime order {q}. How many of "
    "its elements would not be of order {q}, the identity included?",
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
        N = rng.randint(40, 9000)
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
    for _ in attempts(3000, "an order with a prime q exactly dividing it and n_q more than 1"):
        n = rng.choice(_pool_orders())
        primes = [q for q in factor(n) if 3 <= q <= 31 and factor(n)[q] == 1 and len(np_values(n, q)) >= 2]
        if not primes:
            continue
        q = rng.choice(primes)
        x = rng.choice(checked_np(n, q)[1:])
        if n - x * (q - 1) not in (n, x, q):
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
            (key - 1, None),
        ],
        lo=0,
    )


@generator(
    "w5b_sylow_counting",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@uses_rng
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


_EXAM_X = (
    "G is a group of order {n} = {q} × {m}, where {q} is a prime that does not divide {m}. The Sylow {q}-subgroups "
    "of any such group would have prime order {q}.",
    "Let |G| = {n}, which is {q} × {m} for a prime {q} not dividing {m}; a Sylow {q}-subgroup has order {q}.",
)
_EXAM_Y = (
    "G is a group of order {n} = {p} × {q} × {m}, where {p} and {q} are distinct primes that do not divide {m}. "
    "The Sylow {p}-subgroups and Sylow {q}-subgroups would have prime order.",
    "A group G has order {n}, the product {p} × {q} × {m} with the primes {p} and {q} not dividing {m}; its Sylow "
    "{p}-subgroups and {q}-subgroups would have prime order.",
)
_SET_PROMPTS = (
    "Find the set of possible values of n_{q}, the number of Sylow {q}-subgroups. Type it in braces with each comma "
    "followed by a space, like {{a, b, c}}.",
    "List every value that n_{q}, the number of Sylow {q}-subgroups of G, could take, as a set typed in braces with a "
    "space after each comma, like {{a, b, c}}.",
)
_ELEMENTS = (
    "Suppose n_{q} = {xq}. How many elements of G would have order {q}?",
    "Suppose G had exactly {xq} Sylow {q}-subgroups. How many of its elements would then have order {q}?",
)


def _max_prime_power(m):
    return max(p**e for p, e in factor(m).items())


def _set_is_unambiguous(values):
    """A set of values whose comma-separated typing without spaces cannot be read as a thousands-grouped number: the
    grader reads `1,111` as 1111, so no value may have exactly three digits (it follows a smaller value)."""
    return not any(100 <= v <= 999 for v in values)


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


def _elements_part(rng, q, xq, n):
    elems = xq * (q - 1)
    return number_part(
        phrase(rng, _ELEMENTS, q=q, xq=xq),
        elems,
        2,
        [f"{xq} subgroups of order {q} meeting only in the identity give {xq} × {q - 1} = {elems} elements."],
        [
            (xq * q, "sylow-elements-include-identity"),
            (xq, "sylow-elements-one-per-subgroup"),
            (elems + 1, "sylow-elements-count-identity"),
            (n - elems, "sylow-elements-complement"),
        ],
        sensible_range=(0, None),
        lo=1,
    )


_Q_PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97)


def _visible(stem, parts):
    """Whether a numeric key is a number printed in the stem or in any prompt (so it could be read off)."""
    shown = set(re.findall(r"\d+", " ".join([stem, *(p["prompt"] for p in parts)])))
    return any(p["answer_kind"] == "number" and p["answer_text"] in shown for p in parts)


def _pick_x(rng, q, cap):
    """A possible value x = 1 + q t of n_q that is not three digits long (so the set can be typed without spaces)."""
    options = [1 + q * t for t in range(1, 60) if (1 + q * t) <= 99 or 1000 <= 1 + q * t <= cap]
    return rng.choice(options) if options else None


def _exam_x(rng):
    """n = q m: the set of n_q, the elements of order q for a stated n_q, the elements that are not of order q."""
    for _ in attempts(3000, "an order q m with several n_q and room for the other Sylow subgroups"):
        q = rng.choice(_Q_PRIMES)
        x = _pick_x(rng, q, 3000)
        if x is None:
            continue
        m = x * rng.randint(1, max(1, 40000 // (q * x)))
        if m % q == 0 or m < 4:
            continue
        n = q * m
        values = checked_np(n, q)
        if not 2 <= len(values) <= 8 or not _set_is_unambiguous(values):
            continue
        xq = x if rng.random() < 0.75 and x in values[1:] else rng.choice(values[1:])
        left = n - xq * (q - 1)
        if left < _max_prime_power(m):
            continue
        parts = [
            set_part_for(rng, n, q),
            _elements_part(rng, q, xq, n),
            number_part(
                f"How many elements of G are then not of order {q}?",
                left,
                3,
                [f"{n} - {xq * (q - 1)} = {left}."],
                [
                    (xq * (q - 1), "sylow-elements-count-complement"),
                    (left - 1, None),
                    (n - xq, "sylow-elements-one-per-subgroup"),
                    (n - xq * q, "sylow-elements-include-identity"),
                ],
                follow_through=f"{n} - part_b",
                sensible_range=(0, None),
                lo=1,
            ),
        ]
        stem = phrase(rng, _EXAM_X, n=n, q=q, m=m)
        if not _visible(stem, parts):
            return dict(text=stem, parts=parts)
    raise AssertionError  # pragma: no cover


def _exam_y(rng):
    """n = p q m: the set of n_q, the elements of order q, then the elements of neither order p, q nor 1."""
    for _ in attempts(5000, "an order p q m with several n_p and n_q and room for the rest"):
        p, q = rng.sample(
            (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97), 2
        )
        if q == 2:
            continue
        if p * q > 2000:
            continue
        x = _pick_x(rng, q, 3000)
        if x is None:
            continue
        m = x * rng.randint(1, max(1, 40000 // (p * q * x)))
        if m % p == 0 or m % q == 0:
            continue
        n = p * q * m
        sq, sp = checked_np(n, q), checked_np(n, p)
        if not (2 <= len(sq) <= 8 and len(sp) >= 2 and _set_is_unambiguous(sq)):
            continue
        xq, xp = (x if rng.random() < 0.75 and x in sq[1:] else rng.choice(sq[1:])), rng.choice(sp[1:])
        left = n - 1 - xq * (q - 1) - xp * (p - 1)
        if left < _max_prime_power(m) - 1 or left < 1:
            continue
        parts = [
            set_part_for(rng, n, q),
            _elements_part(rng, q, xq, n),
            number_part(
                f"Suppose also that n_{p} = {xp}. How many elements of G then have an order other than 1, {p} or {q}?",
                left,
                3,
                [f"{n} - 1 - {xq * (q - 1)} - {xp} × {p - 1} = {left}."],
                [
                    (left + 1, "sylow-leftover-forgets-identity"),
                    (n - xq * (q - 1) - xp * p, "sylow-leftover-counts-np-times-p"),
                    (n - xq * (q - 1) - xp, "sylow-leftover-one-per-subgroup"),
                    (n - 1 - xp * (p - 1), "sylow-leftover-ignores-second-prime"),
                ],
                follow_through=f"{n} - 1 - part_b - {xp * (p - 1)}",
                sensible_range=(0, None),
                lo=1,
            ),
        ]
        stem = phrase(rng, _EXAM_Y, n=n, p=p, q=q, m=m)
        if not _visible(stem, parts):
            return dict(text=stem, parts=parts)
    raise AssertionError  # pragma: no cover


@generator("exd_x_w5b_sylow_counting", level=19, presets={"exam": dict(tier="exam")})
@uses_rng
@with_latex
def exd_x_w5b_sylow_counting(rng, tier="exam"):
    """Exam chain: possible n_q, elements of order q for a stated n_q, then the elements left over."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam_x(rng) if rng.random() < 0.5 else _exam_y(rng)
