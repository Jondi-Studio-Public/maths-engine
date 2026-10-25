"""Exam source for `senior-use-proof-by-contradiction`: carry a proof by contradiction through with typed numbers.

The practice tiers ask which stated fact contradicts an assumption, as a multiple choice. Here the learner works
inside the proof: the assumption is made, and each part asks for a concrete quantity the argument needs, ending at
the value that produces (or shows) the contradiction. Every answer is a typed whole number. Three families:

  * ``euclid`` -- "suppose these are the only primes". (a) M = (product of the listed primes) + 1; (b) the smallest
    prime factor of M, which cannot be in the list; (c) the cofactor M / (b) (follow-through from (a) and (b)).
  * ``linear`` -- "suppose A m + B n = C has integer solutions" where gcd(A, B) does not divide C. (a) the largest
    whole number dividing A m + B n for every m, n; (b) the remainder C leaves on division by it (never 0, which is
    the contradiction); (c) how many of 1..C can be written A m + B n; (d) the least such number above C
    ((c) and (d) follow through from (a)).
  * ``residue`` -- "suppose m divides n^2 + c". (a) the remainder n^2 must leave on division by m; (b) how many
    different remainders a square can leave; (c) how many of the m remainders a square can never leave
    (follow-through from (b)); (d) the least positive k for which m could divide n^2 + k.

Every key comes from brute force over the integers or the residues, never from the formula the working quotes.
"""

from __future__ import annotations

import math
import random
from itertools import combinations

from sympy import factorint, isprime

from ..core import attempts, check, generator
from ._exd_x_pc_kit import count_part
from ._spread_kit import uses_rng

_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23)


def _list_text(values) -> str:
    return ", ".join(str(v) for v in values)


def _smallest_factor(m: int) -> int:
    return next(d for d in range(2, m + 1) if m % d == 0)


def _euclid(rng: random.Random) -> dict:
    for _ in attempts(1000, "a prime list whose product plus one is composite"):
        size = rng.choice((3, 3, 4, 4, 5))
        if rng.random() < 0.3:
            listed = _PRIMES[:size]
        else:
            listed = (2, *sorted(rng.sample(_PRIMES[1:8], size - 1)))
        product = math.prod(listed)
        m = product + 1
        if not isprime(m):
            break
    low = _smallest_factor(m)
    cofactor = m // low
    check(low not in listed and cofactor > 1, "the smallest factor of M must be new")
    check(all(m % p == 1 for p in listed), "M must leave remainder 1 on division by every listed prime")
    factors = factorint(m)
    largest = max(factors)
    stem = rng.choice(
        (
            f"To prove by contradiction that there are infinitely many primes, suppose that the only primes are "
            f"{_list_text(listed)}. Let M be the product of these primes, plus 1.",
            f"A proof that the primes never run out begins by assuming, for contradiction, that "
            f"{_list_text(listed)} are all the primes there are. Define M = (the product of all of them) + 1.",
            f"Assume, for contradiction, that there are exactly {len(listed)} primes: {_list_text(listed)}. "
            "Let M be one more than the product of all of them.",
        )
    )
    a = count_part(
        "Find M.",
        m,
        1,
        [f"M = {' × '.join(str(p) for p in listed)} + 1 = {product} + 1 = {m}."],
        [
            (product, "pcx-forgets-to-add-one"),
            (sum(listed) + 1, "pcx-adds-instead-of-multiplying"),
            (product - 1, "pcx-subtracts-one"),
            (product + 2, None),
        ],
    )
    b = count_part(
        "Find the smallest prime factor of M. It cannot be one of the listed primes.",
        low,
        2,
        [
            "Every listed prime leaves remainder 1 when divided into M, so none of them divides M.",
            f"Trial division of {m}: the smallest prime factor is {low}, a prime that was not on the list.",
        ],
        [
            (listed[0], "pcx-picks-a-listed-prime"),
            (largest, "pcx-takes-the-largest-factor"),
            (m, "pcx-takes-m-as-the-prime"),
            (cofactor, "pcx-takes-the-cofactor"),
            (low + 1, None),
            (low + 2, None),
        ],
    )
    c = count_part(
        "Write M = p × q, where p is your answer to part (b). Find q.",
        cofactor,
        1,
        [f"q = M ÷ p = {m} ÷ {low} = {cofactor}."],
        [
            (low, "pcx-repeats-the-previous-answer"),
            (m - low, "pcx-subtracts-instead-of-dividing"),
            (m // largest if largest != low else None, "pcx-divides-by-the-wrong-factor"),
            (cofactor + 1, None),
            (cofactor - 1, None),
        ],
        follow=f"{m}/part_b",
    )
    if a is None or b is None or c is None:
        raise ValueError("retry")
    return dict(text=stem, parts=[a, b, c])


def _linear(rng: random.Random) -> dict:
    for _ in attempts(1000, "coefficients with a common factor that misses the constant"):
        g = rng.choice((2, 3, 4, 5, 6, 7, 8, 9))
        p, q = rng.sample(range(1, 10), 2)
        if math.gcd(p, q) != 1:
            continue
        a_coef, b_coef = g * p, g * q
        const = rng.randint(15, 220)
        if const % g == 0:
            continue
        break
    # The largest whole number dividing A m + B n for every integer m, n is found from the values themselves.
    d = 0
    for m in range(-6, 7):
        for n in range(-6, 7):
            d = math.gcd(d, a_coef * m + b_coef * n)
    check(d == g, "the common divisor of every value A m + B n is the gcd")
    r = const % d
    count = const // d
    least_above = (const // d + 1) * d
    check(r != 0 and least_above > const, "C must not be a multiple of the divisor")
    expr = f"{a_coef}m + {b_coef}n"
    stem = rng.choice(
        (
            f"Claim: there are no integers m and n such that {expr} = {const}. To prove it by contradiction, "
            "suppose that such integers m and n exist.",
            f"Suppose, for contradiction, that {expr} = {const} for some integers m and n.",
            f"A student says that {expr} = {const} has no solution in integers m and n. "
            "Begin a proof by contradiction by assuming that it does have a solution.",
        )
    )
    a = count_part(
        f"What is the largest positive integer that divides {expr} for every choice of integers m and n?",
        d,
        1,
        [
            f"Both {a_coef} and {b_coef} are multiples of {d}, so {expr} is too, and no larger number works for all m, n."
        ],
        [
            (min(a_coef, b_coef), "pcx-takes-the-smaller-coefficient"),
            (math.lcm(a_coef, b_coef), "pcx-takes-lcm-for-gcd"),
            (abs(a_coef - b_coef) if abs(a_coef - b_coef) != d else None, "pcx-takes-the-difference"),
            (d + 1, None),
            (d * 2, None),
        ],
    )
    b = count_part(
        f"If the assumption were true, {const} would be divisible by your answer to part (a). "
        f"What remainder does {const} leave when divided by it? (It is not 0, so the assumption is contradicted.)",
        r,
        1,
        [f"{const} = {count} × {d} + {r}, so the remainder is {r}."],
        [
            (v, tag)
            for v, tag in (
                (d - r, "pcx-remainder-from-above"),
                (count, "pcx-quotient-for-remainder"),
                (r + 1, None),
                (r + 2, None),
                (r - 1, None),
            )
            if 0 < v < d  # a remainder on division by d is always less than d
        ],
        hi=d - 1,
        follow=f"{const} - part_a*floor({const}/part_a)",
    )
    c = count_part(
        f"Taking your answer to part (a) as the divisor, how many of the integers from 1 to {const} "
        f"can be written in the form {expr} for some integers m and n?",
        count,
        2,
        [
            f"The integers of the form {expr} are exactly the multiples of {d}.",
            f"Multiples of {d} from 1 to {const}: {const} ÷ {d} gives {count} (rounded down).",
        ],
        [
            (count + 1, "pcx-rounds-up-the-count"),
            (const // min(a_coef, b_coef), "pcx-divides-by-a-coefficient"),
            (const - count, "pcx-counts-the-complement"),
            (count - 1, None),
            (count + 2, None),
        ],
        follow=f"floor({const}/part_a)",
    )
    e = count_part(
        f"What is the smallest integer greater than {const} that can be written in the form {expr}?",
        least_above,
        2,
        [
            f"It must be a multiple of {d}. The multiples of {d} either side of {const} are {count * d} and {least_above}."
        ],
        [
            (count * d, "pcx-takes-the-multiple-below"),
            (const + 1, "pcx-takes-the-next-integer"),
            (const + r, "pcx-adds-the-remainder"),
            (const + d, "pcx-adds-the-divisor"),
            (least_above + d, None),
        ],
        follow=f"(floor({const}/part_a) + 1)*part_a",
    )
    if a is None or b is None or c is None or e is None:
        raise ValueError("retry")
    return dict(text=stem, parts=[a, b, c, e])


def _residue(rng: random.Random) -> dict:
    for _ in attempts(1000, "a modulus and constant where n^2 + c is never divisible by m"):
        m = rng.randint(5, 30)
        c = rng.randint(1, 3 * m)
        squares = {(n * n) % m for n in range(m)}
        if c % m and (-c) % m not in squares and len(squares) < m - 1:
            if next(k for k in range(1, m + 1) if (-k) % m in squares) >= 2:
                break
    needed = (-c) % m
    s = len(squares)
    missing = m - s
    k = next(k for k in range(1, m + 1) if (-k) % m in squares)
    check(needed not in squares and 0 in squares, "the needed remainder must be one a square cannot leave")
    stem = rng.choice(
        (
            f"Claim: n² + {c} is never divisible by {m}, whatever integer n is. To prove it by contradiction, "
            f"suppose that {m} divides n² + {c} for some integer n.",
            f"Suppose, for contradiction, that n² + {c} is a multiple of {m} for some integer n.",
            f"A student says that no integer n makes n² + {c} divisible by {m}. "
            f"Begin a proof by contradiction by assuming that {m} does divide n² + {c} for some integer n.",
        )
    )
    a = count_part(
        f"Under this assumption, what remainder (from 0 to {m - 1}) must n² leave when divided by {m}?",
        needed,
        1,
        [f"n² + {c} ≡ 0 (mod {m}), so n² ≡ -{c} ≡ {needed} (mod {m})."],
        [
            (v, tag)
            for v, tag in (
                (c % m, "pcx-sign-error-in-congruence"),
                (c, "pcx-does-not-reduce-the-remainder"),
                (needed + 1, None),
                (needed - 1, None),
                (needed + 2, None),
                (needed - 2, None),
            )
            if v < m
        ],
        hi=m - 1,
    )
    b = count_part(
        f"How many different remainders can a perfect square leave when divided by {m}?",
        s,
        2,
        [
            f"Square n = 0, 1, ..., {m - 1} and reduce mod {m}: the remainders are {_list_text(sorted(squares))}.",
            f"That is {s} different remainders.",
        ],
        [
            (v, tag)
            for v, tag in (
                (m // 2 + 1, "pcx-assumes-half-the-residues-occur"),
                (s + 1, "pcx-miscounts-the-residues"),
                (s - 1, "pcx-miscounts-the-residues"),
                (missing, "pcx-counts-the-missing-residues"),
                (s + 2, None),
                (s - 2, None),
            )
            if v < m  # the remainder that makes the claim true is never a square remainder, so s < m
        ],
        hi=m - 1,
    )
    c_part = count_part(
        f"Using your answer to part (b), how many of the remainders from 0 to {m - 1} can a perfect square never leave "
        f"on division by {m}?",
        missing,
        1,
        [f"{m} possible remainders, {s} of which occur, so {m} - {s} = {missing} never occur."],
        [
            (v, tag)
            for v, tag in (
                (s, "pcx-counts-the-possible-not-impossible"),
                (missing + 1, None),
                (missing - 1, None),
                (missing + 2, None),
            )
            if v < m  # remainder 0 is always a square remainder, so fewer than m remainders are impossible
        ],
        hi=m - 1,
        follow=f"{m} - part_b",
    )
    d_part = count_part(
        f"What is the smallest positive integer k for which {m} could divide n² + k for some integer n?",
        k,
        2,
        [
            f"We need -k mod {m} to be a remainder a square can leave: one of {_list_text(sorted(squares))}.",
            f"The smallest k that works is {k}.",
        ],
        [
            (m, "pcx-takes-the-trivial-multiple"),
            (needed, "pcx-reuses-the-earlier-remainder"),
            (c, "pcx-reuses-the-claim-constant"),
            (k + 1, None),
            (k + 2, None),
            (k - 1, None),
        ],
    )
    if a is None or b is None or c_part is None or d_part is None:
        raise ValueError("retry")
    return dict(text=stem, parts=[a, b, c_part, d_part])


_BUILDERS = (_euclid, _linear, _residue)


@generator("exd_x_pc1_contradiction_chain", level=13, presets={"exam": dict(tier="exam")})
@uses_rng
def exd_x_pc1_contradiction_chain(rng: random.Random, tier: str = "exam") -> dict:
    """See the module docstring. Only the exam tier exists."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    family = rng.choice(_BUILDERS)  # noqa
    for _ in attempts(60, "a family draw with three usable wrong options per part"):
        try:
            return family(rng)
        except ValueError as exc:
            if str(exc) != "retry":
                raise
    raise AssertionError("unreachable")
