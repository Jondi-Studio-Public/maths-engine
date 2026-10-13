"""Generates `r4c_units_and_zero_divisors_zn` questions (easy, medium, hard) and the exam chain
`exd_x_r4c_units_and_zero_divisors_zn`.

``uc2-units-and-zero-divisors-zn`` (``abstract-algebra``): the structure of the ring Z_n. In Z_n an element is a unit when
it is coprime to n, a zero divisor when it is non-zero and shares a factor with n, and nilpotent when every prime of n
divides it. A zero divisor is a non-zero a with ab = 0 for some non-zero b; the questions say so.

* easy (set): every nilpotent element a of Z_n (a^k = 0 for some k), which is every multiple of the product of the
  distinct primes of n.
* medium (set): every solution x of ax = b (mod n) when gcd(a, n) > 1, as a set of residues.
* hard (set): every idempotent e (e·e = e) of Z_n, as a set of residues. There are 2 to the number of distinct primes.
* exam (Z_n): (a) the number of units, (b) the number of non-zero zero divisors (follows from (a)), (c) the number of
  nilpotent elements, (d) the idempotents.

The tests find every answer by running through all residues.
"""

from __future__ import annotations

from math import gcd

from ..core import attempts, check, generator
from ._r4c_kit import factorise, number_part, radical, set_part, set_question, totient


def _easy(rng):
    n = rng.choice([n for n in range(8, 130) if 2 <= n // radical(n) <= 14 and radical(n) > 1 and n % radical(n) == 0])
    rad = radical(n)
    key = [a for a in range(n) if any(pow(a, k, n) == 0 for k in range(1, 12))]
    check(key == list(range(0, n, rad)), "the nilpotents are not the multiples of the radical")
    smallest = min(p for p, _ in factorise(n))
    stem = rng.choice(
        [
            f"An element a of the ring Z_{n} is nilpotent if a^k = 0 for some positive integer k. "
            f"Find all nilpotent elements of Z_{n}, as a set of residues from 0 to {n - 1}, for example the set with elements 0 and 5.",
            f"List every a from 0 to {n - 1} for which some power of a is 0 in the ring Z_{n}. "
            "Give the answer as a set, for example the set with elements 0 and 5.",
            f"In the ring Z_{n}, which elements are nilpotent, meaning a^k ≡ 0 (mod {n}) for some k ≥ 1? "
            "Give them as a set of residues, for example the set with elements 0 and 5.",
        ]
    )
    return set_question(
        stem,
        key,
        [
            f"a^k is divisible by {n} for some k exactly when every prime dividing {n} divides a.",
            f"The primes dividing {n} multiply to {rad}, so the nilpotents are the multiples of {rad}: {', '.join(map(str, key))}.",
        ],
        [
            ([0], "nilpotents-only-zero"),
            (list(range(0, n, smallest)) if n // smallest <= 16 else None, "nilpotents-multiples-of-one-prime"),
            (
                list(range(0, n, n // rad)) if rad <= 16 else None,
                "nilpotents-confused-with-multiples-of-n-over-radical",
            ),
            ([a for a in range(n) if (a * a) % n == 0], "nilpotents-square-zero-only"),
            (key[1:], "nilpotents-leave-out-zero"),
            (key[:-1], None),
        ],
    )


def _solutions(a, b, n):
    return [x for x in range(n) if (a * x - b) % n == 0]


def _medium(rng):
    for _ in attempts(1000, "a congruence ax = b (mod n) with gcd(a, n) between 2 and n/3 that is solvable"):
        n = rng.choice([12, 15, 16, 18, 20, 21, 24, 25, 27, 28, 30, 32, 33, 35, 36, 40, 42, 45, 48, 50, 54, 56, 60])
        a = rng.randrange(2, n)
        g = gcd(a, n)
        if not 2 <= g <= n // 3:
            continue
        b = g * rng.randrange(1, n // g)
        if b % n == 0:
            continue
        break
    key = _solutions(a, b, n)
    check(len(key) == g, "the number of solutions differs from gcd(a, n)")
    step = n // g
    x0 = key[0]
    stem = rng.choice(
        [
            f"Find every solution x from 0 to {n - 1} of the congruence {a}x ≡ {b} (mod {n}). "
            "Give the answer as a set of residues, for example the set with elements 2 and 9.",
            f"Which residues x modulo {n} satisfy {a}x ≡ {b} (mod {n})? Give all of them as a set, for example the set with elements 2 and 9.",
            f"Solve {a}x = {b} in the ring Z_{n}. List every solution x as a residue from 0 to {n - 1}, in a set such as the one with elements 2 and 9.",
        ]
    )
    return set_question(
        stem,
        key,
        [
            f"gcd({a}, {n}) = {g} divides {b}, so there are {g} solutions, spaced {n} / {g} = {step} apart.",
            f"Dividing by {g}: {a // g}x ≡ {b // g} (mod {step}) has the solution x ≡ {x0} (mod {step}).",
            f"The solutions are {', '.join(map(str, key))}.",
        ],
        [
            ([x0], "congruence-single-solution-only"),
            (sorted((x0 + k * g) % n for k in range(step)) if step <= 8 else None, "congruence-spacing-uses-the-gcd"),
            (sorted((x0 + k * step) % n for k in range(g - 1)) if g > 2 else None, "congruence-misses-a-solution"),
            (_solutions(a, b // g, n) or None, "congruence-divides-only-the-right-side"),
            (sorted((x0 + 1 + k * step) % n for k in range(g)), None),
        ],
    )


_HARD_N = [
    30,
    36,
    42,
    45,
    48,
    50,
    60,
    63,
    66,
    70,
    72,
    75,
    78,
    84,
    90,
    98,
    99,
    100,
    102,
    105,
    108,
    110,
    112,
    117,
    120,
    126,
    130,
    132,
    135,
    140,
    147,
    150,
    156,
    160,
    168,
    175,
    180,
    198,
    200,
    210,
]


def _idempotents(n):
    return [e for e in range(n) if (e * e - e) % n == 0]


def _hard(rng):
    n = rng.choice(_HARD_N)
    key = _idempotents(n)
    primes = [p for p in range(2, n + 1) if n % p == 0 and all(p % q for q in range(2, p))]
    check(len(key) == 2 ** len(primes), "the idempotents do not number 2 to the number of primes")
    stem = rng.choice(
        [
            f"An element e of the ring Z_{n} is idempotent if e·e = e. Find all idempotents of Z_{n}, "
            f"as a set of residues from 0 to {n - 1}, for example the set with elements 0 and 1.",
            f"List every e from 0 to {n - 1} with e² ≡ e (mod {n}). Give the answer as a set, for example the set with elements 0 and 1.",
            f"Which elements of the ring Z_{n} equal their own square? Give them as a set of residues, for example the set with elements 0 and 1.",
        ]
    )
    nilpotent = [a for a in range(n) if pow(a, n, n) == 0]
    roots_of_one = [x for x in range(n) if (x * x - 1) % n == 0]
    return set_question(
        stem,
        key,
        [
            f"By the Chinese remainder theorem Z_{n} splits into one factor for each prime power dividing {n}; e is idempotent exactly when it is 0 or 1 in each factor.",
            f"That gives 2^{len(primes)} = {len(key)} idempotents: {', '.join(map(str, key))}.",
        ],
        [
            ([0, 1], "idempotents-only-the-trivial-ones"),
            (roots_of_one, "idempotents-confused-with-square-roots-of-one"),
            (nilpotent if len(nilpotent) <= 10 else None, "idempotents-confused-with-nilpotents"),
            (key[:-1], "idempotents-misses-one"),
            (sorted(set(key) | {n - 1}), None),
        ],
    )


@generator(
    "r4c_units_and_zero_divisors_zn",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4c_units_and_zero_divisors_zn(rng, tier="easy"):
    """Units, zero divisors and idempotents of Z_n; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r4c_units_and_zero_divisors_zn", level=19, presets={"exam": dict(tier="exam")})
def exd_x_r4c_units_and_zero_divisors_zn(rng, tier="exam"):
    """Exam chain: units, zero divisors, nilpotents and idempotents of Z_n."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n = rng.choice(
        [12, 18, 20, 24, 28, 36, 40, 44, 45, 48, 50, 52, 54, 60, 63, 72, 75, 76, 80, 84, 90, 98, 99, 100, 108, 117, 120]
    )
    units = totient(n)
    zero_divisors = n - 1 - units
    nilpotents = n // radical(n)
    idempotents = _idempotents(n)
    check(
        nilpotents == sum(1 for a in range(n) if pow(a, n, n) == 0),
        "the nilpotent count differs from the direct search",
    )
    check(units == sum(1 for a in range(n) if gcd(a, n) == 1), "the unit count differs from the direct search")
    return dict(
        text=rng.choice(
            [
                f"Consider the ring Z_{n}. A non-zero element a is a zero divisor if ab = 0 for some non-zero b, "
                "an element is nilpotent if some power of it is 0, and an element e is idempotent if e·e = e.",
                f"In the ring Z_{n}, an element is nilpotent if some power of it is 0 and idempotent if it equals its own square; "
                "a zero divisor is a non-zero a with ab = 0 for some non-zero b.",
                f"Let R = Z_{n}. Recall: a non-zero a is a zero divisor if ab = 0 for some non-zero b; "
                "a is nilpotent if a power of it is 0; e is idempotent if e·e = e.",
            ]
        ),
        parts=[
            number_part(
                f"How many units does Z_{n} have?",
                units,
                2,
                [f"The units are the residues coprime to {n}; their number is φ({n}) = {units}."],
                [
                    (n - units, "units-counted-as-non-units"),
                    (n // 2 if n % 2 == 0 else n - 1, "units-only-the-odd-residues"),
                    (units + 1, None),
                    (units - 1, None),
                ],
                sensible_range=(0, n),
            ),
            number_part(
                f"How many zero divisors does Z_{n} have?",
                zero_divisors,
                1,
                [f"Every non-zero element is a unit or a zero divisor: {n} - 1 - {units} = {zero_divisors}."],
                [
                    (n - units, "zero-divisors-include-zero"),
                    (units, "zero-divisors-counted-as-units"),
                    (zero_divisors + 1, None),
                    (n - 1, "zero-divisors-every-nonzero-element"),
                ],
                follow_through=f"{n} - 1 - part_a",
                sensible_range=(0, n),
            ),
            number_part(
                f"How many nilpotent elements does Z_{n} have, counting 0?",
                nilpotents,
                2,
                [
                    f"a is nilpotent exactly when every prime dividing {n} divides a, that is, when {radical(n)} divides a. The multiples of {radical(n)} number {n} / {radical(n)} = {nilpotents}."
                ],
                [
                    (1, "nilpotents-only-zero"),
                    (radical(n), "nilpotents-counted-by-radical"),
                    (n - units, "nilpotents-counted-as-non-units"),
                    (nilpotents + 1, None),
                    (nilpotents * 2, None),
                ],
                sensible_range=(0, n),
            ),
            set_part(
                f"Find all idempotents of Z_{n}, as a set of residues from 0 to {n - 1}, for example the set with elements 0 and 1.",
                idempotents,
                3,
                [
                    f"Z_{n} splits by the Chinese remainder theorem into one factor for each prime power dividing {n}, and e is idempotent exactly when it is 0 or 1 in each factor.",
                    f"The idempotents are {', '.join(map(str, idempotents))}.",
                ],
                [
                    ([0, 1], "idempotents-only-the-trivial-ones"),
                    ([x for x in range(n) if (x * x - 1) % n == 0], "idempotents-confused-with-square-roots-of-one"),
                    ([a for a in range(n) if pow(a, n, n) == 0], "idempotents-confused-with-nilpotents"),
                    (idempotents[:-1], "idempotents-misses-one"),
                ],
            ),
        ],
    )
