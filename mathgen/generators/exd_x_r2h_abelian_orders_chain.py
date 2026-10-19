"""Exam question on the element orders of a product of two cyclic groups, used to tell it from a cyclic group.

The stem gives G = Zm x Zn with gcd(m, n) > 1, so G is not cyclic. Four parts:

(a) the largest order of an element of G, lcm(m, n)                           number (whole), 1 mark
(b) the number of elements of G of order exactly d (d > 1 divides lcm(m, n))  number (whole), 2 marks
(c) the number of cyclic subgroups of G of order d, (b) / phi(d)              number (whole), 1 mark, follow_through part_b / phi(d)
(d) how many more elements of order d G has than the cyclic group Z(mn)      number (whole), 2 marks, follow_through part_b - phi(d)

The count in (b) is taken by listing every element (a, b) of G and computing its order; the closed form
sum over e | d of mu(d/e) gcd(e, m) gcd(e, n) must agree. A cyclic group of order mn has exactly phi(d) elements of
order d for each d dividing mn, which is what makes (d) positive: it is the invariant that separates G from Z(mn).
"""

from __future__ import annotations

import math
import random

from ..core import attempts, check, generator, phrase
from .exd_x_r2h_support import build_part

_ORDERS = tuple(range(2, 25))

_STEM = (
    "Let G = Z{m} x Z{n}, the direct product of the cyclic groups of orders {m} and {n} under componentwise addition.",
    "Consider the group G = Z{m} x Z{n}, formed from cyclic groups of orders {m} and {n} with componentwise addition.",
    "G is the group Z{m} x Z{n}: pairs (a, b) with a added modulo {m} and b added modulo {n}.",
)
_PROMPT_A = (
    "Find the largest order of an element of G.",
    "What is the largest order that an element of G can have?",
)
_PROMPT_B = (
    "How many elements of G have order exactly {d}?",
    "Find the number of elements of G of order {d}.",
)
_PROMPT_C = (
    "Hence find the number of cyclic subgroups of G of order {d}.",
    "Each cyclic subgroup of order {d} has phi({d}) generators. Use this to find how many cyclic subgroups of "
    "order {d} G has.",
)
_PROMPT_D = (
    "The cyclic group Z{mn} has phi({d}) elements of order {d}. How many more elements of order {d} does G have "
    "than Z{mn}?",
    "Z{mn} is cyclic, so it has exactly phi({d}) elements of order {d}. By how many does the number of elements of "
    "order {d} in G exceed this?",
)


def _order(a: int, b: int, m: int, n: int) -> int:
    """The order of (a, b) in Zm x Zn: the lcm of the orders of a in Zm and of b in Zn."""
    return math.lcm(m // math.gcd(a, m), n // math.gcd(b, n))


def _is_prime(x: int) -> bool:
    return x > 1 and all(x % p for p in range(2, int(x**0.5) + 1))


def _phi(d: int) -> int:
    return sum(1 for k in range(1, d + 1) if math.gcd(k, d) == 1)


def _mobius(d: int) -> int:
    result, p = 1, 2
    while d > 1:
        if d % p == 0:
            d //= p
            if d % p == 0:
                return 0
            result = -result
        p += 1
    return result


def _build(rng: random.Random) -> dict:
    for _ in attempts(2000, "a non-cyclic product with a composite order to count"):
        m, n = rng.choice(_ORDERS), rng.choice(_ORDERS)
        g = math.gcd(m, n)
        if g == 1 or m == n:
            continue
        top = math.lcm(m, n)
        composites = [x for x in range(4, top + 1) if top % x == 0 and not _is_prime(x)]
        if not composites:
            continue
        d = rng.choice(composites)
        count = sum(1 for a in range(m) for b in range(n) if _order(a, b, m, n) == d)
        if count >= 2 * _phi(d):
            break
    # Second route: count by "d x = 0" and Mobius inversion.
    closed = sum(_mobius(d // e) * math.gcd(e, m) * math.gcd(e, n) for e in range(1, d + 1) if d % e == 0)
    check(closed == count, "element count disagrees with the Mobius sum")
    check(max(_order(a, b, m, n) for a in range(m) for b in range(n)) == top, "largest order disagrees with lcm")
    phi_d = _phi(d)
    check(count % phi_d == 0, "elements of order d do not split into cyclic subgroups")
    subgroups = count // phi_d
    diff = count - phi_d
    check(diff > 0, "G has no more elements of this order than the cyclic group")

    dividing = math.gcd(d, m) * math.gcd(d, n)
    count_m = sum(1 for a in range(m) if math.gcd(a, m) and m // math.gcd(a, m) == d) if m % d == 0 else 0
    count_n = sum(1 for a in range(n) if n // math.gcd(a, n) == d) if n % d == 0 else 0
    a_wrong = [
        (m * n, "order-of-group-for-exponent"),
        (max(m, n), "max-of-component-orders"),
        (math.gcd(m, n), "gcd-for-lcm"),
        (m + n, "adds-orders"),
    ]
    b_wrong = [
        (dividing, "counts-order-dividing-d"),
        (phi_d, "counts-one-cyclic-subgroup"),
        (count_m + count_n, "adds-component-counts"),
        (dividing - 1, "counts-order-dividing-d"),
        (count + phi_d, "counts-one-cyclic-subgroup"),
    ]
    c_wrong = [
        (count, "counts-elements-as-subgroups"),
        (count * phi_d, "multiplies-instead-of-dividing"),
        (subgroups + 1, "counts-extra-trivial-subgroup"),
        (subgroups - 1 if subgroups > 1 else None, "off-by-one-count"),
    ]
    d_wrong = [
        (count, "does-not-subtract-cyclic-count"),
        (count - (d - 1), "takes-d-minus-1-as-phi"),
        (count - 1, "assumes-unique-element-of-order"),
        (diff + phi_d, "does-not-subtract-cyclic-count"),
        (subgroups - 1, "counts-subgroups-not-elements"),
        (count - subgroups, "subtracts-subgroup-count"),
    ]
    # A wrong count of elements or subgroups below one is not a plausible answer.
    b_wrong, c_wrong, d_wrong = (
        [(v, t) for v, t in w if v is not None and v >= 1] for w in (b_wrong, c_wrong, d_wrong)
    )
    stem = phrase(rng, _STEM, m=m, n=n)
    parts = [
        build_part(
            phrase(rng, _PROMPT_A),
            top,
            kind="number",
            marks=1,
            working=[f"The order of (a, b) is lcm(order of a, order of b), at most lcm({m}, {n}) = {top}."],
            wrong=a_wrong,
        ),
        build_part(
            phrase(rng, _PROMPT_B, d=d),
            count,
            kind="number",
            marks=2,
            working=[
                f"Elements with {d}x = 0 number gcd({d}, {m}) × gcd({d}, {n}) = {dividing}; "
                "subtract those of smaller order by inclusion-exclusion over the divisors of "
                f"{d}.",
                f"This leaves {count} elements of order exactly {d}.",
            ],
            wrong=b_wrong,
        ),
        build_part(
            phrase(rng, _PROMPT_C, d=d),
            subgroups,
            kind="number",
            marks=1,
            working=[f"phi({d}) = {phi_d}, so {count} / {phi_d} = {subgroups} cyclic subgroups."],
            wrong=c_wrong,
            follow_through=f"part_b/{phi_d}",
        ),
        build_part(
            phrase(rng, _PROMPT_D, d=d, mn=m * n),
            diff,
            kind="number",
            marks=2,
            working=[f"{count} - phi({d}) = {count} - {phi_d} = {diff}. A cyclic group would have none extra."],
            wrong=d_wrong,
            follow_through=f"part_b - {phi_d}",
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r2h_abelian_orders_chain", level=19, presets={"exam": dict(tier="exam")})
def exd_x_r2h_abelian_orders_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Largest element order, the number of elements of a given order, cyclic subgroups, and the gap to a cyclic group."""
    check(tier == "exam", "exd_x_r2h_abelian_orders_chain: only the exam tier exists")
    return _build(rng)
