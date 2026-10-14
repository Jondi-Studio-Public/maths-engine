"""Generates `r4c_symmetric_group_conjugacy` questions (easy, medium, hard) and the exam chain `exd_x_r4c_symmetric_group_conjugacy`.

``uc2-symmetric-group-conjugacy`` (``abstract-algebra``, answer ``number``): two permutations of {1, ..., n} are
conjugate exactly when they have the same cycle type, so conjugacy classes, centralisers and element counts in S_n come
from partitions of n.

* easy: the number of permutations in S_n conjugate to a permutation given as a product of disjoint cycles.
* medium: the order of the centraliser of such a permutation (n! divided by the class size).
* hard: how many elements of S_n have a given order k, summing the class sizes of every cycle type whose cycle lengths
  have least common multiple k.
* exam: for a permutation σ in S_n: (a) its order, (b) the size of its class, (c) the order of its centraliser
  (follows from (b)), (d) the cycle type of σ^e.

The class of cycle type with m_i cycles of length i has n! / prod(i^m_i · m_i!) elements. The tests count by running
through all of S_n.
"""

from __future__ import annotations

from math import factorial, gcd, lcm

from ..core import attempts, check, generator
from ._r4c_kit import number_part, number_question, ordered_part

Partition = tuple[int, ...]


def partitions(n, largest=None):
    """Partitions of n as tuples of parts in descending order."""
    largest = n if largest is None else largest
    if n == 0:
        yield ()
        return
    for first in range(min(n, largest), 0, -1):
        for rest in partitions(n - first, first):
            yield (first, *rest)


def _multiplicities(part: Partition):
    out = {}
    for length in part:
        out[length] = out.get(length, 0) + 1
    return out


def class_size(part: Partition) -> int:
    n = sum(part)
    denom = 1
    for length, m in _multiplicities(part).items():
        denom *= length**m * factorial(m)
    return factorial(n) // denom


def _order(part: Partition) -> int:
    return lcm(*part)


def _draw_cycles(rng, part, n):
    """Disjoint cycles of the given lengths (lengths above 1) on a random set of points of {1, ..., n}."""
    points = list(range(1, n + 1))
    rng.shuffle(points)
    cycles, at = [], 0
    for length in part:
        if length > 1:
            cycles.append(tuple(points[at : at + length]))
        at += length
    return cycles


def _cycles_text(cycles) -> str:
    return "".join("(" + " ".join(map(str, c)) + ")" for c in cycles)


def _power_type(part: Partition, e: int) -> Partition:
    out = []
    for length in part:
        g = gcd(length, e)
        out.extend([length // g] * g)
    return tuple(sorted(out, reverse=True))


def _type_text(part) -> str:
    return "(" + ", ".join(map(str, part)) + ")"


def _pick(rng, ns, min_nontrivial=2):
    """A partition of some n in `ns` with at least `min_nontrivial` cycles of length above 1 and a repeat or fixed point."""
    for _ in attempts(1000, "a partition with several cycles"):
        n = rng.choice(ns)
        part = rng.choice([p for p in partitions(n) if sum(1 for x in p if x > 1) >= min_nontrivial and p[0] < n])
        mult = _multiplicities(part)
        if max(mult.values()) >= 2 or 1 in mult:
            return n, part
    raise AssertionError  # pragma: no cover


def _easy(rng):
    n, part = _pick(rng, [6, 7, 8])
    cycles = _draw_cycles(rng, part, n)
    key = class_size(part)
    mult = _multiplicities(part)
    prod_len = 1
    for length in part:
        prod_len *= length
    sigma = _cycles_text(cycles)
    stem = rng.choice(
        [
            f"In the symmetric group S_{n}, let σ = {sigma} (points not listed are fixed). "
            f"How many permutations in S_{n} are conjugate to σ?",
            f"Let σ = {sigma} be a permutation in S_{n}; points not written are fixed. Find the size of the conjugacy class of σ in S_{n}.",
            f"The permutation σ = {sigma} lies in S_{n} (unlisted points are fixed). "
            f"Count the permutations τ in S_{n} of the form ρσρ⁻¹ for some ρ in S_{n}.",
        ]
    )
    m_fact = 1
    for m in mult.values():
        m_fact *= factorial(m)
    return number_question(
        stem,
        key,
        [
            f"σ has cycle type {_type_text(part)}, including fixed points as cycles of length 1.",
            f"The class has {factorial(n)} / ({' · '.join(f'{i}^{m} · {m}!' for i, m in sorted(mult.items(), reverse=True))}) = {key} elements.",
        ],
        [
            (factorial(n) // prod_len, "class-size-forgets-repeated-cycle-symmetry"),
            (factorial(n) // m_fact, "class-size-forgets-cycle-rotations"),
            (factorial(n) // key, "class-size-is-centraliser-order"),
            (len({tuple(c) for c in cycles}), "class-size-counts-cycles"),
            (key * 2, None),
        ],
    )


def _medium(rng):
    n, part = _pick(rng, [6, 7, 8, 9])
    cycles = _draw_cycles(rng, part, n)
    size = class_size(part)
    key = factorial(n) // size
    mult = _multiplicities(part)
    prod_len = 1
    for length in part:
        prod_len *= length
    sigma = _cycles_text(cycles)
    stem = rng.choice(
        [
            f"In the symmetric group S_{n}, let σ = {sigma} (points not listed are fixed). "
            f"Find the order of the centraliser of σ, the group of permutations in S_{n} that commute with σ.",
            f"Let σ = {sigma} be a permutation in S_{n} (unlisted points are fixed). How many permutations in S_{n} commute with σ?",
            f"Find |C(σ)|, the number of elements of S_{n} that commute with σ = {sigma}; points not written are fixed.",
        ]
    )
    direct = 1
    for i, m in mult.items():
        direct *= i**m * factorial(m)
    check(direct == key, "the centraliser order differs from the product over cycle lengths")
    return number_question(
        stem,
        key,
        [
            f"The class of σ has {size} elements, so by orbit and stabiliser its centraliser has {factorial(n)} / {size} = {key}.",
            "Directly: each cycle of length i can be rotated in i ways, and m cycles of the same length can be permuted in m! ways.",
        ],
        [
            (size, "centraliser-order-is-class-size"),
            (prod_len, "centraliser-order-is-product-of-cycle-lengths"),
            (_order(part), "centraliser-order-is-element-order"),
            (_prod_i_m(mult), "centraliser-forgets-repeated-cycle-symmetry"),
            (key * 2, None),
        ],
    )


def _prod_i_m(mult):
    out = 1
    for i, m in mult.items():
        out *= i**m
    return out


def _hard(rng):
    for _ in attempts(500, "an order reached by several cycle types"):
        n = rng.choice([5, 6, 7, 8, 9])
        k = rng.choice(sorted({_order(p) for p in partitions(n)}))
        types = [p for p in partitions(n) if _order(p) == k]
        if len(types) >= 2 and k > 2:
            break
    sizes = [class_size(p) for p in types]
    key = sum(sizes)
    dividing = sum(class_size(p) for p in partitions(n) if k % _order(p) == 0)
    stem = rng.choice(
        [
            f"How many elements of the symmetric group S_{n} have order exactly {k}?",
            f"Count the permutations in S_{n} whose order is exactly {k}.",
            f"S_{n} is the group of all permutations of 1, ..., {n}. For how many of them is {k} the least positive "
            "power that gives the identity?",
        ]
    )
    return number_question(
        stem,
        key,
        [
            f"The order of a permutation is the lcm of its cycle lengths, so list the cycle types of {n} with lcm {k}: "
            + ", ".join(_type_text(p) for p in types)
            + ".",
            "Sum the class sizes: " + " + ".join(map(str, sizes)) + f" = {key}.",
        ],
        [
            (dividing, "element-count-order-divides-instead-of-equals"),
            (key - min(sizes), "element-count-misses-a-cycle-type"),
            (key - max(sizes), "element-count-misses-a-cycle-type"),
            (max(sizes), "element-count-single-cycle-type"),
            (len(types), "element-count-types-not-elements"),
            (key + 1, None),
        ],
    )


@generator(
    "r4c_symmetric_group_conjugacy",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r4c_symmetric_group_conjugacy(rng, tier="easy"):
    """Conjugacy classes in S_n; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r4c_symmetric_group_conjugacy", level=19, presets={"exam": dict(tier="exam")})
def exd_x_r4c_symmetric_group_conjugacy(rng, tier="exam"):
    """Exam chain: order, class size, centraliser order and cycle type of a power."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "a permutation whose power changes its cycle type"):
        n, part = _pick(rng, [7, 8])
        e = rng.choice([2, 3, 4])
        if any(gcd(length, e) > 1 for length in part) and _power_type(part, e) != part and _power_type(part, e)[0] > 1:
            break
    cycles = _draw_cycles(rng, part, n)
    size = class_size(part)
    centraliser = factorial(n) // size
    power = _power_type(part, e)
    order = _order(part)
    mult = _multiplicities(part)
    return dict(
        text=rng.choice(
            [
                f"In the symmetric group S_{n}, let σ = {_cycles_text(cycles)} (points not listed are fixed).",
                f"Let σ = {_cycles_text(cycles)} be a permutation in S_{n}; points that are not written are fixed.",
                f"The permutation σ = {_cycles_text(cycles)} belongs to S_{n}, with the points not listed fixed.",
            ]
        ),
        parts=[
            number_part(
                "Find the order of σ.",
                order,
                1,
                [f"The order is the lcm of the cycle lengths {_type_text(part)}: {order}."],
                [
                    (sum(part), "order-as-sum-of-cycle-lengths"),
                    (_product(part), "perm-order-is-product-of-cycle-lengths"),
                    (max(part), None),
                    (order + 1, None),
                ],
                sensible_range=(0, None),
            ),
            number_part(
                f"How many permutations in S_{n} are conjugate to σ?",
                size,
                3,
                [
                    f"The cycle type is {_type_text(part)}.",
                    f"The class has {factorial(n)} / ({' · '.join(f'{i}^{m} · {m}!' for i, m in sorted(mult.items(), reverse=True))}) = {size} elements.",
                ],
                [
                    (factorial(n) // _product(part), "class-size-forgets-repeated-cycle-symmetry"),
                    (centraliser, "class-size-is-centraliser-order"),
                    (size * 2, None),
                    (size + 1, None),
                ],
                sensible_range=(0, None),
            ),
            number_part(
                f"Find the order of the centraliser of σ in S_{n}.",
                centraliser,
                1,
                [f"{factorial(n)} / {size} = {centraliser}."],
                [
                    (size, "centraliser-order-is-class-size"),
                    (_product(part), "centraliser-order-is-product-of-cycle-lengths"),
                    (centraliser + 1, None),
                    (centraliser * 2, None),
                ],
                follow_through=f"{factorial(n)} / part_b",
                sensible_range=(0, None),
            ),
            ordered_part(
                f"Find the cycle type of σ^{e}: list the lengths of all its cycles, fixed points included as 1s, in descending order, as (l1, l2, ...).",
                list(power),
                3,
                [
                    f"A cycle of length L becomes gcd(L, {e}) cycles of length L / gcd(L, {e}).",
                    f"σ^{e} has cycle type {_type_text(power)}.",
                ],
                [
                    (list(part), "power-keeps-cycle-type"),
                    (
                        sorted((length // gcd(length, e) for length in part), reverse=True),
                        "power-cycle-lengths-without-splitting",
                    ),
                    (list(_power_type(part, e + 1)), "power-exponent-off-by-one"),
                    (sorted(power), None),
                ],
            ),
        ],
    )


def _product(values):
    out = 1
    for v in values:
        out *= v
    return out
