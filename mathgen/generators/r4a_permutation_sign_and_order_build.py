"""Generates `r4a_permutation_sign_and_order` questions (easy, medium, hard) and the exam chain
`exd_x_r4a_permutation_sign_and_order`.

``uc2-permutation-sign-and-order`` (``abstract-algebra``, answer ``number``): the order of a permutation, its parity and
the order of its powers.

* easy: the order of a permutation of {1, ..., n} (n = 8 or 9) given by its list of images, so the cycles must be found
  first and then combined by the least common multiple.
* medium: how many of five permutations, each written as a product of disjoint cycles, are even.
* hard: the order of a high power σ^k of a permutation given as a product of two overlapping cycles, which needs the
  disjoint cycles of the product, the order, and what a power does to each cycle.
* exam: a permutation of {1, ..., 9} given by its images: (a) its order, (b) the least number of transpositions whose
  product it is, (c) the order of σ^k, (d) how many of σ, σ², ..., σ^m (m the order) are even, which follows through
  from (a) and (b).

Orders are found by cycle lengths and checked by repeated composition; parity by cycle count and checked by counting
inversions.
"""

from __future__ import annotations

from math import gcd, lcm, prod

from ..core import attempts, check, generator, phrase
from ._r4a_alg_kit import (
    with_latex,
    compose,
    cycles,
    from_cycles,
    number_part,
    number_question,
    order,
    order_by_powers,
    power,
    product_text,
    sign,
    sign_by_inversions,
)


def _random_perm(rng, n):
    items = list(range(1, n + 1))
    rng.shuffle(items)
    return tuple(items)


def _lengths(p):
    return [len(c) for c in cycles(p)]


def _perm_of_type(rng, n, lens):
    """A permutation of {1, ..., n} whose non-trivial cycles have the lengths `lens`, on random points."""
    points = rng.sample(range(1, n + 1), sum(lens))
    cyc, at = [], 0
    for length in lens:
        cyc.append(tuple(points[at : at + length]))
        at += length
    return from_cycles(cyc, n)


def _easy(rng):
    for _ in attempts(3000, "a cycle type whose order is not an obvious sum, maximum or product"):
        n = rng.choice([9, 10, 11, 12])
        lens = [rng.randint(2, 6) for _ in range(rng.randint(2, 4))]
        o = lcm(*lens)
        if sum(lens) <= n and o < prod(lens) and 4 <= o <= 60 and o not in (sum(lens), max(lens), n):
            break
    p = _perm_of_type(rng, n, lens)
    lens = _lengths(p)
    full = [len(c) for c in cycles(p, fixed=True)]
    check(order_by_powers(p) == o == order(p), "order differs from repeated composition")
    key = cycles(p)
    slips = [
        (sum(lens), "order-as-sum-of-cycle-lengths"),
        (max(lens), "order-as-longest-cycle"),
        (prod(lens), "order-as-product-of-cycle-lengths"),
        (len(full), "order-as-number-of-cycles"),
    ]
    stem = (
        f"The permutation σ of {{1, ..., {n}}} sends 1, 2, ..., {n} to {', '.join(map(str, p))} respectively. "
        "Find the order of σ in the symmetric group."
    )
    return number_question(
        stem,
        o,
        [
            f"σ = {product_text(key)} in disjoint cycles, of lengths {', '.join(map(str, lens))}.",
            f"The order is the least common multiple of the cycle lengths, {o}.",
        ],
        slips,
        lo=1,
    )


def _wrong_sign_by_cycle_count(p):
    return -1 if len(cycles(p, fixed=True)) % 2 else 1


def _wrong_sign_by_length_parity(p):
    s = 1
    for c in cycles(p):
        s *= 1 if len(c) % 2 == 0 else -1
    return s


def _random_cycle_type(rng, n):
    """Disjoint cycle lengths (each at least 2) fitting in n points, one to three cycles."""
    for _ in attempts(200, "a cycle type"):
        lens = [rng.choice([2, 2, 3, 3, 4, 5]) for _ in range(rng.randint(1, 3))]
        if sum(lens) <= n:
            return lens
    raise AssertionError  # pragma: no cover


def _medium(rng):
    for _ in attempts(2000, "five permutations with two to four even"):
        n = rng.choice([8, 9, 10])
        perms = []
        for _ in range(5):
            lens = _random_cycle_type(rng, n)
            points = rng.sample(range(1, n + 1), sum(lens))
            cyc, at = [], 0
            for length in lens:
                cyc.append(tuple(points[at : at + length]))
                at += length
            perms.append(cyc)
        if len({product_text(c) for c in perms}) < 5:
            continue
        built = [from_cycles(c, n) for c in perms]
        evens = sum(1 for b in built if sign(b) == 1)
        by_cycle_count = sum(1 for b in built if _wrong_sign_by_cycle_count(b) == 1)
        by_length = sum(1 for b in built if _wrong_sign_by_length_parity(b) == 1)
        if 1 <= evens <= 4 and len({evens, by_cycle_count, by_length, 5 - evens}) >= 3:
            break
    check(all(sign(b) == sign_by_inversions(b) for b in built), "parity differs from the inversion count")
    slips = [
        (5 - evens, "even-and-odd-counts-swapped"),
        (by_length, "sign-from-cycle-length-parity"),
        (by_cycle_count, "sign-from-cycle-count"),
    ]
    listing = "; ".join(f"({chr(97 + i)}) {product_text(c)}" for i, c in enumerate(perms))
    stem = (
        f"Five permutations in S_{n} are written as products of disjoint cycles: {listing}. "
        "How many of them are even permutations?"
    )
    steps = []
    for i, (c, b) in enumerate(zip(perms, built)):
        t = n - len(cycles(b, fixed=True))
        steps.append(
            f"({chr(97 + i)}) needs {t} transposition{'s' if t != 1 else ''}, so it is {'even' if t % 2 == 0 else 'odd'}"
        )
    return number_question(
        stem,
        evens,
        [
            "A cycle of length m is a product of one fewer than m transpositions. " + "; ".join(steps) + ".",
            f"{evens} are even.",
        ],
        slips,
        lo=0,
        hi=5,
    )


_HARD_STEMS = (
    "The permutation σ of {{1, ..., {n}}} sends 1, 2, ..., {n} to {images} respectively. Find the order of σ^{k}.",
    "Let σ be the permutation of {{1, ..., {n}}} with images {images} of 1, 2, ..., {n} in turn. "
    "What is the order of the permutation σ^{k}?",
)


def _hard(rng):
    for _ in attempts(5000, "a cycle type whose power has a smaller, non-obvious order"):
        n = rng.choice([11, 12, 13])
        lens = [rng.randint(3, 9) for _ in range(rng.randint(2, 3))]
        if sum(lens) > n or len(set(lens)) < 2:
            continue
        o = lcm(*lens)
        k = rng.randint(4, 60)
        g = gcd(o, k)
        target = o // g
        by_each = [length // gcd(length, k) for length in lens]
        wrong = {o, max(by_each), sum(by_each)} | ({o // k} if o % k == 0 else set())
        if o < 8 or g < 2 or target in (o, 1) or target in wrong or len(wrong) < 3 or target != lcm(*by_each):
            continue
        break
    r = _perm_of_type(rng, n, lens)
    check(sorted(_lengths(r)) == sorted(lens), "cycle type differs from the draw")
    check(order_by_powers(power(r, k)) == target, "order of the power differs from repeated composition")
    slip_list = [
        (o, "power-order-ignores-power"),
        (o // k if o % k == 0 else None, "power-order-divided-by-exponent"),
        (max(by_each), "order-as-longest-cycle"),
        (sum(by_each), "order-as-sum-of-cycle-lengths"),
    ]
    return number_question(
        phrase(rng, _HARD_STEMS, n=n, images=", ".join(map(str, r)), k=k),
        target,
        [
            f"The disjoint cycles of σ have lengths {', '.join(map(str, sorted(lens)))}, so σ has order {o}.",
            f"σ^{k} has order {o}/gcd({o}, {k}) = {o}/{g} = {target}.",
        ],
        slip_list,
        lo=1,
    )


@generator(
    "r4a_permutation_sign_and_order",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@with_latex
def r4a_permutation_sign_and_order(rng, tier="easy"):
    """Order and parity of permutations; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r4a_permutation_sign_and_order", level=19, presets={"exam": dict(tier="exam")})
@with_latex
def exd_x_r4a_permutation_sign_and_order(rng, tier="exam"):
    """Exam chain: order, transposition count, order of a power and the number of even powers of a permutation."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n = 9
    for _ in attempts(5000, "a permutation with a clean chain of answers"):
        p = _random_perm(rng, n)
        lens = _lengths(p)
        full = [len(c) for c in cycles(p, fixed=True)]
        o = lcm(*lens) if lens else 1
        k = rng.randint(4, 30)
        g = gcd(o, k)
        t = n - len(full)
        if len(lens) < 2 or not 6 <= o <= 60 or g < 2 or o // g in (1, o) or o in (sum(lens), max(lens), len(full)):
            continue
        if t in (sum(lens), len(lens), len(full)) or t < 3:
            continue
        break
    check(order_by_powers(p) == o, "order differs from repeated composition")
    check(order_by_powers(power(p, k)) == o // g, "order of the power differs from repeated composition")
    check((-1) ** t == sign_by_inversions(p), "transposition parity differs from the inversion count")
    many = 2 * rng.randint(15, 40)
    evens = sum(1 for j in range(1, many + 1) if sign_by_inversions(power(p, j)) == 1)
    check(evens == many * (3 + (-1) ** t) // 4, "count of even powers differs from the closed form")
    by_each = [length // gcd(length, k) for length in lens]
    key = cycles(p)
    return dict(
        text=rng.choice(
            [
                f"The permutation σ of {{1, ..., {n}}} sends 1, 2, ..., {n} to {', '.join(map(str, p))} respectively.",
                f"Let σ be the permutation of {{1, ..., {n}}} that takes 1, 2, ..., {n} to {', '.join(map(str, p))} respectively.",
                f"A permutation σ of {{1, ..., {n}}} maps 1, 2, ..., {n} to {', '.join(map(str, p))} respectively.",
            ]
        ),
        parts=[
            number_part(
                "Find the order of σ.",
                o,
                2,
                [
                    f"σ = {product_text(key)}, with cycle lengths {', '.join(map(str, lens))} and "
                    f"{n - sum(lens)} fixed point{'s' if n - sum(lens) != 1 else ''}.",
                    f"The order is the least common multiple, {o}.",
                ],
                [
                    (sum(lens), "order-as-sum-of-cycle-lengths"),
                    (max(lens), "order-as-longest-cycle"),
                    (prod(lens), "order-as-product-of-cycle-lengths"),
                    (len(full), "order-as-number-of-cycles"),
                ],
                sensible_range=(0, None),
            ),
            number_part(
                "Find the least number of transpositions whose product is σ.",
                t,
                2,
                [f"A cycle of length m is a product of one fewer than m transpositions, so σ needs {t}."],
                [
                    (sum(lens), "transposition-count-moved-points"),
                    (len(lens), "transposition-count-one-per-cycle"),
                    (len(full), "order-as-number-of-cycles"),
                    (t + 1, None),
                ],
                sensible_range=(0, None),
            ),
            number_part(
                f"Find the order of σ^{k}.",
                o // g,
                2,
                [f"σ^{k} has order {o}/gcd({o}, {k}) = {o}/{g} = {o // g}."],
                [
                    (o, "power-order-ignores-power"),
                    (o // k if o % k == 0 else None, "power-order-divided-by-exponent"),
                    (max(by_each), "order-as-longest-cycle"),
                    (sum(by_each), "order-as-sum-of-cycle-lengths"),
                    (g, None),
                    (o // g + 1, None),
                ],
                sensible_range=(0, None),
            ),
            number_part(
                f"Among σ, σ², ..., σ^{many}, how many are even permutations?",
                evens,
                3,
                [
                    f"σ is {'even' if t % 2 == 0 else 'odd'}, so σ^j is even {'for every j' if t % 2 == 0 else 'exactly when j is even'}.",
                    f"That gives {evens} even permutations among the {many} powers.",
                ],
                [
                    (many // 2 if t % 2 == 0 else many, "even-powers-parity-misjudged"),
                    (evens - 1, "even-count-off-by-one"),
                    (evens + 1, "even-count-off-by-one"),
                    (many - 1, None),
                ],
                follow_through=f"{many}*(3 + (-1)**part_b)/4",
                sensible_range=(0, None),
            ),
        ],
    )
