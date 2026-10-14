"""Generates `r4a_permutation_cycle_notation` questions (easy, medium, hard) and the exam chain
`exd_x_r4a_permutation_cycle_notation`.

``uc2-permutation-cycle-notation`` (``abstract-algebra``, answer ``set``): write a permutation as a product of disjoint
cycles. The answer is the set of non-trivial cycles; each cycle starts at its smallest entry and fixed points are left
out, which the stem states, so one set is the only correct reply.

* easy: a permutation of {1, ..., n} given by the list of its images (n from 6 to 8, two or three non-trivial cycles).
* medium: the product of two such permutations, each given by its images, composed right to left.
* hard: a product of three overlapping cycles written in cycle notation, composed right to left, in S_7 to S_9.
* exam: a product of four overlapping cycles in S_8: (a) its disjoint cycles, (b) its order, (c) the least positive
  r with the permutation to the power r equal to its power N (follows through from (b)), (d) the image of a point
  under the power N.

Each key is found by tracing every point through the written cycles one at a time, and checked against the permutation
rebuilt from the answer cycles and against the order found by repeated composition.
"""

from __future__ import annotations

from ..core import attempts, check, generator, phrase
from ._r4a_alg_kit import (
    with_latex,
    compose,
    cycles,
    cycles_options,
    cycles_answer,
    cycles_part,
    from_cycles,
    inverse,
    number_part,
    order,
    order_by_powers,
    power,
    product_text,
)

_CONVENTION = (
    "Give the set of its non-trivial cycles, each cycle written starting at its smallest entry with commas "
    "between entries, for example {(1, 4, 2), (3, 5)}. Leave out fixed points."
)


def _trace(cycle_list, n):
    """The permutation as a tuple, found by sending each point through the cycles right to left one at a time."""
    images = []
    for start in range(1, n + 1):
        point = start
        for cycle in reversed(cycle_list):
            if point in cycle:
                point = cycle[(cycle.index(point) + 1) % len(cycle)]
        images.append(point)
    return tuple(images)


def _random_perm(rng, n):
    items = list(range(1, n + 1))
    rng.shuffle(items)
    return tuple(items)


def _easy(rng):
    for _ in attempts(500, "a permutation with two or three non-trivial cycles"):
        n = rng.randint(6, 8)
        p = _random_perm(rng, n)
        key = cycles(p)
        if 2 <= len(key) <= 3 and len(set(map(len, key))) >= 2:
            break
    check(from_cycles(key, n) == p, "the cycles do not rebuild the permutation")
    value, text = cycles_answer(key)
    fixed = [x for x in range(1, n + 1) if p[x - 1] == x]
    start = p.index(1)
    one_cycle = tuple(p[start:] + p[:start])  # the image list read as a single cycle, rotated to start at 1
    cands = [
        (cycles(inverse(p)), "cycle-notation-direction-reversed"),
        ([*key, (fixed[0],)] if fixed else None, "cycle-fixed-point-listed"),
        ([key[0] + key[1], *key[2:]], "cycle-merged-into-one"),
        ([one_cycle], "cycle-one-line-read-as-one-cycle"),
    ]
    options = cycles_options(key, cands, 3)
    check(len(options) >= 2, "too few distractors")
    stem = (
        f"The permutation σ of {{1, ..., {n}}} sends 1, 2, ..., {n} to {', '.join(map(str, p))} respectively. "
        f"Write σ as a product of disjoint cycles. {_CONVENTION}"
    )
    return dict(
        text=stem,
        answer_kind="set",
        answer=value,
        answer_text=text,
        working=[
            "Follow each point until it returns: " + "; ".join("→".join(map(str, c + (c[0],))) for c in key) + ".",
            f"So σ = {product_text(key)}.",
        ],
        distractors=options,
    )


def _medium(rng):
    for _ in attempts(1000, "a product with two or three non-trivial cycles"):
        n = rng.randint(7, 8)
        p, q = _random_perm(rng, n), _random_perm(rng, n)
        r = compose(p, q)
        key = cycles(r)
        if 2 <= len(key) <= 3 and r not in (p, q) and compose(q, p) != r and cycles(compose(q, p)) != key:
            break
    check(from_cycles(key, n) == r, "the cycles do not rebuild the product")
    check(all(r[i] == p[q[i] - 1] for i in range(n)), "pointwise product differs")
    value, text = cycles_answer(key)
    swapped = cycles(compose(q, p))
    cands = [
        (swapped, "reverses-composition-order"),
        (cycles(inverse(r)), "cycle-notation-direction-reversed"),
        (cycles(p), "copies-one-permutation"),
        ([key[0] + key[1], *key[2:]] if len(key) >= 2 else None, "cycle-merged-into-one"),
    ]
    options = cycles_options(key, cands, 3)
    pq_images = f"p = ({', '.join(map(str, p))}) and q = ({', '.join(map(str, q))})"
    stem = rng.choice(
        [
            f"Permutations of {{1, ..., {n}}} are given by their images of 1, ..., {n}: {pq_images}. "
            f"The product pq means apply q first, then p. Write pq as a product of disjoint cycles. {_CONVENTION}",
            f"Let {pq_images} be permutations of {{1, ..., {n}}}, each listed by its images of 1, ..., {n}. "
            f"Compose them as pq (q first, then p) and express the result as disjoint cycles. {_CONVENTION}",
            f"Two permutations of {{1, ..., {n}}} are written as lists of images of 1, ..., {n}: {pq_images}. "
            f"Applying q first and then p gives the product pq. Find pq in disjoint cycle notation. {_CONVENTION}",
        ]
    )
    return dict(
        text=stem,
        answer_kind="set",
        answer=value,
        answer_text=text,
        working=[
            f"pq sends i to p(q(i)), so pq has images {', '.join(map(str, r))}.",
            f"Following each point until it returns gives pq = {product_text(key)}.",
        ],
        distractors=options,
    )


def _random_cycles(rng, n, count):
    out = []
    for _ in range(count):
        k = rng.choice([2, 3, 3, 4])
        out.append(tuple(rng.sample(range(1, n + 1), k)))
    return out


def _draw_product(rng, n_choices, count):
    for _ in attempts(3000, "overlapping cycles whose product has two or three non-trivial cycles"):
        n = rng.choice(n_choices)
        cyc = _random_cycles(rng, n, count)
        if len({x for c in cyc for x in c}) < n - 1 or any(len(set(a) & set(b)) == 0 for a, b in zip(cyc, cyc[1:])):
            continue
        r = from_cycles(cyc, n)
        left_to_right = from_cycles(list(reversed(cyc)), n)
        key = cycles(r)
        if not 2 <= len(key) <= 3 or left_to_right == r or cycles(left_to_right) == key:
            continue
        if _trace(cyc, n) != r:
            continue
        return n, cyc, r, key, left_to_right
    raise AssertionError  # pragma: no cover


_HARD_STEMS = (
    "In S_{n}, let σ = {prod}, where cycles are composed right to left (the rightmost cycle acts first). "
    "Write σ as a product of disjoint cycles. {conv}",
    "Compute the product {prod} in S_{n}, composing the cycles right to left. Write the result as a product of disjoint "
    "cycles. {conv}",
)


def _hard(rng):
    n, cyc, r, key, left_to_right = _draw_product(rng, (7, 8, 9), 3)
    value, text = cycles_answer(key)
    cands = [
        (cycles(left_to_right), "reverses-composition-order"),
        (cycles(inverse(r)), "cycle-notation-direction-reversed"),
        ([c for c in cycles(from_cycles(cyc[1:], n))], "cycle-factor-left-out"),
        ([key[0] + key[1], *key[2:]], "cycle-merged-into-one"),
    ]
    options = cycles_options(key, cands, 3)
    check(len(options) >= 2, "too few distractors")
    return dict(
        text=phrase(rng, _HARD_STEMS, n=n, prod=product_text(cyc), conv=_CONVENTION),
        answer_kind="set",
        answer=value,
        answer_text=text,
        working=[
            "Send each point through the cycles from the right: "
            + "; ".join(f"{x}→{r[x - 1]}" for x in range(1, n + 1) if r[x - 1] != x)
            + ".",
            f"Collecting the points that return gives σ = {product_text(key)}.",
        ],
        distractors=options,
    )


@generator(
    "r4a_permutation_cycle_notation",
    level=19,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
@with_latex
def r4a_permutation_cycle_notation(rng, tier="easy"):
    """Disjoint cycle notation of a permutation; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


@generator("exd_x_r4a_permutation_cycle_notation", level=19, presets={"exam": dict(tier="exam")})
@with_latex
def exd_x_r4a_permutation_cycle_notation(rng, tier="exam"):
    """Exam chain: disjoint cycles of a product of four cycles in S_8, its order, a reduced power and an image."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n = 8
    for _ in attempts(3000, "a product of cycles with order at least 4 and a safe power"):
        _n, cyc, r, key, left_to_right = _draw_product(rng, (n,), 4)
        lengths = [len(c) for c in cycles(r, fixed=True)]
        ordr = order(r)
        big = rng.randint(40, 95)
        x = rng.randint(1, n)
        if (
            ordr < 4
            or ordr in (sum(lengths), max(lengths), len(lengths), n)
            or big % ordr in (0, 1)
            or big % n in (0, 1)
        ):
            continue
        y = power(r, big)[x - 1]
        slips_c = {power(r, big % n)[x - 1], r[x - 1], power(left_to_right, big)[x - 1], power(r, (-big) % ordr)[x - 1]}
        if y in slips_c or len(slips_c) < 3:
            continue
        break
    check(ordr == order_by_powers(r), "order differs from repeated composition")
    check(power(r, big) == power(r, big % ordr), "the reduced power differs")
    check(from_cycles(key, n) == r, "the cycles do not rebuild the permutation")
    check(_trace(cyc, n) == r, "tracing differs from composition")
    red = big % ordr
    key_text = product_text(key)
    cycle_lengths = ", ".join(str(len(c)) for c in cycles(r, fixed=True))
    return dict(
        text=(
            f"In S_{n}, let σ = {product_text(cyc)}, where cycles are composed right to left "
            f"(the rightmost cycle acts first)."
        ),
        parts=[
            cycles_part(
                f"Write σ as a product of disjoint cycles. {_CONVENTION}",
                key,
                3,
                [f"Tracing each point through the four cycles from the right gives σ = {key_text}."],
                [
                    (cycles(left_to_right), "reverses-composition-order"),
                    (cycles(inverse(r)), "cycle-notation-direction-reversed"),
                    ([key[0] + key[1], *key[2:]], "cycle-merged-into-one"),
                    ([c for c in cycles(from_cycles(cyc[1:], n))], "cycle-factor-left-out"),
                ],
            ),
            number_part(
                "Find the order of σ.",
                ordr,
                2,
                [
                    f"The disjoint cycles of σ, with fixed points, have lengths {cycle_lengths}.",
                    f"The order is their least common multiple, {ordr}.",
                ],
                [
                    (sum(lengths), "order-as-sum-of-cycle-lengths"),
                    (max(lengths), "order-as-longest-cycle"),
                    (len(lengths), "order-as-number-of-cycles"),
                    (n, None),
                ],
                sensible_range=(0, None),
            ),
            number_part(
                f"Find the least positive integer r such that σ^r = σ^{big}.",
                red,
                2,
                [f"σ^{ordr} is the identity and {big} = {big // ordr}×{ordr} + {red}, so σ^{big} = σ^{red}."],
                [(big % n, "exponent-reduced-by-wrong-modulus"), (big, None), (ordr - red, None), (big // ordr, None)],
                follow_through=f"{big} - part_b*floor({big}/part_b)",
                sensible_range=(0, None),
            ),
            number_part(
                f"Find the image of {x} under σ^{big}.",
                y,
                2,
                [f"σ^{big} = σ^{red}, and applying σ a total of {red} times sends {x} to {y}."],
                [
                    (power(r, big % n)[x - 1], "exponent-reduced-by-wrong-modulus"),
                    (r[x - 1], None),
                    (power(left_to_right, big)[x - 1], "reverses-composition-order"),
                    (power(r, (-big) % ordr)[x - 1], None),
                ],
                sensible_range=(0, n + 1),
            ),
        ],
    )
