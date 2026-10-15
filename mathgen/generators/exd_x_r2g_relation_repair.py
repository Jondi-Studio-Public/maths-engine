"""Exam question: repairing a relation into a function, and counting the functions it allows
(`ucv1-definition-quantifier-check`).

A relation R from a finite set A to a finite set B is listed as pairs. It is a function exactly when every element
of A has exactly one image, so a real exam checks the quantifiers by counting the failures. The parts chain:

* (a) the number of elements of A with no image (the "there exists" fails);
* (b) the number of pairs that must be deleted so no element has two images (the "unique" fails);
* (c) the fewest pairs to add or delete to make R a function A -> B, (a) + (b) (follow-through from (a) and (b));
* (d) the number of functions A -> B that keep R's value wherever R gives one image, pick one of R's images
  where it gives several, and are free where it gives none: |B|^(a) times the product of the image counts
  (follow-through from (a)).
"""

from __future__ import annotations

import random
from math import prod

from ..core import check, generator
from .exd_x_r2g_support import build_part, latex_with_sets, redraws

_LETTERS = "bcdefghjkmnpqrstuvwxyz"
_KS = [2, 3, 4]

_INTRO = [
    "Let A = {A} and B = {B}, and let R = {R} be a relation from A to B.",
    "A relation R from A = {A} to B = {B} is given by R = {R}.",
    "The relation R = {R} is defined from the set A = {A} to the set B = {B}.",
]


def _draw(rng: random.Random) -> dict:
    for _ in range(1000):
        free = rng.randint(1, 3)
        multi = [rng.choice(_KS) for _ in range(rng.randint(1, 3))]
        low = max(1, 5 - free - len(multi))
        singles = rng.randint(low, low + 1)
        size_b = rng.randint(max(3, max(multi) + 1), 6)
        a_set = sorted(rng.sample(range(1, 16), free + len(multi) + singles))
        b_set = sorted(rng.sample(_LETTERS, size_b))
        owners = a_set[:]
        rng.shuffle(owners)
        no_image = owners[:free]
        multi_owners = owners[free : free + len(multi)]
        single_owners = owners[free + len(multi) :]
        pairs = [(x, rng.choice(b_set)) for x in single_owners]
        for x, k in zip(multi_owners, multi):
            pairs += [(x, y) for y in rng.sample(b_set, k)]
        rng.shuffle(pairs)
        return dict(a_set=a_set, b_set=b_set, pairs=pairs, free=free, multi=multi, no_image=no_image, singles=singles)
    check(False, "no acceptable relation draw")


def _set_text(items) -> str:
    return "{" + ", ".join(str(i) for i in items) + "}"


@generator("exd_x_r2g_relation_repair", level=15, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_r2g_relation_repair(rng: random.Random, tier: str = "exam") -> dict:
    """Count the elements without an image, the surplus pairs, the repair cost and the functions allowed."""
    check(tier == "exam", "exd_x_r2g_relation_repair: only the exam tier exists")
    d = _draw(rng)
    a_set, b_set, pairs = d["a_set"], d["b_set"], d["pairs"]
    a, multi = d["free"], d["multi"]
    nb, na = len(b_set), len(a_set)
    images = {x: [y for (u, y) in pairs if u == x] for x in a_set}
    check(sum(1 for x in a_set if not images[x]) == a, "elements without an image disagree")
    check(sorted(len(v) for v in images.values() if len(v) > 1) == sorted(multi), "image counts disagree")
    surplus = sum(k - 1 for k in multi)
    repair = a + surplus
    product = prod(multi)
    count = nb**a * product
    pair_text = "{" + ", ".join(f"({x}, {y})" for x, y in pairs) + "}"
    stem = rng.choice(_INTRO).format(A=_set_text(a_set), B=_set_text(b_set), R=pair_text)
    unused = nb - len({y for (_x, y) in pairs})
    parts = [
        build_part(
            "How many elements of A have no image under R?",
            a,
            kind="number",
            pad_counts=True,
            marks=1,
            working=[f"Elements of A that never appear first in a pair: {a}."],
            wrong=[
                (len(multi), "confuses-no-image-with-multiple-images"),
                (d["singles"], "counts-single-image-elements"),
                (na - len(pairs) if na > len(pairs) else None, "confuses-pairs-with-elements"),
                (unused, "counts-unused-codomain-elements"),
                (a + len(multi), "counts-all-failures-together"),
                (na - a, "counts-the-complement"),
                (a + 1, "off-by-one-count"),
                (a - 1 if a > 1 else None, "off-by-one-count"),
                (a + 2, "off-by-one-count"),
            ],
        ),
        build_part(
            "What is the fewest number of pairs that must be deleted from R so that no element of A has more than "
            "one image?",
            surplus,
            kind="number",
            pad_counts=True,
            marks=2,
            working=[
                "An element with k images needs k - 1 of its pairs deleted.",
                f"The elements with several images have {', '.join(str(k) for k in sorted(multi))} images, so "
                f"{surplus} pair{' goes' if surplus == 1 else 's go'}.",
            ],
            wrong=[
                (len(multi), "counts-elements-not-pairs"),
                (sum(multi), "deletes-every-pair-of-a-multi-image-element"),
                (a, "swaps-the-two-repairs"),
                (repair, "counts-all-failures-together"),
                (len(pairs) - na if len(pairs) > na else None, "confuses-pairs-with-elements"),
                (surplus + 1, "off-by-one-count"),
                (surplus - 1 if surplus > 1 else None, "off-by-one-count"),
            ],
        ),
        build_part(
            "Hence find the fewest pairs that must be added or deleted to turn R into a function from A to B.",
            repair,
            kind="number",
            pad_counts=True,
            marks=2,
            follow_through="part_a + part_b",
            working=[
                f"Each element with no image needs one pair added, and the surplus pairs go: {a} + {surplus} = {repair}."
            ],
            wrong=[
                (max(a, surplus), "takes-larger-of-the-two-repairs"),
                (a + sum(multi), "deletes-every-pair-of-a-multi-image-element"),
                (a * surplus, "multiplies-instead-of-adds"),
                (repair + 1, "off-by-one-count"),
                (repair - 1, "off-by-one-count"),
            ],
        ),
        build_part(
            "Count the functions f from A to B such that f(x) is the image of x whenever R gives x exactly one "
            "image, f(x) is one of the images of x whenever R gives x several, and f(x) can be any element of B "
            "whenever R gives x none.",
            count,
            kind="number",
            pad_counts=True,
            marks=2,
            follow_through=f"{nb}**part_a*{product}",
            working=[
                f"{'The 1 element' if a == 1 else f'Each of the {a} elements'} with no image has {nb} choices; the others have {', '.join(str(k) for k in sorted(multi))} choices.",
                f"{nb}^{a} x {product} = {count}.",
            ],
            wrong=[
                (nb**a, "ignores-choice-among-several-images"),
                (product, "ignores-free-elements"),
                (nb**na, "counts-all-functions-from-a-to-b"),
                (nb * a + product, "adds-instead-of-multiplies"),
                (a**nb * product, "base-and-exponent-swapped"),
            ],
        ),
    ]
    return dict(text=stem, latex=latex_with_sets(stem), parts=parts)
