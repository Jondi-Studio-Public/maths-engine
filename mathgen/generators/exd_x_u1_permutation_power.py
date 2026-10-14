"""Exam-tier source for `permutation-composition`: compose two permutations, then use the composite's cycles.

The stem gives two permutations of {1, ..., n} as lists of images and says `p∘q` means "apply q first, then p".

* (a) the composite `p∘q`, written as its list of images;
* (b) the order of `p∘q` in the symmetric group, which needs its cycle structure (the least common multiple of the
  cycle lengths, not their sum);
* (c) the image of one point under a high power of `p∘q`, which is answered by reducing the exponent modulo the
  order from (b), not modulo n.

The draw keeps the composite's order at least 3 and keeps the three wrong routes in (c) (exponent taken as
written modulo n, a single application, the inverse direction) distinct from the key, so each is a real slip.
Every key is recomputed by composing the maps one point at a time and by repeated application, with no use of
the cycle shortcut the working describes.
"""

from __future__ import annotations

import random
from math import lcm

from ..answers import Distractor, exact_value
from ..core import check, generator, pick_distractors

_STEMS = [
    "Let p = {p} and q = {q} be permutations of {{1, ..., {n}}}, each written as the list of images of 1, ..., {n}. "
    "The composite p∘q applies q first and then p.",
    "The permutations p = {p} and q = {q} of {{1, ..., {n}}} are given as lists of images. "
    "Write p∘q for the permutation that applies q first, then p.",
]
_STEM_LATEX = [
    r"\text{{Let }}p={p}\text{{ and }}\,q={q}\text{{ be permutations of }}\{{1,\dots,{n}\}}\text{{, each written as the "
    r"list of images of }}1,\dots,{n}\text{{. The composite }}p\circ q\text{{ applies }}\,q\text{{ first and then }}\,p\text{{.}}",
    r"\text{{The permutations }}p={p}\text{{ and }}\,q={q}\text{{ of }}\{{1,\dots,{n}\}}\text{{ are given as lists of "
    r"images. Write }}p\circ q\text{{ for the permutation that applies }}\,q\text{{ first, then }}\,p\text{{.}}",
]
_C_PROMPTS = [
    "Find the image of {j} under (p∘q)^{big}.",
    "Compute (p∘q)^{big} applied to {j}, that is, the image of {j} after applying p∘q a total of {big} times.",
]


def _text(perm) -> str:
    return "(" + ", ".join(map(str, perm)) + ")"


def _compose(p, q):
    """p after q: i -> p(q(i)), both written as 1-based image tuples."""
    return tuple(p[q[i] - 1] for i in range(len(q)))


def _cycles(perm):
    seen, out = set(), []
    for start in range(1, len(perm) + 1):
        if start in seen:
            continue
        cycle, point = [], start
        for _ in range(len(perm)):
            if point in seen:
                break
            seen.add(point)
            cycle.append(point)
            point = perm[point - 1]
        out.append(cycle)
    return out


def _apply_times(perm, point: int, times: int) -> int:
    for _ in range(times):
        point = perm[point - 1]
    return point


def _order_by_powers(perm) -> int:
    """Brute force: smallest k >= 1 with perm^k the identity."""
    identity = tuple(range(1, len(perm) + 1))
    power = perm
    for k in range(1, 721):
        if power == identity:
            return k
        power = _compose(perm, power)
    raise RuntimeError("order not found")


def _draw(rng: random.Random):
    for _ in range(1000):
        n = rng.choice([6, 7, 8, 9])
        p = tuple(rng.sample(range(1, n + 1), n))
        q = tuple(rng.sample(range(1, n + 1), n))
        r = _compose(p, q)
        cycles = _cycles(r)
        lengths = [len(c) for c in cycles]
        order = lcm(*lengths)
        if order < 4 or order == sum(lengths) or order == max(lengths) or order == len(cycles) or order == n:
            continue
        j = rng.randint(1, n)
        big = rng.randint(2 * n + 3, 140)
        if big % order in (0, 1) or big % n in (0, 1):
            continue
        key = _apply_times(r, j, big)
        slips = {
            _apply_times(r, j, big % n),
            _apply_times(r, j, 1),
            _apply_times(_compose(q, p), j, big),
        }
        inverse_power = (-big) % order
        slips.add(_apply_times(r, j, inverse_power))
        if key in slips or len(slips) < 3:
            continue
        if _compose(q, p) == r or r in (p, q):
            continue
        return n, p, q, r, lengths, order, j, big, key
    raise RuntimeError("no suitable pair of permutations in 1000 draws")


@generator("exd_x_u1_permutation_power", level=16, presets={"exam": dict(tier="exam")})
def exd_x_u1_permutation_power(rng: random.Random, tier: str = "exam") -> dict:
    """Exam tier: composite of two permutations, its order, and the image of a point under a high power."""
    check(tier == "exam", f"unknown tier {tier!r}")
    n, p, q, r, lengths, order, j, big, key = _draw(rng)

    # Independent routes: pointwise composition, order by repeated powers, image by repeated application.
    check(r == tuple(p[q[i] - 1] for i in range(n)), "composite disagrees with pointwise composition")
    check(order == _order_by_powers(r), "order disagrees with repeated powers")
    check(key == _apply_times(r, j, big % order), "image under the power disagrees with the reduced exponent")
    check(sorted(r) == list(range(1, n + 1)), "composite is not a permutation")

    r_text = _text(r)
    qp = _compose(q, p)
    a_candidates = [
        Distractor(_text(qp), "reverses-composition-order", list(qp)),
        Distractor(_text(p), "copies-one-permutation", list(p)),
        Distractor(_text(q), "copies-one-permutation", list(q)),
        Distractor(_text(r[1:] + r[:1]), None, list(r[1:] + r[:1])),
    ]
    b_candidates = [
        Distractor(str(sum(lengths)), "order-as-sum-of-cycle-lengths", sum(lengths)),
        Distractor(str(max(lengths)), "order-as-longest-cycle", max(lengths)),
        Distractor(str(len(lengths)), "order-as-number-of-cycles", len(lengths)),
        Distractor(str(n), None, n),
    ]
    c_slips = [
        (_apply_times(r, j, big % n), "exponent-reduced-by-wrong-modulus"),
        (_apply_times(r, j, 1), None),
        (_apply_times(qp, j, big), "reverses-composition-order"),
        (_apply_times(r, j, (-big) % order), None),
    ]
    c_candidates = [Distractor(str(v), tag, v) for v, tag in c_slips]
    a_dist = pick_distractors(r_text, a_candidates, 3)
    b_dist = pick_distractors(str(order), b_candidates, 3)
    c_dist = pick_distractors(str(key), c_candidates, 3)
    check(len(a_dist) >= 2 and len(b_dist) >= 2 and len(c_dist) >= 2, "too few distractors")

    cycle_text = ", ".join("(" + " ".join(map(str, c)) + ")" for c in _cycles(r))
    variant = rng.randrange(len(_STEMS))
    return dict(
        text=_STEMS[variant].format(p=_text(p), q=_text(q), n=n),
        latex=_STEM_LATEX[variant].format(p=_text(p), q=_text(q), n=n),
        parts=[
            dict(
                prompt="Find p∘q, written as the list of images of 1, ..., {n}.".format(n=n),
                answer_kind="ordered_list",
                answer=list(r),
                answer_text=r_text,
                marks=2,
                working=[f"(p∘q)(i) = p(q(i)) for each i, which gives {r_text}."],
                distractors=a_dist,
            ),
            dict(
                prompt="Find the order of p∘q in the symmetric group S_{n}.".format(n=n),
                prompt_latex=rf"\text{{Find the order of }}p\circ q\text{{ in the symmetric group }}S_{{{n}}}\text{{.}}",
                answer_kind="number",
                answer=order,
                answer_text=str(order),
                marks=2,
                follow_through="perm_order(part_a)",
                sensible_range=(0, None),
                working=[
                    f"In disjoint cycles p∘q = {cycle_text}.",
                    f"The order is the least common multiple of the cycle lengths {', '.join(map(str, lengths))}, "
                    f"which is {order}.",
                ],
                distractors=b_dist,
            ),
            dict(
                prompt=rng.choice(_C_PROMPTS).format(j=j, big=big),
                answer_kind="number",
                answer=key,
                answer_text=str(key),
                marks=2,
                follow_through=f"perm_image(part_a, {j}, {big})",
                sensible_range=(0, None),
                working=[
                    f"(p∘q)^{order} is the identity, and {big} = {big // order}×{order} + {big % order}.",
                    f"So (p∘q)^{big} acts as (p∘q)^{big % order}, which sends {j} to {key}.",
                ],
                distractors=c_dist,
            ),
        ],
    )
