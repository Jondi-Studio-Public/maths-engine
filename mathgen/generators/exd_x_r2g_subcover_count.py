"""Exam question: sizes and counts of finite subcovers of a finite open cover (`ucv1-compactness-finite-check`).

A finite space X is compact, so every open cover has a finite subcover; an exam then asks how small one can be and
how many there are. The stem lists a finite set X and an open cover U1, ..., UN of it. The parts chain:

* (a) the number of essential members, those holding a point of X that no other member holds (2 marks);
* (b) the size of a smallest subcover (2 marks);
* (c) the number of members that may be discarded when a smallest subcover is kept, N - (b) (follow-through);
* (d) how many of the 2^N subfamilies are themselves subcovers of X (2 marks).

Everything is found by brute force over the 2^N subfamilies, so the key is exact. A draw is kept only when the
answers are varied enough to be worth asking (at least one inessential member, a smallest subcover of 3 to N - 2
sets, and some choice in which subcover to take).
"""

from __future__ import annotations

import random
import string
from itertools import combinations

from ..core import check, generator
from .exd_x_r2g_support import build_part, latex_with_sets, redraws

_LETTERS = [c for c in string.ascii_uppercase if c not in "IOQ"]


def _analyse(masks: list[int], full: int) -> dict | None:
    n = len(masks)
    covers = []
    for sub in range(1 << n):
        union = 0
        for i in range(n):
            if sub >> i & 1:
                union |= masks[i]
        if union == full:
            covers.append(sub)
    if not covers:
        return None
    sizes = [bin(c).count("1") for c in covers]
    smallest = min(sizes)
    essential = 0
    for i in range(n):
        others = 0
        for j in range(n):
            if j != i:
                others |= masks[j]
        if masks[i] & ~others:
            essential += 1
    # greedy choice, a typical wrong method for the smallest subcover
    covered, greedy = 0, 0
    while covered != full:
        best = max(range(n), key=lambda i: (bin(masks[i] & ~covered).count("1"), -i))
        covered |= masks[best]
        greedy += 1
    once = sum(1 for p in range(full.bit_length()) if sum(masks[i] >> p & 1 for i in range(n)) == 1)
    return dict(
        covers=len(covers),
        smallest=smallest,
        smallest_count=sizes.count(smallest),
        essential=essential,
        greedy=greedy,
        once=once,
        singletons=sum(1 for m in masks if bin(m).count("1") == 1),
        largest=max(bin(m).count("1") for m in masks),
    )


def _draw(rng: random.Random) -> dict:
    for _ in range(2000):
        m = rng.randint(7, 11)
        n = rng.randint(6, 8)
        points = sorted(rng.sample(_LETTERS, m))
        full = (1 << m) - 1
        masks = []
        for _i in range(n):
            size = rng.choice([1, 2, 2, 3, 3, 4, 4, 5])
            members = rng.sample(range(m), size)
            masks.append(sum(1 << p for p in members))
        if len(set(masks)) < n:
            continue
        info = _analyse(masks, full)
        if info is None:
            continue
        if not (3 <= info["smallest"] <= n - 3) or info["essential"] == n:
            continue
        if info["smallest_count"] < 2 and rng.random() < 0.6:
            continue
        if info["covers"] < 6 or info["greedy"] == info["smallest"] and rng.random() < 0.5:
            continue
        return dict(points=points, masks=masks, m=m, n=n, info=info)
    check(False, "no acceptable finite cover")


def _fmt_set(points: list[str], mask: int) -> str:
    return "{" + ", ".join(p for i, p in enumerate(points) if mask >> i & 1) + "}"


_INTRO = [
    "Let X = {points} be a finite topological space, and let {names} be an open cover of X, where",
    "The finite space X = {points} has an open cover {names} given by",
    "A finite topological space has underlying set X = {points}. Its open sets include the following, which form "
    "an open cover {names} of X:",
]


@generator("exd_x_r2g_subcover_count", level=15, presets={"exam": dict(tier="exam")})
@redraws
def exd_x_r2g_subcover_count(rng: random.Random, tier: str = "exam") -> dict:
    """Essential members, smallest subcover and subcover counts for a finite open cover."""
    check(tier == "exam", "exd_x_r2g_subcover_count: only the exam tier exists")
    d = _draw(rng)
    points, masks, n, info = d["points"], d["masks"], d["n"], d["info"]
    a, b, total = info["essential"], info["smallest"], info["covers"]
    x_text = "{" + ", ".join(points) + "}"
    names = f"{{U1, ..., U{n}}}"
    listing = "; ".join(f"U{i + 1} = {_fmt_set(points, mask)}" for i, mask in enumerate(masks))
    intro = rng.choice(_INTRO).format(points=x_text, names=names)
    stem = f"{intro} {listing}."

    parts = [
        build_part(
            "Call a member of the cover essential if it contains a point of X that lies in no other member. "
            "How many members of the cover are essential?",
            a,
            kind="number",
            pad_counts=True,
            marks=2,
            working=["A point lying in exactly one member forces that member into every subcover.", f"There are {a}."],
            wrong=[
                (info["once"], "counts-points-not-sets"),
                (n - a, "counts-the-complement"),
                (info["singletons"], "single-point-sets-taken-as-essential"),
                (a + 1, "off-by-one-count"),
                (a - 1 if a else None, "off-by-one-count"),
            ],
        ),
        build_part(
            "Find the smallest number of members of the cover whose union is X (the size of a smallest finite "
            "subcover).",
            b,
            kind="number",
            pad_counts=True,
            marks=2,
            working=[f"Checking every subfamily, the smallest union equal to X uses {b} members."],
            wrong=[
                (n, "takes-whole-cover-as-minimum"),
                (info["greedy"] if info["greedy"] != b else None, "greedy-choice-taken-as-minimum"),
                (-(-d["m"] // info["largest"]), "counts-points-over-largest-set"),
                (a, "essential-members-taken-as-subcover"),
                (b + 1, "off-by-one-count"),
            ],
        ),
        build_part(
            "Hence find how many members of the cover can be discarded when a smallest subcover is kept.",
            n - b,
            kind="number",
            pad_counts=True,
            marks=1,
            follow_through=f"{n} - part_b",
            working=[f"{n} - {b} = {n - b} members are not needed."],
            wrong=[
                (b, "reports-subcover-size-not-discards"),
                (n - b + 1, "off-by-one-count"),
                (n - b - 1, "off-by-one-count"),
                (n, "takes-whole-cover-as-minimum"),
            ],
        ),
        build_part(
            f"How many of the 2^{n} = {2**n} subfamilies of the cover (counting the whole cover) have union X?",
            total,
            kind="number",
            pad_counts=True,
            marks=2,
            working=[f"Every subfamily containing the essential members is tested; {total} of them cover X."],
            wrong=[
                (info["smallest_count"], "counts-only-minimum-subcovers"),
                (2**n - total, "counts-the-complement"),
                (total - 1, "omits-the-whole-cover"),
                (2 ** (n - a), "assumes-essential-members-suffice"),
            ],
        ),
    ]
    check(
        sum(1 for r in range(n + 1) for c in combinations(range(n), r) if _union(masks, c) == (1 << d["m"]) - 1)
        == total,
        "subcover count disagrees with a second enumeration",
    )
    return dict(text=stem, latex=latex_with_sets(stem), parts=parts)


def _union(masks, chosen) -> int:
    out = 0
    for i in chosen:
        out |= masks[i]
    return out
