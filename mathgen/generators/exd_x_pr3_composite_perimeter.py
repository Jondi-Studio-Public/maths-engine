"""Exam source for L-shaped perimeter: a chain from missing sides to the cost of a fence.

The shape is described, not drawn: a rectangle with a rectangular corner cut out, so two of the six boundary
lengths are not given and have to be found. Part (a) finds the remaining part of the first long side, part (b) the
remaining part of the second side, part (c) uses both to find the perimeter, and part (d) prices a fence round the
boundary with a gateway left open. Parts (c) and (d) carry `follow_through`.

The key is built forwards from whole-metre dimensions and checked by walking the six vertices of the outline.
"""

from __future__ import annotations

from fractions import Fraction

from mathgen.core import attempts, check, generator

from ._exd_pr3_support import article, leaks, part

_PLACES = [
    "A courtyard at a school in Parramatta",
    "A vegetable garden in Launceston",
    "A pool deck at a resort on the Gold Coast",
    "A backyard lawn in Perth",
    "A sports shed slab in Dubbo",
    "A playground area in Ballarat",
]
_CUTS = [
    "is L-shaped. It was made by cutting a rectangle out of one corner of {aW} {W} m by {H} m rectangle. The cut-out "
    "is {w} m along {aW} {W} m side and {h} m along {aH} {H} m side.",
    "is L-shaped. Start with {aW} {W} m by {H} m rectangle and remove a rectangular corner that measures {w} m by "
    "{h} m, with the {w} m edge along {aW} {W} m side and the {h} m edge along {aH} {H} m side.",
]
_FENCES = [
    "A fence is built all the way round the outside of the L-shape, except across a {g} m gateway. The fencing "
    "costs ${r} per metre.",
    "Fencing costing ${r} per metre goes all the way round the boundary, leaving a {g} m gap for a gate.",
]
_RATES = (12, 15, 18, 20, 24, 25, 28, 30, 35, 40, 45)


def _outline_perimeter(W: int, H: int, w: int, h: int) -> int:
    """Perimeter by walking the six vertices of the L (cut from the top right corner)."""
    pts = [(0, 0), (W, 0), (W, H - h), (W - w, H - h), (W - w, H), (0, H)]
    return sum(abs(x2 - x1) + abs(y2 - y1) for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))


@generator("exd_x_pr3_composite_perimeter", level=8, presets={"exam": dict(tier="exam")})
def exd_x_pr3_composite_perimeter(rng, tier="exam"):
    check(tier == "exam", "only the exam tier exists")
    for _ in attempts(1000, "an L-shape whose answers are not in the prompt"):
        H = rng.randint(10, 30)
        W = rng.randint(H + 1, 40)
        w = rng.randint(3, W - 3)
        h = rng.randint(3, H - 3)
        r = rng.choice(_RATES)
        g = rng.randint(2, 4)
        a, b = W - w, H - h
        P = W + H + w + h + a + b
        check(P == _outline_perimeter(W, H, w, h) == 2 * (W + H), "perimeter disagrees with the outline walk")
        cost = (P - g) * r
        stem = (
            f"{rng.choice(_PLACES)} {rng.choice(_CUTS).format(W=W, H=H, w=w, h=h, aW=article(W), aH=article(H))} "
            f"{rng.choice(_FENCES).format(g=g, r=r)}"
        )
        if leaks(stem, [a, b, P, cost]) or len({W, H, w, h}) < 4:
            continue
        break
    parts = [
        part(
            f"How long is the part of the {W} m side that is left after the cut, in metres?",
            a,
            marks=1,
            working=[f"{W} − {w} = {a} m."],
            candidates=[
                (W + w, "side-added-not-subtracted"),
                (H - h, "wrong-dimension-for-missing-side"),
                (w, "used-cut-out-as-remainder"),
            ],
        ),
        part(
            f"How long is the part of the {H} m side that is left after the cut, in metres?",
            b,
            marks=1,
            working=[f"{H} − {h} = {b} m."],
            candidates=[
                (H + h, "side-added-not-subtracted"),
                (W - w, "wrong-dimension-for-missing-side"),
                (h, "used-cut-out-as-remainder"),
            ],
        ),
        part(
            "Using your answers to (a) and (b), find the perimeter of the L-shape, in metres.",
            P,
            marks=2,
            follow_through=f"{W}+{H}+{w}+{h}+part_a+part_b",
            working=[f"Six sides: {W} + {b} + {w} + {h} + {a} + {H} = {P} m."],
            candidates=[
                (W + H + a + b, "omits-indented-sides"),
                (W * H - w * h, "perimeter-as-area"),
                (W + H + w + h, "adds-only-given-lengths"),
            ],
        ),
        part(
            "Find the cost of the fencing, in dollars.",
            cost,
            marks=2,
            follow_through=f"(part_c-{g})*{r}",
            working=[f"Fence length = {P} − {g} = {P - g} m.", f"Cost = {P - g} × {r} = ${cost}."],
            candidates=[
                (P * r, "ignored-gap-in-fence"),
                ((P + g) * r, "gap-added-instead-of-removed"),
                (P * r - g, "subtracted-unlike-quantities"),
                (P - g + r, "rate-added-not-multiplied"),
            ],
        ),
    ]
    check(all(Fraction(p["answer"]) > 0 for p in parts), "a part is not positive")
    return dict(text=stem, parts=parts)
