"""Exam-tier source for the skill "Use ordered pair notation and column vector notation to represent a vector".

The practice tiers go from two points to one vector, or from a point and one
component to the other endpoint.  This tier chains vector statements so that the
learner must decide which direction to travel along each vector:

* ``chain``         -- the vectors A->B and C->B are given; find a coordinate of C.
* ``midpoint``      -- M is the midpoint of AB and A->M is given; find a coordinate of B.
* ``parallelogram`` -- A, B and C are consecutive vertices of a parallelogram ABCD; find a coordinate of D.

Built backwards from the target point, so every coordinate is an integer.
"""

from __future__ import annotations

import random

from ..core import check_equivalent, generator, pick_distractors

_R = range(-9, 10)


def _build(rng: random.Random) -> dict:
    form = rng.choice(["chain", "midpoint", "parallelogram"])
    axis = rng.choice([0, 1])
    name = "xy"[axis]
    for _ in range(300):
        a = (rng.choice(_R), rng.choice(_R))
        if form == "chain":
            ab = (rng.randint(-8, 8), rng.randint(-8, 8))
            cb = (rng.randint(-8, 8), rng.randint(-8, 8))
            if 0 in ab or 0 in cb:
                continue
            c = (a[0] + ab[0] - cb[0], a[1] + ab[1] - cb[1])
            target = c
            check_equivalent(c[axis] + cb[axis], a[axis] + ab[axis])  # C + CB = A + AB = B
            text = (
                f"Point A is ({a[0]}, {a[1]}). The vector from A to B is the column vector ({ab[0]}, {ab[1]}) "
                f"and the vector from C to B is the column vector ({cb[0]}, {cb[1]}). "
                f"What is the {name}-coordinate of C?"
            )
            other = 1 - axis
            cands = [
                a[axis] + ab[axis] + cb[axis],  # travels along C->B the wrong way
                a[axis] - ab[axis] + cb[axis],  # reverses A->B
                a[axis] + ab[axis],  # reports B, forgetting the second vector
                a[other] + ab[other] - cb[other],  # other coordinate
            ]
        elif form == "midpoint":
            am = (rng.randint(-8, 8), rng.randint(-8, 8))
            if 0 in am:
                continue
            b = (a[0] + 2 * am[0], a[1] + 2 * am[1])
            target = b
            check_equivalent(a[axis] + am[axis], (a[axis] + b[axis]) / 2)  # M is the midpoint of A and B
            text = (
                f"Point A is ({a[0]}, {a[1]}). M is the midpoint of AB and the vector from A to M "
                f"is the column vector ({am[0]}, {am[1]}). What is the {name}-coordinate of B?"
            )
            other = 1 - axis
            cands = [
                a[axis] + am[axis],  # reports M
                2 * am[axis],  # forgets the starting point
                a[axis] - 2 * am[axis],  # travels the wrong way
                a[other] + 2 * am[other],  # other coordinate
            ]
        else:
            b = (a[0] + rng.randint(-8, 8), a[1] + rng.randint(-8, 8))
            c = (b[0] + rng.randint(-8, 8), b[1] + rng.randint(-8, 8))
            if len({a, b, c}) < 3 or (c[0] - b[0]) * (b[1] - a[1]) == (c[1] - b[1]) * (b[0] - a[0]):
                continue  # degenerate: vertices collinear
            d = (a[0] + c[0] - b[0], a[1] + c[1] - b[1])
            target = d
            # opposite sides: vector A->B equals vector D->C
            check_equivalent(b[axis] - a[axis], c[axis] - d[axis])
            text = (
                f"ABCD is a parallelogram with A({a[0]}, {a[1]}), B({b[0]}, {b[1]}) and C({c[0]}, {c[1]}). "
                f"What is the {name}-coordinate of D?"
            )
            other = 1 - axis
            cands = [
                b[axis] + c[axis] - a[axis],  # pairs the wrong vertices
                a[axis] + b[axis] - c[axis],
                a[axis] + b[axis] + c[axis],  # adds all three
                a[other] + c[other] - b[other],  # other coordinate
            ]
        answer = target[axis]
        distractors = pick_distractors(str(answer), [str(v) for v in cands])
        if len(distractors) >= 3 and abs(answer) <= 40:
            break
    else:  # pragma: no cover
        raise ValueError("could not draw a question")
    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=[f"{name}-coordinate = {answer}"],
        distractors=distractors[:3],
    )


@generator(
    "exd_x_g3_vec_notation",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_g3_vec_notation(rng: random.Random, tier: str = "exam") -> dict:
    """Chain vector statements (two vectors, a midpoint, or a parallelogram) to a coordinate."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
