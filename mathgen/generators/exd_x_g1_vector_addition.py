"""Exam tier for vector addition: position vectors in a geometric setting.

The practice tiers hand the learner two vectors and an operation. Here the operation is not named:
the learner has to see that a fourth vertex of a parallelogram, or a point dividing a segment in a
given ratio, is found by combining position vectors, and choose the combination.

Questions are built backwards from the answer so the key is exact for every seed:

* parallelogram ``ABCD`` (vertices in order): ``D = A + C - B``. ``A``, ``B`` and ``D`` are drawn and
  ``C`` is derived from them, so the three given points are never collinear by accident (checked).
* ratio ``AP : PB = m : n``: the step ``d`` and the point ``A`` are drawn, then ``B = A + (m+n) d`` and
  ``P = A + m d``, so every coordinate is an integer.

The skill's answer kind is ``list``, which grades in any order, so asking for the whole vector would
accept its components permuted. Each question therefore asks for one named coordinate of the point: the
learner still has to find the whole position vector, but the key is a single number that no reordering
can reach. Distractors are the same coordinate of each principled wrong vector.
"""

from __future__ import annotations

import random

from ..core import check, check_equivalent, generator, pick_distractors

_PARALLELOGRAM = (
    "ABCD is a parallelogram, with its vertices labelled in order. A is the point {a}, B is {b} and C is {c}. "
    "Find the {axis}-coordinate of D.",
    "The points A{a}, B{b} and C{c} are three consecutive vertices of the parallelogram ABCD. "
    "Find the {axis}-coordinate of the fourth vertex D.",
)
_RATIO = (
    "A is the point {a} and B is the point {b}. The point P lies on the line segment AB with AP : PB = {m} : {n}. "
    "Find the {axis}-coordinate of P.",
    "The point P divides the segment from A{a} to B{b} internally in the ratio {m} : {n} (AP : PB). "
    "Find the {axis}-coordinate of P.",
)


_AXES = ("x", "y", "z")


def _fmt(v) -> str:
    return "(" + ", ".join(str(x) for x in v) + ")"


def _vsub(u, v):
    return [a - b for a, b in zip(u, v)]


def _vadd(u, v):
    return [a + b for a, b in zip(u, v)]


def _rand_vec(rng: random.Random, lo: int, hi: int):
    return [rng.randint(lo, hi) for _ in range(3)]


def _parallelogram(rng: random.Random) -> dict | None:
    k = rng.randrange(3)  # which coordinate is asked for
    while True:
        a, b, d = (_rand_vec(rng, -6, 6) for _ in range(3))
        c = _vadd(_vsub(d, a), b)  # D = A + C - B  =>  C = D - A + B
        ab, bc = _vsub(b, a), _vsub(c, b)
        cross = [
            ab[1] * bc[2] - ab[2] * bc[1],
            ab[2] * bc[0] - ab[0] * bc[2],
            ab[0] * bc[1] - ab[1] * bc[0],
        ]
        if any(cross) and len({_fmt(a), _fmt(b), _fmt(c), _fmt(d)}) == 4:
            break
    check_equivalent(_vsub(_vadd(a, c), b), d, "parallelogram: D = A + C - B does not return the drawn D")
    # AC and BD bisect each other: the midpoints of the diagonals coincide.
    check([x + y for x, y in zip(a, c)] == [x + y for x, y in zip(b, d)], "diagonals do not share a midpoint")
    cands = [
        _vsub(_vadd(b, c), a),  # paired the vertices wrongly: A taken as the vertex opposite D
        _vsub(_vadd(a, b), c),  # paired the vertices wrongly: C taken as the vertex opposite D
        _vadd(a, c),  # forgot to subtract B
    ]
    axis = _AXES[k]
    text = rng.choice(_PARALLELOGRAM).format(a=_fmt(a), b=_fmt(b), c=_fmt(c), axis=axis)
    return _result(text, d, cands, k)


def _ratio(rng: random.Random) -> dict | None:
    k = rng.randrange(3)  # which coordinate is asked for
    m, n = rng.choice([(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (2, 5), (4, 1)])
    while True:
        a = _rand_vec(rng, -6, 6)
        step = _rand_vec(rng, -3, 3)
        if any(step):
            break
    b = [x + (m + n) * s for x, s in zip(a, step)]
    p = [x + m * s for x, s in zip(a, step)]
    # Independent route: the section formula (n A + m B) / (m + n).
    section = [(n * x + m * y) / (m + n) for x, y in zip(a, b)]
    check_equivalent(section, p, "ratio: section formula disagrees with the constructed P")
    swapped = [x + n * s for x, s in zip(a, step)]  # weights the wrong way round (PB : AP)
    unscaled = [x + m * (y - x) for x, y in zip(a, b)]  # A + m AB, forgot to divide by m + n
    summed = _vadd(a, b)  # added the position vectors
    cands = [swapped, unscaled, summed]
    if (m + n) % 2 == 0 or all(s % 2 == 0 for s in step):
        mid = [x + (m + n) * s // 2 for x, s in zip(a, step)]
        cands.append(mid)  # ignored the ratio and took the midpoint
    axis = _AXES[k]
    text = rng.choice(_RATIO).format(a=_fmt(a), b=_fmt(b), m=m, n=n, axis=axis)
    return _result(text, p, cands, k)


def _result(text: str, ans, cands, k: int) -> dict | None:
    """Key and distractors are the ``k``-th coordinate of the answer and of each wrong vector."""
    answer_text = str(ans[k])
    distractors = pick_distractors(answer_text, [str(c[k]) for c in cands])
    if len(distractors) < 2:
        return None  # caller redraws
    return dict(
        text=text,
        latex=text,
        answer=[ans[k]],
        answer_text=answer_text,
        answer_latex=str(ans[k]),
        distractors=distractors,
    )


@generator(
    "exd_x_g1_vector_addition",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_g1_vector_addition(rng: random.Random, tier: str = "exam") -> dict:
    """Position vector of a parallelogram vertex or of a point dividing a segment in a ratio."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(200):
        built = _parallelogram(rng) if rng.random() < 0.5 else _ratio(rng)
        if built is not None:
            return built
    raise ValueError("no usable draw in 200 attempts")
