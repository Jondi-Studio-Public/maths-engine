"""Exam tier for the cross product: the vectors are not handed over ready to multiply.

Two kinds, chosen per seed:

* ``points``: three points ``A``, ``B``, ``C`` are given and the learner must find ``AB x AC``.
  The edge vectors have to be formed first, and the order of the product matters.
* ``bilinear``: ``p`` and ``q`` are given and the learner is asked for ``(a p + b q) x (c p + d q)``.
  Expanding with the rules ``p x p = 0`` and ``q x p = -(p x q)`` reduces this to
  ``(a d - b c) (p x q)``, so a learner who chooses that route does one cross product; one who does not
  forms two combined vectors and crosses them. Either way the key is the same.

The skill's answer kind is ``list``, which grades in any order, so asking for the whole vector would
accept its components permuted. Each question therefore asks for one named component: the learner still
forms the whole product, but the key is a single number that no reordering can reach. Distractors are the
same component of each principled wrong vector.
"""

from __future__ import annotations

import random

from sympy import Matrix

from ..core import check, check_equivalent, generator, join_terms, pick_distractors

_POINTS = (
    "The points A{a}, B{b} and C{c} are given. Find the {axis}-component of AB × AC.",
    "A is {a}, B is {b} and C is {c}. Calculate the {axis}-component of the cross product AB × AC.",
)
_BILINEAR = (
    "p = {p} and q = {q}. Find the {axis}-component of ({e1}) × ({e2}).",
    "Given p = {p} and q = {q}, calculate the {axis}-component of ({e1}) × ({e2}).",
)


def _fmt(v) -> str:
    return "(" + ", ".join(str(x) for x in v) + ")"


def _cross(u, v) -> list[int]:
    w = Matrix(u).cross(Matrix(v))
    return [int(x) for x in w]


def _sub(u, v):
    return [a - b for a, b in zip(u, v)]


def _scale(k, v):
    return [k * x for x in v]


_AXES = ("x", "y", "z")


def _result(text: str, ans, cands, k: int) -> dict | None:
    """Key and distractors are the ``k``-th component of the answer and of each wrong vector."""
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


def _points(rng: random.Random) -> dict | None:
    k = rng.randrange(3)  # which component is asked for
    while True:
        a, b, c = ([rng.randint(-5, 5) for _ in range(3)] for _ in range(3))
        ab, ac = _sub(b, a), _sub(c, a)
        ans = _cross(ab, ac)
        if any(ans):
            break
    # Independent check: the product is orthogonal to both edges.
    check(sum(x * y for x, y in zip(ans, ab)) == 0 and sum(x * y for x, y in zip(ans, ac)) == 0)
    cands = [
        _scale(-1, ans),  # AC x AB: the order reversed
        _cross(b, c),  # crossed the position vectors of B and C, never formed the edges
        _cross(_sub(a, b), ac),  # edge BA used for AB
    ]
    text = rng.choice(_POINTS).format(a=_fmt(a), b=_fmt(b), c=_fmt(c), axis=_AXES[k])
    return _result(text, ans, cands, k)


def _bilinear(rng: random.Random) -> dict | None:
    k = rng.randrange(3)  # which component is asked for
    while True:
        p = [rng.randint(-4, 4) for _ in range(3)]
        q = [rng.randint(-4, 4) for _ in range(3)]
        a, c = rng.choice([1, 2, 3]), rng.choice([1, 2, 3])
        b, d = rng.choice([-3, -2, -1, 1, 2, 3]), rng.choice([-3, -2, -1, 1, 2, 3])
        f = a * d - b * c
        pq = _cross(p, q)
        if f != 0 and any(pq) and a * d + b * c != f and a * d != f:
            break
    left = [a * x + b * y for x, y in zip(p, q)]
    right = [c * x + d * y for x, y in zip(p, q)]
    ans = _cross(left, right)
    check_equivalent(ans, _scale(f, pq), "bilinear: direct product disagrees with (ad - bc)(p x q)")
    cands = [
        _scale(a * d + b * c, pq),  # treated q x p as +(p x q)
        _scale(-f, pq),  # reversed the order of the two factors
        _scale(a * d, pq),  # dropped the q x p term
    ]
    e1 = join_terms([(a, "p"), (b, "q")])
    e2 = join_terms([(c, "p"), (d, "q")])
    text = rng.choice(_BILINEAR).format(p=_fmt(p), q=_fmt(q), e1=e1, e2=e2, axis=_AXES[k])
    return _result(text, ans, cands, k)


@generator(
    "exd_x_g1_vector_cross_product",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_g1_vector_cross_product(rng: random.Random, tier: str = "exam") -> dict:
    """AB x AC from three points, or the cross product of two linear combinations of p and q."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(200):
        built = _points(rng) if rng.random() < 0.5 else _bilinear(rng)
        if built is not None:
            return built
    raise ValueError("no usable draw in 200 attempts")
