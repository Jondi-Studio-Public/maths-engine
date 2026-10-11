"""Exam tier for the dot product: no components are given for the main kind.

Two kinds, chosen per seed:

* ``angle``: only the lengths ``|a|``, ``|b|`` and the angle between them are known, and the learner
  evaluates ``(p a + q b) . (r a + s b)``. This needs bilinearity and ``a . b = |a||b| cos(theta)``;
  there is nothing to multiply component by component. The angle is 60 or 120 degrees and ``|a||b|`` is
  even, so ``a . b`` is an integer.
* ``perpendicular``: find ``k`` so that ``u + k v`` is perpendicular to ``w``. The answer is
  ``k = -(u . w) / (v . w)``. It is built backwards: ``k``, ``v``, ``w`` and two components of ``u`` are
  drawn, and the last component of ``u`` is solved from ``(u + k v) . w = 0`` (``w`` has a +-1 in that slot).

Answer kind is ``number`` and every key is an integer.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, check_equivalent, generator, join_terms, pick_distractors

_ANGLE = (
    "The vectors a and b have |a| = {la} and |b| = {lb}, and the angle between them is {deg}°. Find ({e1}) · ({e2}).",
    "Two vectors satisfy |a| = {la}, |b| = {lb}, and the angle between a and b is {deg}°. Calculate ({e1}) · ({e2}).",
)
_PERP = (
    "u = {u}, v = {v} and w = {w}. Find the value of k for which u + kv is perpendicular to w.",
    "Given u = {u}, v = {v} and w = {w}, find k so that the vector u + kv is perpendicular to w.",
)


def _fmt(v) -> str:
    return "(" + ", ".join(str(x) for x in v) + ")"


def _dot(u, v) -> int:
    return sum(x * y for x, y in zip(u, v))


def _result(text: str, ans: int, cands) -> dict | None:
    distractors = pick_distractors(str(ans), [str(c) for c in cands])
    if len(distractors) < 2:
        return None  # caller redraws
    return dict(
        text=text,
        latex=text,
        answer=ans,
        answer_text=str(ans),
        answer_latex=str(ans),
        distractors=distractors,
    )


def _angle(rng: random.Random) -> dict | None:
    while True:
        la, lb = rng.randint(2, 7), rng.randint(2, 7)
        if (la * lb) % 2:
            continue
        deg = rng.choice([60, 120])
        p, q, r, s = (rng.choice([-3, -2, -1, 1, 2, 3]) for _ in range(4))
        ab = la * lb // 2 * (1 if deg == 60 else -1)  # a . b = |a||b| cos(deg)
        ans = p * r * la * la + (p * s + q * r) * ab + q * s * lb * lb
        # Independent route: represent a and b by explicit plane coordinates and dot them out.
        # a = (|a|, 0); b = |b| (cos, sin) -- scaled by 2 to keep integer arithmetic, sin(60) = sqrt(3)/2
        # contributes only through a . b, so check it with exact fractions on the Gram matrix instead.
        gram = [[Fraction(la * la), Fraction(ab)], [Fraction(ab), Fraction(lb * lb)]]
        coef1, coef2 = (p, q), (r, s)
        direct = sum(coef1[i] * coef2[j] * gram[i][j] for i in range(2) for j in range(2))
        check_equivalent(direct, ans, "angle: Gram-matrix route disagrees with the expanded product")
        if p * s - q * r != 0 and ans != 0:
            break
    cross_term = p * s + q * r
    cands = [
        p * r * la * la + q * s * lb * lb,  # dropped the a . b terms
        p * r * la * la + cross_term * (-ab) + q * s * lb * lb,  # wrong sign for cos(theta)
        p * r * la + cross_term * ab + q * s * lb,  # used |a| for a . a instead of |a| squared
        (p * r * la * la + q * s * lb * lb) + p * s * ab + q * r * (-ab),  # took b . a as -(a . b)
    ]
    e1 = join_terms([(p, "a"), (q, "b")])
    e2 = join_terms([(r, "a"), (s, "b")])
    text = rng.choice(_ANGLE).format(la=la, lb=lb, deg=deg, e1=e1, e2=e2)
    return _result(text, ans, cands)


def _perpendicular(rng: random.Random) -> dict | None:
    while True:
        k = rng.choice([-5, -4, -3, -2, 2, 3, 4, 5])
        v = [rng.randint(-4, 4) for _ in range(3)]
        w = [rng.randint(-4, 4), rng.randint(-4, 4), rng.choice([-1, 1])]
        u = [rng.randint(-5, 5), rng.randint(-5, 5), 0]
        u[2] = (-k * _dot(v, w) - u[0] * w[0] - u[1] * w[1]) // w[2]
        if abs(_dot(v, w)) >= 2 and _dot(u, w) != 0 and -5 <= u[2] <= 9:
            break
    combined = [x + k * y for x, y in zip(u, v)]
    check(_dot(combined, w) == 0, "perpendicular: u + kv is not perpendicular to w")
    check(Fraction(-_dot(u, w), _dot(v, w)) == k, "perpendicular: closed form disagrees with the drawn k")
    cands = [-k, -_dot(u, w), _dot(u, w)]  # sign slip; forgot to divide by v . w (two signs)
    if _dot(u, v) != 0 and _dot(u, w) % _dot(u, v) == 0:
        cands.append(-_dot(u, w) // _dot(u, v))  # divided by u . v rather than v . w
    text = rng.choice(_PERP).format(u=_fmt(u), v=_fmt(v), w=_fmt(w))
    return _result(text, k, cands)


@generator(
    "exd_x_g1_vector_dot_product",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_g1_vector_dot_product(rng: random.Random, tier: str = "exam") -> dict:
    """Dot product from lengths and an angle, or the k that makes u + kv perpendicular to w."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(200):
        built = _angle(rng) if rng.random() < 0.5 else _perpendicular(rng)
        if built is not None:
            return built
    raise ValueError("no usable draw in 200 attempts")
