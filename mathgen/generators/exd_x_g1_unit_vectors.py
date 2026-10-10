"""Exam tier for unit vectors and perpendicular unit vectors (2D, i-j components).

The practice tiers give the vector and its magnitude. Here the vector has to be assembled first and
its magnitude found by the learner, and the unit vector may then be rotated through a right angle.

Two kinds, chosen per seed:

* ``points``: points ``A`` and ``B`` are given; find one coefficient of the unit vector perpendicular
  to ``AB`` (obtained by rotating ``AB`` a quarter turn anticlockwise or clockwise, as stated).
* ``combination``: ``p`` and ``q`` are given; find one coefficient of the unit vector in the direction
  of ``a p + b q``.

Both are built backwards from a Pythagorean triple ``(|x|, |y|, m)`` so the magnitude is an integer and
the unit vector is exact. Answer kind is ``fraction``, graded exactly.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, join_terms, pick_distractors

# (leg, leg, hypotenuse), each leg nonzero and the two unequal.
_TRIPLES = [(3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 15, 17), (9, 12, 15), (7, 24, 25), (12, 16, 20), (20, 21, 29)]

_POINTS = (
    "A is the point ({ax}, {ay}) and B is the point ({bx}, {by}). Rotating AB through 90° {turn} gives a vector "
    "perpendicular to AB. Find the coefficient of {axis} in the unit vector in that direction.",
    "Given A({ax}, {ay}) and B({bx}, {by}), let w be the vector AB turned through 90° {turn}. "
    "What is the coefficient of {axis} in the unit vector in the direction of w?",
)
_COMBINATION = (
    "p = ({px}, {py}) and q = ({qx}, {qy}). Find the coefficient of {axis} in the unit vector in the direction of {e}.",
    "Let p = ({px}, {py}) and q = ({qx}, {qy}). The vector {e} has a unit vector in the same direction; "
    "find its coefficient of {axis}.",
)


def _frac_text(f: Fraction) -> str:
    return str(f)


def _result(text: str, ans: Fraction, cands) -> dict | None:
    answer_text = _frac_text(ans)
    distractors = pick_distractors(answer_text, [_frac_text(Fraction(c)) for c in cands])
    if len(distractors) < 2:
        return None  # caller redraws
    return dict(
        text=text,
        latex=text,
        answer={"p": ans.numerator, "q": ans.denominator},
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
    )


def _draw_vector(rng: random.Random):
    """A target vector (x, y) with integer magnitude m, any quadrant, either leg order."""
    a, b, m = rng.choice(_TRIPLES)
    if rng.random() < 0.5:
        a, b = b, a
    x, y = a * rng.choice([1, -1]), b * rng.choice([1, -1])
    check(x * x + y * y == m * m, "target vector does not have integer magnitude m")
    return x, y, m


def _points(rng: random.Random) -> dict | None:
    x, y, m = _draw_vector(rng)
    ax, ay = rng.randint(-6, 6), rng.randint(-6, 6)
    bx, by = ax + x, ay + y
    anticlockwise = rng.random() < 0.5
    # Rotate AB a quarter turn: anticlockwise (x, y) -> (-y, x); clockwise (x, y) -> (y, -x).
    wx, wy = (-y, x) if anticlockwise else (y, -x)
    ask_i = rng.random() < 0.5
    ans = Fraction(wx if ask_i else wy, m)
    # Independent check: the rotated unit vector has length 1 and is perpendicular to AB.
    check(Fraction(wx, m) ** 2 + Fraction(wy, m) ** 2 == 1 and wx * x + wy * y == 0)
    other = Fraction(wy if ask_i else wx, m)
    unrotated = Fraction(x if ask_i else y, m)  # unit vector of AB itself, never rotated
    opposite_turn = -ans  # turned the wrong way
    raw = wx if ask_i else wy  # forgot to divide by |AB|
    cands = [raw, other, unrotated, opposite_turn]
    text = rng.choice(_POINTS).format(
        ax=ax,
        ay=ay,
        bx=bx,
        by=by,
        turn="anticlockwise" if anticlockwise else "clockwise",
        axis="i" if ask_i else "j",
    )
    return _result(text, ans, cands)


def _combination(rng: random.Random) -> dict | None:
    x, y, m = _draw_vector(rng)
    alpha = rng.choice([2, 3, 4, -2, -3])
    beta = rng.choice([1, -1])
    px, py = rng.randint(-4, 4), rng.randint(-4, 4)
    # a p + b q = (x, y)  =>  q = (r - a p) / b, with b = +-1 so q is integral.
    qx, qy = beta * (x - alpha * px), beta * (y - alpha * py)
    check((alpha * px + beta * qx, alpha * py + beta * qy) == (x, y), "combination does not give the drawn vector")
    ask_i = rng.random() < 0.5
    ans = Fraction(x if ask_i else y, m)
    other = Fraction(y if ask_i else x, m)
    sign_slip = (alpha * px - beta * qx, alpha * py - beta * qy)  # subtracted the q term whatever its sign
    sm2 = sign_slip[0] ** 2 + sign_slip[1] ** 2
    sm = int(sm2**0.5)
    cands = [x if ask_i else y, other, -ans]
    if sm * sm == sm2 and sm > 0:
        cands.append(Fraction(sign_slip[0 if ask_i else 1], sm))
    e = join_terms([(alpha, "p"), (beta, "q")])
    text = rng.choice(_COMBINATION).format(
        px=px,
        py=py,
        qx=qx,
        qy=qy,
        e=e,
        axis="i" if ask_i else "j",
    )
    return _result(text, ans, cands)


@generator(
    "exd_x_g1_unit_vectors",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_g1_unit_vectors(rng: random.Random, tier: str = "exam") -> dict:
    """A coefficient of the unit vector perpendicular to AB, or in the direction of a p + b q."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(200):
        built = _points(rng) if rng.random() < 0.5 else _combination(rng)
        if built is not None:
            return built
    raise ValueError("no usable draw in 200 attempts")
