"""Exam tier for vector magnitude: the exact length of a vector that is never written in components.

Two kinds, chosen per seed:

* ``angle``: only ``|a|``, ``|b|`` and the angle between them are given (60 or 120 degrees), and the
  learner finds ``|p a + q b|``. Squaring the magnitude gives
  ``p^2|a|^2 + q^2|b|^2 + 2 p q |a||b| cos(theta)``, so the key is the square root of an integer.
* ``projection``: the exact length of the vector projection of ``u`` onto ``v``, which is
  ``|u . v| / |v|``. It composes a dot product with a magnitude, and the learner has to recall
  the projection formula.

Answer kind is ``expression`` (an exact surd). The grader also accepts a number within 0.005 of
``answer_alt``, so every distractor is checked to be well clear of the key numerically.
"""

from __future__ import annotations

import random

from sympy import Integer, Rational, sqrt

from ..core import check, check_equivalent, generator, join_terms, pick_distractors

_ANGLE = (
    "The vectors a and b have |a| = {la} and |b| = {lb}, and the angle between them is {deg}°. Find |{e}| (exact).",
    "Two vectors satisfy |a| = {la}, |b| = {lb}, and the angle between a and b is {deg}°. "
    "Find the exact value of |{e}|.",
)
_PROJECTION = (
    "u = {u} and v = {v}. Find the exact length of the vector projection of u onto v.",
    "For u = {u} and v = {v}, find the exact magnitude of the projection of u onto v.",
)
_MIN_GAP = 0.02  # well clear of the grader's 0.005 numeric window


def _fmt(v) -> str:
    return "(" + ", ".join(str(x) for x in v) + ")"


def _is_perfect_square(n: int) -> bool:
    r = int(n**0.5)
    return r * r == n or (r + 1) * (r + 1) == n


def _result(text: str, ans, cands) -> dict | None:
    key = float(ans)
    usable = [c for c in cands if abs(float(c) - key) > _MIN_GAP]
    distractors = pick_distractors(str(ans), [str(c) for c in usable])
    if len(distractors) < 2:
        return None  # caller redraws
    return dict(
        text=text,
        latex=text,
        answer=str(ans),
        answer_text=str(ans),
        answer_latex=str(ans).replace("sqrt", r"\sqrt"),
        answer_alt=round(key, 2),
        distractors=distractors,
    )


def _angle(rng: random.Random) -> dict | None:
    while True:
        la, lb = rng.randint(2, 7), rng.randint(2, 7)
        deg = rng.choice([60, 120])
        p, q = rng.choice([1, 2, 3]), rng.choice([-3, -2, -1, 1, 2, 3])
        cos_sign = 1 if deg == 60 else -1
        n = p * p * la * la + q * q * lb * lb + p * q * la * lb * cos_sign  # 2 pq |a||b| cos = pq |a||b| (+-1)
        wrong_n = p * p * la * la + q * q * lb * lb - p * q * la * lb * cos_sign
        if not _is_perfect_square(n) and wrong_n > 0 and not _is_perfect_square(wrong_n):
            break
    # Independent route: build explicit plane vectors a = (|a|, 0), b = |b| (cos, sin) with sympy exact trig
    # values and take the norm of the combination.
    cos_t = Rational(cos_sign, 2)
    sin_t = sqrt(3) / 2
    comb = (p * la + q * lb * cos_t, q * lb * sin_t)
    ans = sqrt(Integer(n))
    check_equivalent(sqrt(comb[0] ** 2 + comb[1] ** 2), ans, "angle: coordinate route disagrees")
    cands = [
        sqrt(Integer(wrong_n)),  # sign of the cos term wrong (the supplementary angle)
        sqrt(Integer(p * p * la * la + q * q * lb * lb)),  # dropped the 2 a . b term
        Integer(abs(p) * la + abs(q) * lb),  # magnitudes added: |pa| + |qb|
        Integer(n),  # forgot the square root
    ]
    e = join_terms([(p, "a"), (q, "b")])
    text = rng.choice(_ANGLE).format(la=la, lb=lb, deg=deg, e=e)
    return _result(text, ans, cands)


def _projection(rng: random.Random) -> dict | None:
    while True:
        u = [rng.randint(-5, 5) for _ in range(3)]
        v = [rng.randint(-4, 4) for _ in range(3)]
        uv = sum(x * y for x, y in zip(u, v))
        vv = sum(y * y for y in v)
        if uv != 0 and vv != 0 and not _is_perfect_square(vv) and any(u):
            break
    ans = abs(uv) / sqrt(Integer(vv))
    # Independent route: project explicitly, (u.v / v.v) v, then take its norm.
    proj = [Rational(uv * y, vv) for y in v]
    check_equivalent(sqrt(sum(c * c for c in proj)), ans, "projection: explicit projection vector disagrees")
    uu = sum(x * x for x in u)
    cands = [
        Integer(abs(uv)),  # forgot to divide by |v|
        abs(uv) / Integer(vv),  # divided by |v|^2 (the scalar multiple of v, not its length)
        sqrt(Integer(uu)),  # reported |u|
        abs(uv) / sqrt(Integer(uu)),  # divided by |u| instead of |v|
    ]
    text = rng.choice(_PROJECTION).format(u=_fmt(u), v=_fmt(v))
    return _result(text, ans, cands)


@generator(
    "exd_x_g1_vector_magnitude",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_g1_vector_magnitude(rng: random.Random, tier: str = "exam") -> dict:
    """|pa + qb| from lengths and an angle, or the length of a vector projection."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in range(200):
        built = _angle(rng) if rng.random() < 0.5 else _projection(rng)
        if built is not None:
            return built
    raise ValueError("no usable draw in 200 attempts")
