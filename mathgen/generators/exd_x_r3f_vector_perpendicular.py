"""Generates `exd_x_r3f_vector_perpendicular` questions: an exam chain on a perpendicularity condition.

Two vectors u and v are given, with w = 2u - v and p = u + kv. The chain is the working a vector paper sets out:

(a) the dot product u . w                                  number, 1 mark
(b) the dot product v . w                                  number, 1 mark
(c) the value of k for which p is perpendicular to w       number, 2 marks   k = -(u . w)/(v . w)
(d) |p|^2 for that value of k                               number, 2 marks

Part (c) follows through from (a) and (b), and part (d) from (c), so a slip in a dot product costs only the
part it was made in. The vectors are drawn so that k is a non-zero integer.
"""

from __future__ import annotations

import random

from ..core import attempts, check, generator, phrase
from ._exd_r3f_parts import number_part
from ._exd_r3_distinct import distinct_part_answers

_STEMS = (
    "The vectors u = ({u1}, {u2}) and v = ({v1}, {v2}) are given. Let w = 2u - v and p = u + kv, "
    "where k is a constant.",
    "Let u = ({u1}, {u2}) and v = ({v1}, {v2}). The vector w is defined by w = 2u - v, and p = u + kv for a "
    "constant k.",
    "Two vectors are u = ({u1}, {u2}) and v = ({v1}, {v2}). Define w = 2u - v and p = u + kv, where k is a "
    "real constant.",
)


def _dot(p, q) -> int:
    return p[0] * q[0] + p[1] * q[1]


@generator("exd_x_r3f_vector_perpendicular", level=17, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3f_vector_perpendicular(rng: random.Random, tier: str = "exam") -> dict:
    """Dot products, then the k that makes u + kv perpendicular to 2u - v, then the length squared of u + kv."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(5000, "vectors with an integer, non-zero k"):
        u = (rng.randint(-6, 6), rng.randint(-6, 6))
        v = (rng.randint(-6, 6), rng.randint(-6, 6))
        if u == (0, 0) or v == (0, 0) or u[0] * v[1] == u[1] * v[0]:
            continue
        w = (2 * u[0] - v[0], 2 * u[1] - v[1])
        uw, vw = _dot(u, w), _dot(v, w)
        if vw == 0 or uw == 0 or uw % vw != 0:
            continue
        k = -uw // vw
        if k != 0 and abs(k) <= 12:
            break
    p = (u[0] + k * v[0], u[1] + k * v[1])
    check(_dot(p, w) == 0, "p is not perpendicular to w")
    p_sq = _dot(p, p)

    def dot_candidates(x):
        return [
            (x[0] * w[1] + x[1] * w[0], "dot-product-multiplied-across"),
            (x[0] * w[0] - x[1] * w[1], "dot-product-subtracted-components"),
            (_dot(x, (2 * u[0] + v[0], 2 * u[1] + v[1])), "combination-sign-slip"),
            (_dot(x, (u[0] - 2 * v[0], u[1] - 2 * v[1])), "scalars-attached-to-wrong-vectors"),
        ]

    def exact_or_none(num, den):
        return num // den if den != 0 and num % den == 0 else None

    k_candidates = [
        (-k, "solved-for-scalar-with-wrong-sign"),
        (-(uw - vw), "subtracted-instead-of-divided"),
        (exact_or_none(-vw, uw), "ratio-inverted-solving-for-k"),
        (exact_or_none(-uw, _dot(v, v)), "perpendicular-condition-used-wrong-vector"),
        (exact_or_none(-_dot(u, v), _dot(v, v)), "perpendicular-condition-used-wrong-vector"),
    ]
    sq = lambda q: _dot(q, q)  # noqa: E731
    p_candidates = [
        (sq((u[0] + v[0], u[1] + v[1])), "forgot-scalar-multiple"),
        (sq((u[0] + k, u[1] + k)), "scalar-added-not-multiplied"),
        (sq((u[0] - k * v[0], u[1] - k * v[1])), "combination-sign-slip"),
        (sq((k * v[0], k * v[1])), "perpendicular-dropped-first-vector"),
        (sq(u) + k * k * sq(v), "squared-sum-without-cross-term"),
    ]

    stem = phrase(rng, _STEMS, u1=u[0], u2=u[1], v1=v[0], v2=v[1])
    return dict(
        text=stem,
        parts=[
            number_part(
                "Find the dot product u \u00b7 w.",
                uw,
                dot_candidates(u),
                marks=1,
                working=[f"w = 2u - v = ({w[0]}, {w[1]}), so u \u00b7 w = {uw}."],
            ),
            number_part(
                "Find the dot product v \u00b7 w.",
                vw,
                dot_candidates(v),
                marks=1,
                working=[f"v \u00b7 w = ({v[0]})({w[0]}) + ({v[1]})({w[1]}) = {vw}."],
            ),
            number_part(
                "Hence find the value of k for which p is perpendicular to w.",
                k,
                k_candidates,
                marks=2,
                follow_through="-part_a/part_b",
                working=[f"p \u00b7 w = u \u00b7 w + k(v \u00b7 w) = {uw} + {vw}k = 0, so k = {k}."],
            ),
            number_part(
                "Find the value of |p|\u00b2 for this value of k.",
                p_sq,
                p_candidates,
                marks=2,
                sensible_range=(0, None),
                follow_through=f"({u[0]}+part_c*({v[0]}))**2+({u[1]}+part_c*({v[1]}))**2",
                working=[f"p = ({p[0]}, {p[1]}), so |p|\u00b2 = {p_sq}."],
            ),
        ],
    )
