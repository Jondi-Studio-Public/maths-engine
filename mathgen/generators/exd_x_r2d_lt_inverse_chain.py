"""Exam chain for inverses of linear transformations: undo a map and reverse an area (four parts).

``T`` maps ``(x, y)`` to ``(a x + b y, c x + d y)`` and maps an unknown point ``P`` to a stated image ``Q``.

(a) the determinant of the matrix of ``T``;
(b) the x-coordinate of ``P`` (follow-through: the adjugate row over ``part_a``);
(c) the y-coordinate of ``P`` (follow-through, same shape);
(d) the area of a region whose image under ``T`` has a stated area (follow-through ``image area / |part_a|``).

``P`` is drawn first and ``Q = T(P)`` derived, so ``P`` is a whole-number point. The determinant is never ``+1`` or
``-1``, so the division by it is a real step, and its sign varies so the area factor has to be taken as ``|det|``.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    attempts,
    generator,
    join_terms,
    latex_answer,
    pick_distractors,
    rational_text,
)

from ._exd_distinct_parts import distinct_part_answers

_REGIONS = ("triangle", "rectangle", "parallelogram", "trapezium")
_STEM = (
    "A linear transformation T maps (x, y) to ({row1}, {row2}). T maps the point P to the point Q{q}.",
    "The linear transformation T is given by (x, y) → ({row1}, {row2}). It maps a point P to the image Q{q}.",
)


class _Redraw(Exception):
    """This draw leaves too few distractors for some part; draw again."""


def _pt(x, y) -> str:
    return f"({x}, {y})"


def _frac_part(prompt, key: Fraction, marks, working, pairs, ft=None, minimum: int = 2):
    cands = [Distractor(rational_text(v), tag, exact_value(v)) for tag, v in pairs if v != key]
    out = pick_distractors(rational_text(key), cands, 3)
    if len(out) < minimum:
        raise _Redraw
    part = dict(
        prompt=prompt,
        answer_kind="fraction",
        answer=exact_value(key),
        answer_text=rational_text(key),
        answer_latex=latex_answer(rational_text(key)),
        marks=marks,
        working=working,
        distractors=out,
    )
    if ft:
        part["follow_through"] = ft
    return part


@generator("exd_x_r2d_lt_inverse_chain", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2d_lt_inverse_chain(rng: random.Random, tier: str = "exam") -> dict:
    """The determinant of T, the original point from its image, and the original area from the image area."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a transformation with enough distractors"):
        try:
            return _build(rng)
        except _Redraw:
            continue


@distinct_part_answers
def _build(rng: random.Random) -> dict:
    for _ in attempts(2000, "an invertible matrix with |det| > 1"):
        a, b, c, d = (rng.randint(-4, 4) for _ in range(4))
        det = a * d - b * c
        if abs(det) < 2 or 0 in (a, b, c, d) or abs(det) > 14:
            continue
        px, py = rng.randint(-6, 6), rng.randint(-6, 6)
        if 0 in (px, py):
            continue
        qx, qy = a * px + b * py, c * px + d * py
        if 0 in (qx, qy):
            continue
        break
    image_area = abs(det) * rng.randint(3, 12)
    shape = rng.choice(_REGIONS)
    # Independent route: the inverse matrix applied to Q, with exact fractions.
    inv = ((Fraction(d, det), Fraction(-b, det)), (Fraction(-c, det), Fraction(a, det)))
    back = (inv[0][0] * qx + inv[0][1] * qy, inv[1][0] * qx + inv[1][1] * qy)
    assert back == (px, py)
    original_area = Fraction(image_area, abs(det))

    adj = (d * qx - b * qy, -c * qx + a * qy)  # the adjugate applied to Q: P times the determinant
    fwd = (a * qx + b * qy, c * qx + d * qy)  # T applied to Q instead of its inverse
    a_pairs = [
        ("determinant-order-reversed", Fraction(-det)),
        ("substitution-sign-error", Fraction(a * d + b * c)),
    ]
    b_pairs = [
        ("forward-map-for-inverse", Fraction(fwd[0])),
        ("forgot-to-divide-by-determinant", Fraction(adj[0])),
        ("determinant-order-reversed", Fraction(-px)),
        ("substitution-sign-error", Fraction(d * qx + b * qy, det)),
    ]
    c_pairs = [
        ("forward-map-for-inverse", Fraction(fwd[1])),
        ("forgot-to-divide-by-determinant", Fraction(adj[1])),
        ("determinant-order-reversed", Fraction(-py)),
        ("substitution-sign-error", Fraction(c * qx + a * qy, det)),
    ]
    d_pairs = [
        ("area-scaled-the-wrong-way", Fraction(image_area * abs(det))),
        ("signed-determinant-as-area-factor", Fraction(image_area, det)),
        ("forgot-area-scale-factor", Fraction(image_area)),
    ]
    stem = rng.choice(_STEM).format(
        row1=join_terms([(a, "x"), (b, "y")]), row2=join_terms([(c, "x"), (d, "y")]), q=_pt(qx, qy)
    )
    return dict(
        text=stem,
        parts=[
            _frac_part(
                "Find the determinant of the matrix of T.",
                Fraction(det),
                2,
                [f"The matrix of T is [[{a}, {b}], [{c}, {d}]].", f"det = ({a})({d}) − ({b})({c}) = {det}."],
                a_pairs,
            ),
            _frac_part(
                "Hence find the x-coordinate of P.",
                Fraction(px),
                2,
                [
                    f"The inverse is (1/{det}) [[{d}, {-b}], [{-c}, {a}]].",
                    f"x = ({d}·{qx} − ({b})·{qy}) ÷ {det} = {px}.",
                ],
                b_pairs,
                ft=f"(({d})*({qx}) - ({b})*({qy}))/part_a",
            ),
            _frac_part(
                "Hence find the y-coordinate of P.",
                Fraction(py),
                1,
                [f"y = (−({c})·{qx} + ({a})·{qy}) ÷ {det} = {py}."],
                c_pairs,
                ft=f"(-({c})*({qx}) + ({a})*({qy}))/part_a",
            ),
            _frac_part(
                f"The image of a {shape} R under T has area {image_area} square units. "
                "Find the area of R, in square units.",
                original_area,
                2,
                [
                    f"Area scale factor = |det| = {abs(det)}.",
                    f"Area of R = {image_area} ÷ {abs(det)} = {rational_text(original_area)}.",
                ],
                d_pairs,
                ft=f"{image_area}/Abs(part_a)",
            ),
        ],
    )
