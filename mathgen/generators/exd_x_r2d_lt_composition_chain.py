"""Exam chain for composing linear transformations: an unknown dilation factor (four parts).

``F`` is a rotation about the origin (given by ``cos`` and ``sin``) or a reflection in a line ``y = (p/q) x`` through
the origin. ``G`` is the dilation ``(x, y) -> (k x, m y)`` with ``m`` stated and ``k`` unknown. The composite ``G o F``
(``F`` first) maps a point ``P`` to a stated image ``P'``.

(a) find ``k`` (``P'`` has x-coordinate ``k`` times the x-coordinate of ``F(P)``);
(b) hence the determinant of the matrix of ``G o F`` (follow-through ``part_a * m * det F``);
(c) hence the area of the image of a region of stated area (follow-through ``area * |part_b|``);
(d) hence the x-coordinate of the image of a second point ``Q`` (follow-through ``part_a`` times the x-coordinate of
    ``F(Q)``).

Points are drawn so that ``F(P)`` and ``F(Q)`` have whole-number coordinates, so every given and asked value is a whole
number. The reflection matrix is ``1/(1 + g^2) [[1 - g^2, 2g], [2g, g^2 - 1]]`` for gradient ``g``.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.answers import Distractor, exact_value
from mathgen.core import (
    attempts,
    check,
    generator,
    latex_answer,
    pick_distractors,
    rational_text,
)

from ._exd_distinct_parts import distinct_part_answers

_TRIPLES = ((3, 4, 5), (4, 3, 5), (5, 12, 13), (12, 5, 13), (8, 15, 17), (15, 8, 17))
_GRADIENTS = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (-1, 2), (-2, 1), (-1, 3), (-3, 1))
_STEM = (
    "{f_text} Let G be the dilation (x, y) → (kx, {m}y), where k is a non-zero constant. "
    "The composite G ∘ F, in which F is applied first and then G, maps the point P{p} to P'{p1}.",
    "{f_text} The transformation G is the dilation (x, y) → (kx, {m}y), where k is a non-zero constant. "
    "Applying F and then G maps the point P{p} to the point P'{p1}.",
)
_REGIONS = ("triangle", "rectangle", "parallelogram", "kite")

Mat = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]


def _pt(x, y) -> str:
    return f"({x}, {y})"


def _apply(mat: Mat, x, y) -> tuple[Fraction, Fraction]:
    return mat[0][0] * x + mat[0][1] * y, mat[1][0] * x + mat[1][1] * y


def _frac_part(prompt, key: Fraction, marks, working, pairs, ft=None):
    cands = [Distractor(rational_text(v), tag, exact_value(v)) for tag, v in pairs if v != key]
    out = pick_distractors(rational_text(key), cands, 3)
    if len(out) < 2:
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


def _f_matrix(rng: random.Random) -> tuple[Mat, str, int]:
    """The matrix of F, a sentence describing it, and its determinant."""
    if rng.random() < 0.5:
        c, s, h = rng.choice(_TRIPLES)
        s = s * rng.choice((-1, 1))
        mat: Mat = ((Fraction(c, h), Fraction(-s, h)), (Fraction(s, h), Fraction(c, h)))
        text = f"Let F be the rotation about the origin through an angle θ, where cos θ = {c}/{h} and sin θ = {Fraction(s, h)}."
        return mat, text, 1
    p, q = rng.choice(_GRADIENTS)
    g = Fraction(p, q)
    d = 1 + g * g
    mat = (((1 - g * g) / d, 2 * g / d), (2 * g / d, (g * g - 1) / d))
    coef = "" if abs(p) == 1 else str(abs(p))
    line = "y = " + ("-" if p < 0 else "") + (f"{coef}x" if q == 1 else f"{coef}x/{q}")
    return mat, f"Let F be the reflection in the line {line}.", -1


class _Redraw(Exception):
    """This draw leaves too few distractors for some part; draw again."""


@generator("exd_x_r2d_lt_composition_chain", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2d_lt_composition_chain(rng: random.Random, tier: str = "exam") -> dict:
    """Find the unknown dilation factor of G in G o F, then its determinant, an area and a second image."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(200, "a composite with enough distractors"):
        try:
            return _build(rng)
        except _Redraw:
            continue


@distinct_part_answers
def _build(rng: random.Random) -> dict:
    for _ in attempts(4000, "a composite with whole-number points"):
        fmat, f_text, det_f = _f_matrix(rng)
        k = rng.choice((-5, -4, -3, -2, 2, 3, 4, 5))
        m = rng.choice((-4, -3, -2, 2, 3, 4))
        if abs(k) == abs(m):
            continue
        px, py = rng.randint(-12, 12), rng.randint(-12, 12)
        qx, qy = rng.randint(-12, 12), rng.randint(-12, 12)
        fp, fq = _apply(fmat, px, py), _apply(fmat, qx, qy)
        if any(v.denominator != 1 for v in (*fp, *fq)):
            continue
        if 0 in (fp[0], fp[1], fq[0]) or (px, py) == (qx, qy) or 0 in (px, py):
            continue
        if abs(fq[0]) == 1:  # (d) would be k again, or its negative: a follow-through that only restates (a)
            continue
        p1x, p1y = k * fp[0], m * fp[1]
        if max(abs(p1x), abs(p1y)) > 70:
            continue
        # k must not be recoverable by the order-reversed or first-map-ignored routes (their answers differ from k).
        rev = (p1x - fmat[0][1] * m * py) / (fmat[0][0] * px) if fmat[0][0] * px else None
        if rev is None or rev == k or Fraction(p1x, px) == k or Fraction(p1x) / fp[1] == k:
            continue
        break
    area = rng.randint(3, 15) * rng.choice((1, 2))
    shape = rng.choice(_REGIONS)
    det = k * m * det_f
    # Independent route: the composite matrix G F worked out entry by entry.
    comp = ((k * fmat[0][0], k * fmat[0][1]), (m * fmat[1][0], m * fmat[1][1]))
    check(comp[0][0] * comp[1][1] - comp[0][1] * comp[1][0] == det, "determinant disagrees with the composite matrix")
    check(_apply(comp, px, py) == (p1x, p1y), "the composite does not map P to P'")
    image_q = _apply(comp, qx, qy)
    check(image_q[0].denominator == 1 and image_q[0] == k * fq[0], "image of Q is not what G o F gives")
    area_key = area * abs(det)

    a_pairs = [
        ("ignored-first-transformation", Fraction(p1x, px)),
        ("composition-order-reversed", rev),
        ("used-wrong-component", Fraction(p1x) / fp[1]),
    ]
    b_pairs = [
        ("added-determinants", Fraction(k * m + det_f)),
        ("determinant-order-reversed", Fraction(-det)),
        ("added-determinants", Fraction(k + m)),
        ("ignored-first-transformation", Fraction(k * m)),
    ]
    c_pairs = [
        ("signed-determinant-as-area-factor", Fraction(area * det)),
        ("added-determinants", Fraction(area * (1 + abs(k * m)))),
        ("area-scaled-the-wrong-way", Fraction(area, abs(det))),
        ("added-determinants", Fraction(area * (abs(k) + abs(m)))),
    ]
    # G first, then F, for Q: F(G(Q)).
    rq = _apply(fmat, k * qx, m * qy)
    d_pairs = [
        ("composition-order-reversed", rq[0]),
        ("ignored-first-transformation", Fraction(k * qx)),
        ("used-wrong-component", k * fq[1]),
    ]

    stem = rng.choice(_STEM).format(f_text=f_text, m=m if m != 1 else "", p=_pt(px, py), p1=_pt(p1x, p1y))
    return dict(
        text=stem,
        parts=[
            _frac_part(
                "Find the value of k.",
                Fraction(k),
                3,
                [
                    f"F(P) = {_pt(*fp)}. G multiplies the x-coordinate by k, so {p1x} = k × {fp[0]}.",
                    f"k = {k}.",
                ],
                a_pairs,
            ),
            _frac_part(
                "Hence find the determinant of the matrix of the composite G ∘ F.",
                Fraction(det),
                2,
                [f"det(G ∘ F) = det G × det F = ({k} × {m}) × ({det_f}) = {det}."],
                b_pairs,
                ft=f"part_a*({m})*({det_f})",
            ),
            _frac_part(
                f"A {shape} of area {area} square units is transformed by G ∘ F. "
                "Hence find the area of its image, in square units.",
                Fraction(area_key),
                2,
                [f"Area scale factor = |det| = {abs(det)}.", f"Image area = {area} × {abs(det)} = {area_key}."],
                c_pairs,
                ft=f"{area}*Abs(part_b)",
            ),
            _frac_part(
                f"The point Q{_pt(qx, qy)} is also transformed by G ∘ F. Hence find the x-coordinate of its image.",
                image_q[0],
                2,
                [f"F(Q) = {_pt(*fq)}.", f"The x-coordinate of the image is {k} × {fq[0]} = {image_q[0]}."],
                d_pairs,
                ft=f"part_a*({fq[0]})",
            ),
        ],
    )
