"""Exam chain for linear transformations: a reflection in an unknown line through the origin (three parts).

The stem says a reflection in a line through the origin maps ``P`` to ``P'`` and asks nothing about the line directly.

(a) the gradient of the mirror line (it passes through the midpoint of ``PP'``, so ``m = (y + y') / (x + x')``);
(b) the x-coordinate of the image of a second point ``Q`` (follow-through from ``part_a`` through the reflection matrix);
(c) the y-coordinate of the image of ``Q`` (follow-through from ``part_a``).

The reflection matrix for a line of gradient ``m`` is ``1/(1 + m^2) [[1 - m^2, 2m], [2m, m^2 - 1]]``. Points are built
from a whole-number step along the line, ``u = (q, p)``, and one across it, ``w = (-p, q)``: ``P = a u + b w`` has image
``a u - b w``, so every given and asked coordinate is a whole number.
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
    paren_negative,
    pick_distractors,
    rational_text,
)

_STEM = (
    "A reflection in a line through the origin maps the point P{p} to P'{p1}.",
    "The point P{p} is reflected in a line that passes through the origin, and its image is P'{p1}.",
    "Under a reflection in a line through the origin, the image of the point P{p} is P'{p1}.",
)
_ASK_X = (
    "Hence find the x-coordinate of the image of the point Q{q} under the same reflection.",
    "Using this reflection, find the x-coordinate of the image of Q{q}.",
)
_ASK_Y = (
    "Hence find the y-coordinate of the image of Q{q}.",
    "Find the y-coordinate of the image of Q{q} under the same reflection.",
)
_GRADIENTS = (
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (-1, 2),
    (-2, 1),
    (-1, 3),
    (-3, 1),
    (-2, 3),
    (-3, 2),
)


def _pt(x, y) -> str:
    return f"({x}, {y})"


def _reflect(m: Fraction, x: int, y: int) -> tuple[Fraction, Fraction]:
    d = 1 + m * m
    return ((1 - m * m) * x + 2 * m * y) / d, (2 * m * x + (m * m - 1) * y) / d


def _rotate(m: Fraction, x: int, y: int) -> tuple[Fraction, Fraction]:
    """The rotation through twice the angle of the line: the matrix a learner uses by mistake."""
    d = 1 + m * m
    return ((1 - m * m) * x - 2 * m * y) / d, (2 * m * x + (1 - m * m) * y) / d


def _frac_part(prompt, key: Fraction, marks, working, pairs, ft=None):
    cands = [Distractor(rational_text(v), tag, exact_value(v)) for tag, v in pairs if v != key]
    out = pick_distractors(rational_text(key), cands, 3)
    check(len(out) >= 2, "too few distractors")
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


def _wrong_images(m: Fraction, qx: int, qy: int, label: int) -> list:
    out = []
    out.append(("confused-reflection-with-rotation", _rotate(m, qx, qy)[label]))
    mm = Fraction(1) / m
    out.append(("inverted-gradient", _reflect(mm, qx, qy)[label]))
    raw = ((1 - m * m) * qx + 2 * m * qy, 2 * m * qx + (m * m - 1) * qy)
    out.append(("forgot-the-normalising-factor", raw[label]))
    mp = -1 / m
    out.append(("used-perpendicular-gradient", _reflect(mp, qx, qy)[label]))
    return out


def _enough(pairs, key) -> bool:
    return len({v for _, v in pairs if v != key}) >= 2


@generator("exd_x_r2d_lt_reflection_gradient", level=12, presets={"exam": dict(tier="exam")})
def exd_x_r2d_lt_reflection_gradient(rng: random.Random, tier: str = "exam") -> dict:
    """The mirror line from a point and its image, then the image of a second point."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a reflection with whole-number points"):
        p, q = rng.choice(_GRADIENTS)
        a, b, g, dl = (rng.choice((-3, -2, -1, 1, 2, 3)) for _ in range(4))
        px, py = a * q - b * p, a * p + b * q
        p1x, p1y = a * q + b * p, a * p - b * q
        qx, qy = g * q - dl * p, g * p + dl * q
        if max(abs(v) for v in (px, py, p1x, p1y, qx, qy)) > 16 or (qx, qy) in {(px, py), (p1x, p1y)}:
            continue
        if 0 in (qx, qy):  # keep both coordinates of Q in play
            continue
        m = Fraction(p, q)
        x1, y1 = _reflect(m, qx, qy)
        if 0 not in (x1, y1) and all(_enough(_wrong_images(m, qx, qy, i), v) for i, v in enumerate((x1, y1))):
            break
    check(m == Fraction(py + p1y, px + p1x), "the midpoint of PP' is not on the line of gradient m")
    check(Fraction(py - p1y, px - p1x) * m == -1, "PP' is not perpendicular to the mirror")
    check(x1.denominator == 1 and y1.denominator == 1, "the image of Q is not a whole-number point")
    check(x1 == g * q + dl * p and y1 == g * p - dl * q, "reflection matrix disagrees with the construction")

    def ft_x(mm="part_a") -> str:
        return f"((1 - {mm}**2)*({qx}) + 2*{mm}*({qy}))/(1 + {mm}**2)"

    def ft_y(mm="part_a") -> str:
        return f"(2*{mm}*({qx}) + ({mm}**2 - 1)*({qy}))/(1 + {mm}**2)"

    slope_pp = Fraction(py - p1y, px - p1x)  # the slope of PP', perpendicular to the mirror
    a_pairs = [
        ("used-perpendicular-gradient", slope_pp),
        ("inverted-gradient", Fraction(q, p)),
    ]

    stem = rng.choice(_STEM).format(p=_pt(px, py), p1=_pt(p1x, p1y))
    working_b = [
        f"The reflection matrix is 1/(1 + m²) [[1 − m², 2m], [2m, m² − 1]] with m = {rational_text(m)}.",
        f"The x-coordinate of the image is {x1.numerator}.",
    ]
    return dict(
        text=stem,
        parts=[
            _frac_part(
                "Find the gradient of the mirror line.",
                m,
                2,
                [
                    "The mirror passes through the midpoint of PP' and through the origin.",
                    f"m = ({py} + {paren_negative(p1y)}) ÷ ({px} + {paren_negative(p1x)}) = {rational_text(m)}.",
                ],
                a_pairs,
            ),
            _frac_part(
                rng.choice(_ASK_X).format(q=_pt(qx, qy)),
                x1,
                2,
                working_b,
                _wrong_images(m, qx, qy, 0),
                ft=ft_x(),
            ),
            _frac_part(
                rng.choice(_ASK_Y).format(q=_pt(qx, qy)),
                y1,
                2,
                [f"The y-coordinate of the image is {y1.numerator}."],
                _wrong_images(m, qx, qy, 1),
                ft=ft_y(),
            ),
        ],
    )
