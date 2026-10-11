"""Exam-tier questions on complex multiplication as a linear transformation of the plane.

The practice tiers always state the multiplier ``w`` and ask for the image of a point. The exam tier hides or
composes the multiplier, so the learner has to reason about the transformation itself:

* ``infer_apply``: an unknown multiplier takes ``z1`` to a stated image; find the image of a second point
  ``z2`` under the same transformation (recover ``w = z1'/z1`` by division, then apply it).
* ``find_multiplier``: the same information, but the question asks for ``w`` itself as the point ``(a, b)``.
* ``compose``: a point is multiplied by ``w1`` and the result by ``w2``; find the final image, which is the
  image under the single multiplier ``w1 * w2``.

Every answer is a pair of integers, re-derived with exact fractions (division uses the conjugate) before the
question is returned.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import check, generator, join_terms, phrase, pick_distractors, xy_solution

Z = tuple[int, int]


def _mul(w: Z, z: Z) -> Z:
    return (w[0] * z[0] - w[1] * z[1], w[0] * z[1] + w[1] * z[0])


def _divide(num: Z, den: Z) -> tuple[Fraction, Fraction]:
    """num / den using the conjugate of den: exact, no floating point."""
    norm = den[0] ** 2 + den[1] ** 2
    re = Fraction(num[0] * den[0] + num[1] * den[1], norm)
    im = Fraction(num[1] * den[0] - num[0] * den[1], norm)
    return re, im


def _fmt(z: Z) -> str:
    return join_terms([(z[0], ""), (z[1], "i")])


def _nonzero_point(rng: random.Random, lo: int = -4, hi: int = 4) -> Z:
    while True:
        z = (rng.randint(lo, hi), rng.randint(lo, hi))
        if z != (0, 0):
            return z


def _multiplier(rng: random.Random) -> Z:
    """A multiplier with both parts non-zero, so it is genuinely complex."""
    return (rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]), rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]))


def _pair(answer: Z, candidates: list[Z], text: str, working: list[str]) -> dict:
    block = xy_solution(*answer)
    wrong = [f"x = {c[0]}, y = {c[1]}" for c in candidates if c != answer]
    distractors = pick_distractors(block["answer_text"], wrong)
    check(len(distractors) >= 1, "no usable distractor")
    return {**block, "distractors": distractors, "text": text, "latex": text, "working": working}


def _conj(w: Z) -> Z:
    return (w[0], -w[1])


def _infer_apply(rng: random.Random) -> dict:
    while True:
        w = _multiplier(rng)
        z1, z2 = _nonzero_point(rng), _nonzero_point(rng)
        if z1 == z2:
            continue
        image1 = _mul(w, z1)
        answer = _mul(w, z2)
        cands = [
            (z2[0] + image1[0] - z1[0], z2[1] + image1[1] - z1[1]),  # the shift z1 -> image1 applied as a translation
            _mul(image1, z2),  # the image of z1 used as the multiplier
            _mul(_conj(w), z2),  # the conjugate multiplier
        ]
        if any(c != answer for c in cands):
            break
    recovered = _divide(image1, z1)
    check(recovered == (w[0], w[1]), "the multiplier is not recovered by division")
    text = phrase(
        rng,
        (
            "Multiplication by a fixed complex number w is a linear transformation of the complex plane. It maps "
            "z = {z1} to {im1}. Find the image of z = {z2} under the same transformation, as an ordered pair (x, y).",
            "A transformation of the complex plane multiplies every point by the same unknown complex number w, and "
            "takes {z1} to {im1}. Where does it take {z2}? Give the image as an ordered pair (x, y).",
        ),
        z1=_fmt(z1),
        im1=_fmt(image1),
        z2=_fmt(z2),
    )
    working = [
        f"w = ({_fmt(image1)}) divided by ({_fmt(z1)}) = {_fmt(w)}. Then w × ({_fmt(z2)}) = {_fmt(answer)}, "
        f"the point ({answer[0]}, {answer[1]})."
    ]
    return _pair(answer, cands, text, working)


def _find_multiplier(rng: random.Random) -> dict:
    while True:
        w = _multiplier(rng)
        z = _nonzero_point(rng)
        image = _mul(w, z)
        cands = [
            (w[1], w[0]),  # the real and imaginary parts swapped
            _conj(w),  # the conjugate
            (image[0] - z[0], image[1] - z[1]),  # the difference image - z, as if the map were an addition
        ]
        if any(c != w for c in cands):
            break
    recovered = _divide(image, z)
    check(recovered == (w[0], w[1]), "the multiplier is not recovered by division")
    text = phrase(
        rng,
        (
            "Multiplying the complex number z = {z} by a complex number w gives {im}. Find w, written as the point "
            "(x, y) that represents a + bi with x = a and y = b.",
            "Under multiplication by an unknown complex number w, the point {z} is mapped to {im}. Find w as an "
            "ordered pair (x, y), where w = x + yi.",
        ),
        z=_fmt(z),
        im=_fmt(image),
    )
    working = [
        f"w = ({_fmt(image)}) divided by ({_fmt(z)}), which simplifies to {_fmt(w)}: the point ({w[0]}, {w[1]})."
    ]
    return _pair(w, cands, text, working)


def _compose(rng: random.Random) -> dict:
    while True:
        w1, w2 = _multiplier(rng), _multiplier(rng)
        z = _nonzero_point(rng)
        answer = _mul(w2, _mul(w1, z))
        both = _mul(w1, w2)
        cands = [
            _mul(w2, z),  # only the second multiplication applied
            _mul((w1[0] + w2[0], w1[1] + w2[1]), z),  # the multipliers added instead of multiplied
            _mul((w1[0] * w2[0] + w1[1] * w2[1], w1[0] * w2[1] + w1[1] * w2[0]), z),  # sign slip in i squared
        ]
        if both != (0, 0) and any(c != answer for c in cands):
            break
    check(_mul(both, z) == answer, "composition must equal multiplication by the product")
    text = phrase(
        rng,
        (
            "The point z = {z} is multiplied by w1 = {w1} and the result is then multiplied by w2 = {w2}. Find the "
            "final image as an ordered pair (x, y).",
            "Two successive transformations of the complex plane are applied to z = {z}: first multiplication by "
            "{w1}, then multiplication by {w2}. Find the final point as an ordered pair (x, y).",
        ),
        z=_fmt(z),
        w1=_fmt(w1),
        w2=_fmt(w2),
    )
    working = [
        f"({_fmt(w1)})({_fmt(w2)}) = {_fmt(both)}, and ({_fmt(both)}) multiplied by ({_fmt(z)}) is {_fmt(answer)}: the "
        f"point ({answer[0]}, {answer[1]})."
    ]
    return _pair(answer, cands, text, working)


@generator(
    "exd_x_h2_complex_mult_transform",
    level=12,
    presets={"exam": dict(tier="exam")},
)
def exd_x_h2_complex_mult_transform(rng: random.Random, tier: str = "exam") -> dict:
    """Exam-tier complex multiplication: infer the multiplier, then apply it; or compose two multiplications."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return rng.choice((_infer_apply, _find_multiplier, _compose))(rng)
