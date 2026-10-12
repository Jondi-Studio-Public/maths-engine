"""Generates `exd_x_r3b_nth_roots`: a multi-part exam question on the n-th roots of a complex number in Cartesian form.

The stem gives ``w = R (cos t + i sin t)`` as ``X + Yi`` with surd coordinates, ``R = r^n`` for an even integer ``r``
and ``t`` a multiple of 30 or 45 degrees that is a multiple of ``n``, so every argument below is a whole number
of degrees. The chain is the one a Specialist exam sets:

(a) |w|                                        number, 1 mark
(b) Arg w, 0 <= t < 360                         number, 1 mark
(c) the modulus of each n-th root               number, 1 mark,  part_a^(1/n)
(d) the argument of the root in a given quadrant number, 2 marks, (part_b + 360 k) / n

Every key is checked against `cmath`: the n roots are generated numerically from the stated Cartesian w and the
one whose argument lies in the stated quadrant must have the stated modulus and argument.
"""

from __future__ import annotations

import cmath
import math
import random
from fractions import Fraction

from ..core import attempts, check, generator, phrase
from .exd_x_r3b_common import number_part
from ._exd_r3_distinct import distinct_part_answers

_ANGLES = (30, 45, 60, 120, 135, 150, 210, 225, 240, 300, 315, 330)
_ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth"}
_ROOT_NAME = {3: "cube", 4: "fourth"}
_MODULI = (2, 4, 6, 8)

_STEMS = (
    "Let w = {w}. The {root} roots of w are plotted in the complex plane.",
    "The complex number w is given by w = {w}. Consider the {root} roots of w.",
    "Consider the complex number w = {w} and its {root} roots.",
)


# (cos, sin) of the acute reference angle as (rational coefficient, radicand); radicand 1 means rational.
_TRIG = {
    30: ((Fraction(1, 2), 3), (Fraction(1, 2), 1)),
    45: ((Fraction(1, 2), 2), (Fraction(1, 2), 2)),
    60: ((Fraction(1, 2), 1), (Fraction(1, 2), 3)),
}


def _coordinate(sign: int, coeff: Fraction, radicand: int) -> str:
    """An exact coordinate such as -8 or 8√3, written for a prompt (the coefficient is a whole number)."""
    check(coeff.denominator == 1, "coordinate coefficient is not whole")
    k = int(coeff)
    body = str(k) if radicand == 1 else f"{'' if k == 1 else k}√{radicand}"
    return ("-" if sign < 0 else "") + body


def _cartesian_text(x_parts, y_parts) -> str:
    x_text = _coordinate(*x_parts)
    y_text = _coordinate(*y_parts)
    imag = y_text.lstrip("-")
    imag = f"({imag})i" if "√" in imag else f"{imag}i"
    return f"{x_text} {'-' if y_text.startswith('-') else '+'} {imag}"


def _quadrant(angle: int) -> int:
    return angle // 90 + 1


def _build(rng: random.Random) -> dict:
    n = rng.choice([3, 3, 4])
    for _ in attempts(500, "an argument divisible by n and a quadrant holding one root"):
        theta = rng.choice(_ANGLES)
        if theta % n:
            continue
        args = [(theta + 360 * k) // n for k in range(n)]
        singles = [
            (k, a) for k, a in enumerate(args) if a % 90 != 0 and sum(_quadrant(b) == _quadrant(a) for b in args) == 1
        ]
        if singles:
            break  # noqa: PLW2901
    k, arg = rng.choice(singles)
    r = rng.choice(_MODULI)
    big = r**n
    ref_angle = {1: theta, 2: 180 - theta, 3: theta - 180, 4: 360 - theta}[_quadrant(theta)]
    cos_part, sin_part = _TRIG[ref_angle]
    x_sign = 1 if theta < 90 or theta > 270 else -1
    y_sign = 1 if theta < 180 else -1
    x_parts = (x_sign, big * cos_part[0], cos_part[1])
    y_parts = (y_sign, big * sin_part[0], sin_part[1])
    x_value = x_sign * float(big * cos_part[0]) * math.sqrt(cos_part[1])
    y_value = y_sign * float(big * sin_part[0]) * math.sqrt(sin_part[1])
    quadrant = _quadrant(arg)

    # Independent route: the n roots from the printed Cartesian number, by cmath.
    w = complex(x_value, y_value)
    check(abs(abs(w) - big) < 1e-9 and abs(math.degrees(cmath.phase(w)) % 360 - theta) < 1e-9, "w disagrees")
    roots = [abs(w) ** (1 / n) * cmath.exp(1j * (cmath.phase(w) + 2 * math.pi * j) / n) for j in range(n)]
    in_quadrant = [z for z in roots if _quadrant(round(math.degrees(cmath.phase(z)) % 360)) == quadrant]
    check(len(in_quadrant) == 1, "the quadrant does not hold exactly one root")
    check(abs(abs(in_quadrant[0]) - r) < 1e-9, "modulus of the root disagrees")
    check(abs(math.degrees(cmath.phase(in_quadrant[0])) % 360 - arg) < 1e-9, "argument of the root disagrees")

    wt = _cartesian_text(x_parts, y_parts)
    stem = phrase(rng, _STEMS, w=wt, root=_ROOT_NAME[n])
    ref = ref_angle
    part_a = number_part(
        "Find |w|.",
        Fraction(big),
        0,
        1,
        [
            f"|w|^2 = ({_coordinate(*x_parts)})^2 + ({_coordinate(*y_parts).lstrip('-')})^2 = {big * big}.",
            f"|w| = {big}.",
        ],
        [("forgot-square-root-in-modulus", Fraction(big * big)), ("reported-root-modulus-for-w", Fraction(r))],
        sensible_range=(0, None),
    )
    part_b = number_part(
        "Find Arg w in degrees, where 0 ≤ Arg w < 360.",
        Fraction(theta),
        0,
        1,
        [f"w is in quadrant {_quadrant(theta)}, with reference angle {ref} degrees.", f"Arg w = {theta} degrees."],
        [
            ("argument-from-reference-angle-without-quadrant", Fraction(ref)),
            ("argument-in-opposite-quadrant", Fraction((theta + 180) % 360)),
            ("argument-of-conjugate-taken", Fraction(360 - theta)),
        ],
        follow_through=None,
    )
    part_c = number_part(
        f"Find the modulus of each {_ROOT_NAME[n]} root of w.",
        Fraction(r),
        0,
        1,
        [f"The modulus of each root is {big}^(1/{n}) = {r}."],
        [
            ("divided-modulus-by-n-not-root", Fraction(big, n)),
            ("kept-modulus-of-w", Fraction(big)),
            ("used-square-root-for-nth-root", Fraction(round(math.sqrt(big)))),
        ],
        follow_through=f"part_a**(1/{n})",
        sensible_range=(0, None),
    )
    part_d = number_part(
        f"One {_ROOT_NAME[n]} root of w lies in the {_ORDINAL[quadrant]} quadrant. "
        "Find its argument in degrees, where 0 ≤ θ < 360.",
        Fraction(arg),
        0,
        2,
        [
            f"The arguments of the roots are (θ + 360k)/{n} for k = 0, 1, ..., {n - 1}.",
            f"The root in the {_ORDINAL[quadrant]} quadrant has k = {k}: ({theta} + {360 * k})/{n} = {arg}.",
        ],
        [
            ("dropped-the-full-turn-term", Fraction(theta, n)),
            ("added-full-turn-after-dividing", Fraction(theta, n) + 360 * k),
            ("chose-root-in-wrong-quadrant", Fraction(args[(k + 1) % n])),
            ("chose-root-in-wrong-quadrant", Fraction(args[(k - 1) % n])),
        ],
        follow_through=f"(part_b + {360 * k})/{n}",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r3b_nth_roots", level=15, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3b_nth_roots(rng: random.Random, tier: str = "exam") -> dict:
    """A complex number in Cartesian form: its modulus and argument, then the root in a stated quadrant."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
