"""Generates `exd_x_r3b_complex_conjugate`: a multi-part exam question on a quotient of complex numbers.

The stem gives ``z = (p + qi) / (r + si)``. The chain a senior exam sets is to work out the quotient's real and
imaginary parts through the modulus squared of the denominator, take the conjugate, and check with
``z z* = |z|^2``:

(a) |r + si|^2                                number,   1 mark
(b) the real part of z                        fraction, 2 marks, (p r + q s) / part_a
(c) the imaginary part of the conjugate of z  fraction, 2 marks, -(q r - p s) / part_a
(d) z multiplied by its conjugate             fraction, 2 marks, part_b^2 + part_c^2

Every key is checked against Python's complex arithmetic on exact `Fraction` parts, and (d) against the
independent value (p^2 + q^2) / (r^2 + s^2).
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator, phrase
from .exd_x_r3b_common import fraction_part, number_part
from ._exd_distinct_parts import distinct_part_answers

_STEMS = (
    "Let z = ({num})/({den}).",
    "The complex number z is given by z = ({num})/({den}).",
    "Consider the complex number z = ({num})/({den}).",
)


def cartesian(a: int, b: int) -> str:
    """``a + bi`` as written in a prompt: ``3 - 4i``, ``-2 + i``, ``1 - i`` (a and b both non-zero)."""
    imag = "i" if abs(b) == 1 else f"{abs(b)}i"
    return f"{a} {'-' if b < 0 else '+'} {imag}"


def _build(rng: random.Random) -> dict:
    for _ in attempts(1000, "a quotient with non-zero real and imaginary parts"):
        p, q = rng.randint(-7, 7), rng.randint(-7, 7)
        r, s = rng.randint(-5, 5), rng.randint(-5, 5)
        if 0 in (p, q, r, s):
            continue
        d = r * r + s * s
        re_num, im_num = p * r + q * s, q * r - p * s
        if re_num == 0 or im_num == 0 or d > 34:
            continue
        break
    x, y = Fraction(re_num, d), Fraction(im_num, d)
    # Independent route: exact complex division with real and imaginary parts as Fractions.
    z_conj_check = (complex(p, q) / complex(r, s)).conjugate()
    check(abs(z_conj_check.real - float(x)) < 1e-9 and abs(z_conj_check.imag + float(y)) < 1e-9, "quotient disagrees")
    product = x * x + y * y
    check(product == Fraction(p * p + q * q, d), "z times its conjugate disagrees with |num|^2 / |den|^2")

    num, den = cartesian(p, q), cartesian(r, s)
    stem = phrase(rng, _STEMS, num=num, den=den)
    part_a = number_part(
        f"Find |{den}|^2.",
        Fraction(d),
        0,
        1,
        [f"|{den}|^2 = {r}^2 + {s}^2 = {d}."],
        [
            ("i-squared-as-plus-one", Fraction(r * r - s * s)),
            ("square-of-sum-for-sum-of-squares", Fraction((r + s) ** 2)),
            ("added-parts-without-squaring", Fraction(abs(r) + abs(s))),
            ("added-parts-without-squaring", Fraction(r + s)),
            ("square-of-sum-for-sum-of-squares", Fraction((abs(r) + abs(s)) ** 2)),
            ("square-of-sum-for-sum-of-squares", Fraction((r - s) ** 2)),
        ],
        sensible_range=(0, None),
    )
    part_b = fraction_part(
        "Find the real part of z.",
        x,
        2,
        [f"z = ({num})({cartesian(r, -s)}) / {d}.", f"Re z = {re_num}/{d}."],
        [
            ("i-squared-as-plus-one", Fraction(p * r - q * s, d)),
            ("forgot-to-divide-by-denominator-modulus", Fraction(re_num)),
            ("swapped-real-and-imaginary-parts", y),
            ("divided-parts-separately", Fraction(p, r)),
        ],
        follow_through=f"({re_num})/part_a",
    )
    part_c = fraction_part(
        "Find the imaginary part of the conjugate of z.",
        -y,
        2,
        [f"Im z = {im_num}/{d}.", f"The conjugate of z has imaginary part {-im_num}/{d}."],
        [
            ("forgot-to-conjugate", y),
            ("cross-term-sign-error-in-division", -Fraction(q * r + p * s, d)),
            ("forgot-to-divide-by-denominator-modulus", Fraction(-im_num)),
            ("swapped-real-and-imaginary-parts", -x),
        ],
        follow_through=f"-({im_num})/part_a",
    )
    part_d = fraction_part(
        "Find the value of z multiplied by its conjugate.",
        product,
        2,
        [f"z times its conjugate = (Re z)^2 + (Im z)^2 = {p * p + q * q}/{d} after cancelling."],
        [
            ("i-squared-as-plus-one", x * x - y * y),
            ("added-conjugates-for-product", 2 * x),
            ("square-of-sum-for-sum-of-squares", (x + y) ** 2),
            ("multiplied-by-denominator-modulus", product * d),
        ],
        follow_through="part_b**2 + part_c**2",
        sensible_range=(0, None),
    )
    return dict(text=stem, parts=[part_a, part_b, part_c, part_d])


@generator("exd_x_r3b_complex_conjugate", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3b_complex_conjugate(rng: random.Random, tier: str = "exam") -> dict:
    """The quotient of two complex numbers, its conjugate and the product of the two."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _build(rng)
