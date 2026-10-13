"""Exam source for `complex-imaginary-part`: a cube taken through its square.

The stem gives z = a + bi. Part (a) is the real part of z^2, part (b) its imaginary part, and part (c) the imaginary
part of z^3, found as z^2 times z from the two parts already found: Im(z^3) = Re(z^2) b + Im(z^2) a. Part (c) carries
`follow_through`, so a slip in the square is paid for once.

The key is checked against SymPy's expansion of (a + bi)^3, which does not use the square.
"""

from __future__ import annotations

import sympy

from mathgen.core import attempts, check, generator, join_terms, paren_negative, phrase

from .exd_x_r3c_parts import number_part
from ._exd_distinct_parts import distinct_part_answers

_STEMS = (
    "Let z = {z}.",
    "The complex number z is given by z = {z}.",
    "Consider the complex number z = {z}.",
)


@generator("exd_x_r3c_complex_cube", level=14, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3c_complex_cube(rng, tier="exam"):
    """z = a + bi: Re(z^2), Im(z^2), then Im(z^3)."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    values = [v for v in range(-8, 9) if v]
    for _ in attempts(200, "a complex number whose square has a non-zero real part"):
        a, b = rng.choice(values), rng.choice(values)
        if abs(a) != abs(b):
            break
    re2, im2 = a * a - b * b, 2 * a * b
    im3 = re2 * b + im2 * a
    cube = sympy.expand((a + b * sympy.I) ** 3)
    check(sympy.im(cube) == im3 and sympy.re(sympy.expand((a + b * sympy.I) ** 2)) == re2, "expansion disagrees")
    check(sympy.im(sympy.expand((a + b * sympy.I) ** 2)) == im2, "expansion disagrees")

    z = join_terms([(a, ""), (b, "i")])
    stem = phrase(rng, _STEMS, z=z)
    part_a = number_part(
        "Find the real part of z².",
        re2,
        working=[
            f"(a + bi)² = a² - b² + 2abi. Here a = {a} and b = {b}, so z² = {join_terms([(re2, ''), (im2, 'i')])}."
        ],
        wrongs=[
            (a * a + b * b, "i-squared-as-plus-one"),
            (b * b - a * a, "reversed-subtraction"),
            (a * a, "dropped-i-squared-term"),
        ],
    )
    part_b = number_part(
        "Find the imaginary part of z².",
        im2,
        working=[f"Im(z²) = 2ab = 2 × {paren_negative(a)} × {paren_negative(b)} = {im2}."],
        wrongs=[
            (a * b, "cross-term-not-doubled"),
            (-im2, "complex-product-sign-slip"),
            (b * b, "squared-each-part-separately"),
        ],
    )
    part_c = number_part(
        "Using your z² from (a) and (b), find the imaginary part of z³.",
        im3,
        marks=2,
        working=[
            f"z³ = z² × z = ({join_terms([(re2, ''), (im2, 'i')])})({z}).",
            f"Im(z³) = Re(z²) × b + Im(z²) × a = {paren_negative(re2)} × {paren_negative(b)} + {paren_negative(im2)} × {paren_negative(a)} = {im3}.",
        ],
        wrongs=[
            (3 * a * a * b + b**3, "i-cubed-as-plus-i"),
            (a**3 - 3 * a * b * b, "swapped-real-and-imaginary-parts"),
            (3 * a * a * b, "dropped-i-cubed-term"),
        ],
        follow_through=f"part_a*({b}) + part_b*({a})",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c])
