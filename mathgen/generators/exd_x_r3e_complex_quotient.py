"""Exam-tier source for `complex-real-part`: a quotient of complex numbers, then a square.

z = u/v is built backwards from a chosen z = x + iy and a divisor v = c + di, with u = z v, so that every part is
a whole number:

(a) Re(z)                  number, 2 marks    x
(b) Im(z)                  number, 1 mark     y
(c) Re(z^2)                number, 2 marks    x^2 - y^2, from (a) and (b)      follow_through: part_a**2 - part_b**2

The real part of a square needs both parts of z, which is what makes (c) a real second step and not a restatement.
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..answers import Distractor
from ..core import check, generator, join_terms, phrase, pick_distractors
from ._exd_r3_distinct import distinct_part_answers

_RANGE = [n for n in range(-7, 8) if n != 0]
_DIVISOR = [n for n in range(-4, 5) if n != 0]


def _cx(re: int, im: int) -> str:
    return join_terms([(re, ""), (im, "i")])


def _plain(value: int) -> str:
    return str(int(value))


def _options(key: int, candidates, n: int = 3):
    """Distractors that are whole numbers different from the key, first `n` distinct."""
    out = []
    seen = {key}
    for tag, value in candidates:
        value = Fraction(value)
        if value.denominator != 1 or int(value) in seen:
            continue
        seen.add(int(value))
        out.append(Distractor(_plain(int(value)), tag, value=int(value)))
    return pick_distractors(_plain(key), out, n)


@generator("exd_x_r3e_complex_quotient", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_complex_quotient(rng, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in range(500):
        x, y = rng.choice(_RANGE), rng.choice(_RANGE)
        c, d = rng.choice(_DIVISOR), rng.choice(_DIVISOR)
        if abs(x) == abs(y):  # keeps x^2 - y^2 from collapsing to 0 and the squares apart
            continue
        a, b = x * c - y * d, x * d + y * c  # u = z v
        modulus_sq = c * c + d * d
        a_opts = _options(
            x,
            [
                ("swapped-real-and-imaginary-parts", y),
                ("forgot-to-divide-by-denominator-modulus", a * c + b * d),
                ("multiplied-by-denominator-not-conjugate", a * c - b * d),
                ("sign-error-in-imaginary-numerator", Fraction(a * c - b * d, modulus_sq)),
            ],
        )
        b_opts = _options(
            y,
            [
                ("sign-error-in-imaginary-numerator", -y),
                ("forgot-to-divide-by-denominator-modulus", b * c - a * d),
                ("multiplied-by-denominator-not-conjugate", a * d + b * c),
            ],
        )
        c_key = x * x - y * y
        c_opts = _options(
            c_key,
            [
                ("i-squared-as-plus-one", x * x + y * y),
                ("imaginary-part-of-square-as-real", 2 * x * y),
                ("ignored-imaginary-part-when-squaring", x * x),
            ],
        )
        if len(a_opts) == len(b_opts) == len(c_opts) == 3:
            break
    else:
        check(False, "no complex quotient with three distractors for every part")

    # Independent check with SymPy's own complex arithmetic.
    z = sympy.expand((sympy.Integer(a) + sympy.I * b) / (sympy.Integer(c) + sympy.I * d), complex=True)
    check(sympy.re(z) == x and sympy.im(z) == y, "quotient disagrees with SymPy")
    check(sympy.re(sympy.expand(z**2)) == c_key, "Re(z^2) disagrees with SymPy")

    stem = phrase(
        rng,
        (
            "Let z = ({u})/({v}).",
            "The complex number z is given by z = ({u})/({v}).",
            "Consider the complex number z = ({u})/({v}), where i^2 = -1.",
        ),
        u=_cx(a, b),
        v=_cx(c, d),
    )
    conj = _cx(c, -d)
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Find Re(z), the real part of z.",
                answer_kind="number",
                answer=x,
                answer_text=_plain(x),
                marks=2,
                working=[
                    f"Multiply top and bottom by the conjugate {conj}: the denominator becomes {modulus_sq}.",
                    f"z = ({a * c + b * d} + {_cx(0, b * c - a * d)})/{modulus_sq} = {_cx(x, y)}, so Re(z) = {x}.",
                ],
                distractors=a_opts,
            ),
            dict(
                prompt="Find Im(z), the coefficient of i in z.",
                answer_kind="number",
                answer=y,
                answer_text=_plain(y),
                marks=1,
                working=[f"From z = {_cx(x, y)}, Im(z) = {y}."],
                distractors=b_opts,
            ),
            dict(
                prompt="Hence find Re(z^2), the real part of z squared.",
                answer_kind="number",
                answer=c_key,
                answer_text=_plain(c_key),
                marks=2,
                working=[f"z^2 = ({x})^2 - ({y})^2 + 2({x})({y})i, so Re(z^2) = {x * x} - {y * y} = {c_key}."],
                distractors=c_opts,
                follow_through="part_a**2 - part_b**2",
            ),
        ],
    )
