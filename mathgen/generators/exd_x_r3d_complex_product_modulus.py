"""Exam source of `complex-modulus`: the modulus of a product, reached through its real and imaginary parts.

The practice tier takes the modulus of one number a + bi. An exam question sets two numbers z and w and asks for the
modulus of their product, in three connected parts:

(a) the real part of zw (a product of two binomials in i, with i squared equal to -1);
(b) the imaginary part of zw;
(c) the modulus of zw, from the two parts just found.

Part (c) follows through from (a) and (b). The numbers are built so that |z| = |w|: both are drawn from two different
ways of writing the same sum of two squares (65 = 1 + 64 = 16 + 49, say). Then |zw| = |z||w| is a whole number, so the
modulus of the product can be asked for as a number, and a learner who notices |z||w| gets there without expanding.
Neither part of the product is zero. An independent check multiplies the numbers with SymPy's imaginary unit.
"""

from __future__ import annotations

import math
import random

import sympy

from mathgen.core import attempts, check, generator, join_terms, latex_answer, phrase

from .exd_x_r3d_common import tagged
from ._exd_r3_distinct import distinct_part_answers


def _representations(limit: int) -> dict[int, list[tuple[int, int]]]:
    """norm -> the ways to write it as a² + b² with a >= b >= 1, for norms that have at least two."""
    found: dict[int, list[tuple[int, int]]] = {}
    for a in range(1, 20):
        for b in range(1, a + 1):
            if a * a + b * b <= limit:
                found.setdefault(a * a + b * b, []).append((a, b))
    return {n: reps for n, reps in found.items() if len(reps) >= 2}


_NORMS = _representations(260)

_STEMS = (
    "Let z = {z} and w = {w}.",
    "The complex numbers z and w are given by z = {z} and w = {w}.",
    "Given z = {z} and w = {w}, consider the product zw.",
    "Two complex numbers are z = {z} and w = {w}.",
)
_PROMPT_A = ("Find the real part of zw.", "Calculate Re(zw).")
_PROMPT_B = ("Find the imaginary part of zw.", "Calculate Im(zw).")
_PROMPT_C = ("Hence find the modulus of zw.", "Using parts (a) and (b), find |zw|.")


def _number(rng: random.Random, rep: tuple[int, int]) -> tuple[int, int]:
    """A Gaussian integer with the given sum of squares: the two parts in either order, with random signs."""
    a, b = rep if rng.random() < 0.5 else rep[::-1]
    return a * rng.choice([-1, 1]), b * rng.choice([-1, 1])


def _text(p: tuple[int, int]) -> str:
    return join_terms([(p[0], ""), (p[1], "i")])


@generator("exd_x_r3d_complex_product_modulus", level=15, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3d_complex_product_modulus(rng: random.Random, tier: str = "exam") -> dict:
    """Re(zw), Im(zw) and |zw| for two numbers of equal modulus."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    norms = sorted(_NORMS)
    for _ in attempts(1000, "a product with both parts non-zero"):
        norm = rng.choice(norms)
        rep_z, rep_w = rng.sample(_NORMS[norm], 2)
        z, w = _number(rng, rep_z), _number(rng, rep_w)
        a, b = z
        c, d = w
        re, im = a * c - b * d, a * d + b * c
        if re != 0 and im != 0:
            break
    modulus = math.isqrt(re * re + im * im)
    check(modulus * modulus == re * re + im * im, "the modulus of zw must be a whole number")
    check(modulus == norm, "|zw| must equal the common squared modulus of z and w")
    # Independent route: SymPy's imaginary unit.
    product = sympy.expand((a + b * sympy.I) * (c + d * sympy.I))
    check(
        sympy.re(product) == re and sympy.im(product) == im and sympy.Abs(product) == modulus,
        "SymPy disagrees with the product",
    )

    a_d = tagged(
        str(re),
        [
            (a * c + b * d, "i-squared-as-plus-one"),
            (a * c, "imaginary-terms-dropped-in-product"),
            (im, "swapped-real-and-imaginary-parts"),
            (a + c, "complex-parts-added-not-multiplied"),
        ],
    )
    b_d = tagged(
        str(im),
        [
            (a * d - b * c, "complex-product-sign-slip"),
            (a * d, "imaginary-terms-dropped-in-product"),
            (re, "swapped-real-and-imaginary-parts"),
            (b + d, "complex-parts-added-not-multiplied"),
        ],
    )
    c_d = tagged(
        str(modulus),
        [
            (re * re + im * im, "magnitude-without-square-root"),
            (abs(re) + abs(im), "magnitude-as-sum-of-components"),
            (2 * modulus, "moduli-squares-added"),
        ],
    )
    check(len(a_d) >= 3 and len(b_d) >= 3 and len(c_d) >= 2, "too few distractors")

    stem = phrase(rng, _STEMS, z=_text(z), w=_text(w))
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=rng.choice(_PROMPT_A),
                answer_kind="number",
                answer=re,
                answer_text=str(re),
                answer_latex=latex_answer(re),
                marks=2,
                working=[
                    f"zw = ({a}{'+' if b >= 0 else ''}{b}i)({c}{'+' if d >= 0 else ''}{d}i); Re = {a}·{c} - ({b})({d}) = {re}."
                ],
                distractors=a_d,
            ),
            dict(
                prompt=rng.choice(_PROMPT_B),
                answer_kind="number",
                answer=im,
                answer_text=str(im),
                answer_latex=latex_answer(im),
                marks=2,
                working=[f"Im = {a}·{d} + ({b})({c}) = {im}."],
                distractors=b_d,
            ),
            dict(
                prompt=rng.choice(_PROMPT_C),
                answer_kind="number",
                answer=modulus,
                answer_text=str(modulus),
                answer_latex=latex_answer(modulus),
                marks=2,
                follow_through="sqrt(part_a**2 + part_b**2)",
                sensible_range=(0, None),
                working=[f"|zw| = √({re}² + {im}²) = {modulus}. (Also |z||w| = √{norm}·√{norm} = {modulus}.)"],
                distractors=c_d,
            ),
        ],
    )
