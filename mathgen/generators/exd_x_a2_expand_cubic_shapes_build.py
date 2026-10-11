"""Exam-tier source for expanding a cubic from factors that are not three distinct linear factors.

The practice tiers multiply three LINEAR factors. The exam question hides the multiplication structure:
the learner must recognise a repeated factor, a perfect cube or a difference of squares inside a product and
choose how to expand it. Four shapes, drawn per seed:

- ``square_times_linear``: ``(px + q)^2 (rx + s)``
- ``cube``: ``(px + q)^3``
- ``quadratic_times_linear``: ``(x^2 + px + q)(rx + s)``
- ``difference_of_squares``: ``(x + a)(x - a)(bx + c)``

The key is assembled from coefficient lists multiplied by hand (a discrete convolution) and confirmed against
``sympy.expand`` as an independent second route. Every shape yields a true cubic (non-zero leading
coefficient); the distractors are named slips:

- ``square-without-middle``: ``(px + q)^2`` expanded as ``p^2x^2 + q^2``.
- ``cube-without-cross-terms``: ``(px + q)^3`` expanded as ``p^3x^3 + q^3``.
- ``sign-slip``: a sign error on the constant of the last factor.
- ``sum-of-squares``: ``(x + a)(x - a)`` expanded as ``x^2 + a^2``.
- ``dropped-product``: the ``x^2 * s`` product of the quadratic-times-linear expansion omitted.
"""

from __future__ import annotations

import random

from sympy import Symbol, expand

from ..core import (
    check,
    check_distinct,
    check_equivalent,
    generator,
    join_terms,
    latex_answer,
    phrase,
    pick_distractors,
)

_VARS = ("x", "y", "n", "t")

_PATTERNS = (
    "Expand {expr}.",
    "Expand and simplify {expr}.",
    "Write {expr} as a polynomial in descending powers.",
)


def _mul(a: list[int], b: list[int]) -> list[int]:
    """Product of two coefficient lists, highest power first."""
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def _poly_text(coefs: list[int], v: str) -> str:
    n = len(coefs) - 1
    terms = []
    for i, c in enumerate(coefs):
        power = n - i
        label = "" if power == 0 else (v if power == 1 else f"{v}^{power}")
        terms.append((c, label))
    return join_terms(terms)


def _expr(coefs: list[int], sym: Symbol):
    n = len(coefs) - 1
    return sum(c * sym ** (n - i) for i, c in enumerate(coefs))


def _lin_text(p: int, q: int, v: str) -> str:
    return join_terms([(p, v), (q, "")])


def _nz(rng: random.Random, lo: int, hi: int) -> int:
    return rng.choice([n for n in range(lo, hi + 1) if n != 0])


def _exam(rng: random.Random) -> dict:
    v = rng.choice(_VARS)
    sym = Symbol(v)
    shape = rng.choice(["square_times_linear", "cube", "quadratic_times_linear", "difference_of_squares"])
    if shape == "square_times_linear":
        p, q, r, s = _nz(rng, -3, 3), _nz(rng, -5, 5), _nz(rng, -3, 3), _nz(rng, -5, 5)
        text_expr = f"({_lin_text(p, q, v)})^2({_lin_text(r, s, v)})"
        key = _mul(_mul([p, q], [p, q]), [r, s])
        wrongs = [
            _mul([p * p, 0, q * q], [r, s]),  # square-without-middle
            _mul(_mul([p, q], [p, q]), [r, -s]),  # sign-slip
            _mul(_mul([p, q], [p, -q]), [r, s]),  # one sign flipped in the repeated factor
        ]
    elif shape == "cube":
        p, q = _nz(rng, -3, 3), _nz(rng, -4, 4)
        text_expr = f"({_lin_text(p, q, v)})^3"
        key = _mul(_mul([p, q], [p, q]), [p, q])
        wrongs = [
            [p**3, 0, 0, q**3],  # cube-without-cross-terms
            [p**3, 3 * p * q, 3 * p * q, q**3],  # middle coefficients written 3pq instead of 3p^2q and 3pq^2
            _mul(_mul([p, q], [p, q]), [p, -q]),  # one sign flipped
        ]
    elif shape == "quadratic_times_linear":
        p, q = _nz(rng, -5, 5), _nz(rng, -6, 6)
        r, s = _nz(rng, -3, 3), _nz(rng, -5, 5)
        text_expr = f"({join_terms([(1, f'{v}^2'), (p, v), (q, '')])})({_lin_text(r, s, v)})"
        key = _mul([1, p, q], [r, s])
        wrongs = [
            [r, p * r, q * r + p * s, q * s],  # the x^2 * s product forgotten
            _mul([1, p, q], [r, -s]),  # sign-slip
            _mul([1, p, -q], [r, s]),  # sign-slip in the quadratic factor
        ]
    else:
        a, b, c = _nz(rng, 1, 6), _nz(rng, -3, 3), _nz(rng, -6, 6)
        text_expr = f"({v} + {a})({v} - {a})({_lin_text(b, c, v)})"
        key = _mul([1, 0, -a * a], [b, c])
        wrongs = [
            _mul([1, 0, a * a], [b, c]),  # sum-of-squares
            _mul([1, 0, -a], [b, c]),  # a not squared
            _mul([1, 0, -a * a], [b, -c]),  # sign-slip
        ]
    text = phrase(rng, _PATTERNS, expr=text_expr)

    # independent second route: sympy parses the displayed factors afresh
    shown = {
        "square_times_linear": lambda: (p * sym + q) ** 2 * (r * sym + s),
        "cube": lambda: (p * sym + q) ** 3,
        "quadratic_times_linear": lambda: (sym**2 + p * sym + q) * (r * sym + s),
        "difference_of_squares": lambda: (sym + a) * (sym - a) * (b * sym + c),
    }[shape]()
    answer_expr = _expr(key, sym)
    check_equivalent(answer_expr, expand(shown), "hand-built cubic disagrees with sympy.expand")
    check(len(key) == 4 and key[0] != 0, "key is not a cubic")

    texts = []
    # a slip that happens to reproduce the key for these particular numbers is not a distractor
    wrongs = [w for w in wrongs if w != key]
    check(bool(wrongs), "no distractor differs from the key")
    for w in wrongs:
        check_distinct(answer_expr, _expr(w, sym), msg="distractor equals the key")
        texts.append(_poly_text(w, v))
    answer_text = _poly_text(key, v)
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_expr,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_expr),
        distractors=pick_distractors(answer_text, texts),
        answer_form="expanded",
    )


@generator("exd_x_a2_expand_cubic_shapes", level=11, presets={"exam": dict(tier="exam")})
def exd_x_a2_expand_cubic_shapes(rng: random.Random, tier: str = "exam") -> dict:
    """Expand a cubic given as a repeated factor, a perfect cube, a quadratic times a linear, or with a
    difference of squares inside."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
