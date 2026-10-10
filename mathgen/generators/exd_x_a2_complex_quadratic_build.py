"""Exam-tier source for complex-conjugate solutions of real quadratic equations.

The practice tiers state a complex solution or ask for one coefficient. The exam question asks the learner to
BUILD a quadratic polynomial from complex information, choosing between direct work with the conjugate pair
and symmetric-function (sum and product) reasoning. Four shapes:

- ``root``: ``p + qi`` is one solution of ``ax^2 + bx + c = 0`` with real ``b`` and ``c`` and a stated
  leading coefficient ``a``; write the polynomial.
- ``scale``, ``shift``, ``square``: ``u`` and ``v`` are the solutions of a stated monic quadratic with
  negative discriminant; write the monic quadratic whose solutions are ``ku`` and ``kv``, ``u + t`` and
  ``v + t``, or ``u^2`` and ``v^2``. The sums and products of the new solutions are real numbers obtainable
  from the original coefficients without ever writing a complex number.

Built backwards from ``p`` and ``q`` (the original solutions are ``p +- qi``), so the key is exact. The
independent second route builds the new polynomial from the actual complex solutions with ``sympy`` and
compares.

Distractors are named slips: the product of a scaled pair multiplied by ``k`` instead of ``k^2``; a shift
applied once instead of twice to the sum; ``i^2`` treated as ``+1`` so ``p^2 - q^2`` replaces ``p^2 + q^2``;
the ``-2c`` forgotten when squaring; a sign error on the linear coefficient.
"""

from __future__ import annotations

import random

from sympy import I, Symbol, expand

from ..core import check, check_distinct, check_equivalent, generator, join_terms, latex_answer, pick_distractors

_X = Symbol("x")


def _poly_text(a: int, b: int, c: int) -> str:
    return join_terms([(a, "x^2"), (b, "x"), (c, "")])


def _poly_expr(a: int, b: int, c: int):
    return a * _X**2 + b * _X + c


def _monic_text(b: int, c: int) -> str:
    return _poly_text(1, b, c)


def _complex_text(p: int, q: int) -> str:
    imag = "i" if q == 1 else f"{q}i"
    return f"{p} + {imag}" if p != 0 else imag


def _exam(rng: random.Random) -> dict:
    while True:
        out = _draw(rng)
        if all(out["key"]):
            return out["question"]


def _draw(rng: random.Random) -> dict:
    shape = rng.choice(["root", "scale", "shift", "square"])
    p = rng.randint(-4, 4)
    q = rng.randint(1, 4)
    c0 = p * p + q * q  # product of the original solutions
    s0 = 2 * p  # their sum
    u, v = p + q * I, p - q * I

    if shape == "root":
        a = rng.choice([2, 3, 4, 5])
        # solutions of the real quadratic are p +- qi: sum 2p, product p^2 + q^2
        key = (a, -a * 2 * p, a * c0)
        text = (
            f"{_complex_text(p, q).replace('+ -', '- ')} is one solution of the equation ax^2 + bx + c = 0, "
            f"where a = {a} and b and c are real numbers. Write the polynomial ax^2 + bx + c."
        )
        wrong = [
            (a, -a * 2 * p, a * (p * p - q * q)),  # i^2 treated as +1
            (a, a * 2 * p, a * c0),  # sign error on the linear coefficient
            (a, -2 * p, c0),  # b and c not scaled by a
            (a, -a * 2 * p, a * p * p),  # imaginary part's square forgotten
        ]
        new = (u, v)
    else:
        given_eq = f"{_monic_text(-s0, c0)} = 0"
        if shape == "scale":
            k = rng.choice([2, 3, 4, -2, -3])
            new = (k * u, k * v)
            key = (1, -k * s0, k * k * c0)
            what = f"{k}u and {k}v"
            wrong = [
                (1, -k * s0, k * c0),  # product scaled by k instead of k^2
                (1, -s0, k * k * c0),  # sum left unscaled
                (1, k * s0, k * k * c0),  # sign error on the linear coefficient
            ]
        elif shape == "shift":
            t = rng.choice([1, 2, 3, -1, -2, -3])
            new = (u + t, v + t)
            key = (1, -(s0 + 2 * t), c0 + t * s0 + t * t)
            what = f"u {'+' if t > 0 else '-'} {abs(t)} and v {'+' if t > 0 else '-'} {abs(t)}"
            wrong = [
                (1, -(s0 + t), c0 + t * s0 + t * t),  # shift counted once in the sum
                (1, -(s0 + 2 * t), c0 + t),  # shift added to the product directly
                (1, -(s0 + 2 * t), c0 + t * t),  # cross term t(u + v) forgotten
                (1, -(s0 - 2 * t), c0 - t * s0 + t * t),  # shift in the wrong direction
            ]
        else:
            new = (u * u, v * v)
            key = (1, -(s0 * s0 - 2 * c0), c0 * c0)
            what = "u^2 and v^2"
            wrong = [
                (1, -(s0 * s0), c0 * c0),  # the -2uv forgotten when squaring the sum
                (1, -(s0 * s0 - 2 * c0), c0),  # product not squared
                (1, s0 * s0 - 2 * c0, c0 * c0),  # sign error on the linear coefficient
                (1, -(s0 * s0 + 2 * c0), c0 * c0),  # +2uv instead of -2uv
            ]
        text = (
            f"The solutions of {given_eq} are u and v. Write the monic quadratic polynomial "
            f"in x whose solutions are {what}."
        )

    # Independent second route: build the polynomial from the actual complex solutions.
    built = expand((_X - new[0]) * (_X - new[1]) * (key[0] if shape == "root" else 1))
    check_equivalent(built, _poly_expr(*key), "polynomial built from the complex solutions disagrees with the key")
    check(expand(built).as_poly(_X).degree() == 2, "key is not quadratic")

    answer_expr = _poly_expr(*key)
    answer_text = _poly_text(*key)
    texts = []
    for w in wrong:
        if w == key:
            continue
        check_distinct(answer_expr, _poly_expr(*w), msg="distractor equals the key")
        texts.append(_poly_text(*w))
    question = dict(
        text=text,
        latex=f"plain:{text}",
        answer=answer_expr,
        answer_text=answer_text,
        answer_latex=latex_answer(answer_expr),
        distractors=pick_distractors(answer_text, texts),
        answer_form="expanded",
    )
    return dict(key=key, question=question)


@generator("exd_x_a2_complex_quadratic", level=12, presets={"exam": dict(tier="exam")})
def exd_x_a2_complex_quadratic(rng: random.Random, tier: str = "exam") -> dict:
    """Write a real quadratic polynomial from complex solutions or from transformed complex solutions."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _exam(rng)
