"""Exam source for integrals that reduce to 1/sqrt(a^2 - u^2) or a/(a^2 + u^2) after completing the square.

The integrand is given with an EXPANDED quadratic, so the learner must first recognise a shifted standard form:

    k / (x^2 + 2h x + h^2 + a^2)  ->  (k/a) * arctan((x + h)/a)
    k / sqrt(a^2 - h^2 - 2h x - x^2)  ->  k * arcsin((x + h)/a)

Limits are chosen as ``x = -h + a*t`` so the shifted variable ``(x + h)/a`` lands on ``t``, a value whose inverse
trig function is a rational multiple of pi. Built backwards from ``(k, a, h, t_lo, t_hi)``.
"""

from __future__ import annotations

import random

from sympy import Rational, Symbol, asin, atan, integrate, pi, sqrt

from ..core import check_equivalent, generator, join_terms, latex_answer, latex_prompt, pick_distractors

_X = Symbol("x", real=True)


def _fmt(expr) -> str:
    return str(expr).replace("**", "^")


def _quadratic_text(lead: int, b: int, c: int) -> str:
    return join_terms([(lead, "x^2"), (b, "x"), (c, "")])


def _arctan_variant(rng: random.Random) -> dict:
    a = rng.randint(2, 5)
    h = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
    k = rng.randint(1, 4)
    t_lo, t_hi = rng.choice([(-1, 1), (0, 1), (-1, 0)])
    lo, hi = -h + a * t_lo, -h + a * t_hi
    denom = _X**2 + 2 * h * _X + h * h + a * a
    integrand = k / denom
    answer = Rational(k, a) * (atan(t_hi) - atan(t_lo))
    check_equivalent(answer, integrate(integrand, (_X, lo, hi)))
    delta = atan(t_hi) - atan(t_lo)
    wrong = [
        k * delta,  # forgot the 1/a that comes with arctan((x+h)/a)
        Rational(k, a * a) * delta,  # divided by a^2, the constant term, instead of by a
        Rational(k, a) * -delta,  # limits taken the wrong way round
        Rational(1, a) * delta,  # dropped the numerator k
    ]
    kk = "" if k == 1 else f"{k}"
    text = (
        f"Find ∫ from x = {lo} to x = {hi} of {kk or '1'}/({_quadratic_text(1, 2 * h, h * h + a * a)}) dx. "
        "Give your answer as an exact value in terms of pi."
    )
    return _pack(text, answer, wrong, f"Complete the square: x^2 + {2 * h}x + {h * h + a * a} = (x + {h})^2 + {a * a}")


def _arcsin_variant(rng: random.Random) -> dict:
    a = rng.choice([2, 4, 6])
    h = rng.choice([-3, -2, -1, 1, 2, 3])
    k = rng.randint(1, 4)
    t_lo, t_hi = rng.choice(
        [(Rational(-1, 2), Rational(1, 2)), (0, Rational(1, 2)), (Rational(-1, 2), 1), (-1, Rational(1, 2)), (0, 1)]
    )
    lo, hi = int(-h + a * t_lo), int(-h + a * t_hi)
    radicand = a * a - h * h - 2 * h * _X - _X**2
    integrand = k / sqrt(radicand)
    answer = k * (asin(t_hi) - asin(t_lo))
    check_equivalent(answer, integrate(integrand, (_X, lo, hi)))
    delta = asin(t_hi) - asin(t_lo)
    wrong = [
        Rational(k, a) * delta,  # divided by a, as for arctan
        k * a * delta,  # multiplied by a
        -k * delta,  # limits taken the wrong way round
        delta,  # dropped the numerator k
    ]
    text = (
        f"Find ∫ from x = {lo} to x = {hi} of {k}/sqrt({_quadratic_text(-1, -2 * h, a * a - h * h)}) dx. "
        "Give your answer as an exact value in terms of pi."
    )
    return _pack(text, answer, wrong, f"Complete the square: {a * a - h * h} - 2·{h}x - x^2 = {a * a} - (x + {h})^2")


def _pack(text: str, answer, wrong, first_step: str) -> dict:
    answer_text = _fmt(answer)
    return {
        "text": text,
        "latex": latex_prompt(text),
        "answer": answer_text,
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer),
        "distractors": pick_distractors(answer_text, [_fmt(w) for w in wrong]),
        "working": [first_step, f"Integrate and substitute the limits: {answer_text}"],
    }


@generator("exd_x_c4_inverse_trig_forms", level=17, presets={"exam": dict(tier="exam")})
def exd_x_c4_inverse_trig_forms(rng: random.Random, tier: str = "exam") -> dict:
    """A definite integral that needs completing the square before an inverse trig antiderivative applies."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _arctan_variant(rng) if rng.random() < 0.5 else _arcsin_variant(rng)
