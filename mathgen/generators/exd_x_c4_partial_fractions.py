"""Exam source for integration with partial fractions where the integrand is not a ready-made proper fraction.

Two kinds, each built backwards from the partial-fraction form so the key is exact:

* improper: ``(x^2 + ...)/((x - p)(x - q))`` -- numerator and denominator have the same degree, so the learner
  divides first (a constant remains) and then splits the proper remainder;
* repeated factor: ``(A x + B)/(x - r)^2`` -- which splits as ``A/(x - r) + (B + A r)/(x - r)^2``, a log term and
  a reciprocal term.

The limits lie to the right of every pole so each logarithm has a positive argument.
"""

from __future__ import annotations

import random

from sympy import Integer, Rational, Symbol, expand, integrate, log, simplify

from ..core import check_equivalent, generator, join_terms, latex_prompt, pick_distractors

_X = Symbol("x", real=True)


def _fmt(expr) -> str:
    return str(expr).replace("**", "^")


def _poly_text(coeffs: list[int]) -> str:
    """Coefficients from the highest power down to the constant, as a student would write them."""
    n = len(coeffs) - 1
    terms = []
    for i, c in enumerate(coeffs):
        power = n - i
        var = "" if power == 0 else ("x" if power == 1 else f"x^{power}")
        terms.append((c, var))
    return join_terms(terms)


def _improper(rng: random.Random) -> dict:
    p, q = rng.sample([-3, -2, -1, 0, 1, 2, 3], 2)
    a_c = rng.choice([-3, -2, -1, 1, 2, 3, 4])
    b_c = rng.choice([-3, -2, -1, 1, 2, 3, 4])
    lead = rng.choice([1, 1, 2])
    lo = max(p, q) + rng.randint(1, 2)
    hi = lo + rng.randint(2, 5)
    # integrand = lead + a_c/(x - p) + b_c/(x - q)
    numerator = expand(lead * (_X - p) * (_X - q) + a_c * (_X - q) + b_c * (_X - p))
    denominator = expand((_X - p) * (_X - q))
    integrand = numerator / denominator
    answer = lead * (hi - lo) + a_c * log(Rational(hi - p, lo - p)) + b_c * log(Rational(hi - q, lo - q))
    check_equivalent(answer, integrate(integrand, (_X, lo, hi)))
    wrong = [
        a_c * log(Rational(hi - p, lo - p)) + b_c * log(Rational(hi - q, lo - q)),  # dropped the whole-number part
        a_c * log(Rational(hi - q, lo - q))
        + b_c * log(Rational(hi - p, lo - p))
        + lead * (hi - lo),  # swapped residues
        lead * (hi - lo) + (a_c + b_c) * log(Rational(hi - p, lo - p)),  # one log term for both factors
        lead * (hi - lo) + a_c * log(hi - p) + b_c * log(hi - q),  # upper limit only in the logs
    ]
    num_c = [int(numerator.coeff(_X, 2)), int(numerator.coeff(_X, 1)), int(numerator.coeff(_X, 0))]
    den_c = [int(denominator.coeff(_X, 2)), int(denominator.coeff(_X, 1)), int(denominator.coeff(_X, 0))]
    text = (
        f"Find ∫ from x = {lo} to x = {hi} of ({_poly_text(num_c)})/({_poly_text(den_c)}) dx. "
        "Give your answer as an exact value."
    )
    return _pack(text, answer, wrong, "The degrees are equal: divide first, then split the proper remainder")


def _repeated(rng: random.Random) -> dict:
    r = rng.choice([-3, -2, -1, 0, 1, 2, 3])
    a_c = rng.choice([-3, -2, -1, 1, 2, 3, 4])
    b_c = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    lo = r + rng.randint(1, 2)
    hi = lo + rng.randint(2, 5)
    c2 = b_c + a_c * r  # coefficient of 1/(x - r)^2
    integrand = (a_c * _X + b_c) / (_X - r) ** 2
    answer = a_c * log(Rational(hi - r, lo - r)) + c2 * (Rational(1, lo - r) - Rational(1, hi - r))
    check_equivalent(answer, integrate(integrand, (_X, lo, hi)))
    wrong = [
        a_c * log(Rational(hi - r, lo - r))
        + b_c * (Rational(1, lo - r) - Rational(1, hi - r)),  # forgot A r in B + A r
        a_c * log(Rational(hi - r, lo - r)) + c2 * log(Rational(hi - r, lo - r)),  # logged the squared factor too
        a_c * log(Rational(hi - r, lo - r)) - c2 * (Rational(1, lo - r) - Rational(1, hi - r)),  # sign of the 1/u term
        c2 * (Rational(1, lo - r) - Rational(1, hi - r)),  # dropped the log term
    ]
    num = _poly_text([a_c, b_c])
    den = f"(x {'-' if r > 0 else '+'} {abs(r)})^2" if r != 0 else "x^2"
    text = f"Find ∫ from x = {lo} to x = {hi} of ({num})/{den} dx. Give your answer as an exact value."
    return _pack(text, answer, wrong, "Write (Ax + B)/(x - r)^2 as A/(x - r) + (B + A·r)/(x - r)^2")


def _pack(text: str, answer, wrong, first_step: str) -> dict:
    answer_text = _fmt(simplify(answer))
    return {
        "text": text,
        "latex": latex_prompt(text),
        "answer": answer_text,
        "answer_text": answer_text,
        "answer_latex": answer_text,
        "distractors": pick_distractors(answer_text, [_fmt(simplify(w)) for w in wrong]),
        "working": [first_step, f"Integrate term by term and substitute the limits: {answer_text}"],
    }


@generator("exd_x_c4_partial_fractions", level=17, presets={"exam": dict(tier="exam")})
def exd_x_c4_partial_fractions(rng: random.Random, tier: str = "exam") -> dict:
    """Partial fractions with an improper integrand or a repeated linear factor."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _improper(rng) if rng.random() < 0.5 else _repeated(rng)
