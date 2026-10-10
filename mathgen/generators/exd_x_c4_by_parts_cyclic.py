"""Exam source for integration by parts where the method has to be chosen and applied more than once.

Three kinds, each needing a repeated application of the by-parts formula:

* ``e^(a x) sin(b x)`` over ``[0, pi/b]`` -- the integral reappears on the right-hand side and is solved for;
* ``e^(a x) cos(b x)`` over ``[0, pi/(2 b)]`` -- the same cyclic structure with a different boundary term;
* ``x^2 e^(k x)`` over ``[0, 1]`` -- two applications, the second on ``x e^(k x)``.

Every key is checked against sympy integration. Distractors are the usual slips of each kind.
"""

from __future__ import annotations

import random

from sympy import E, Rational, Symbol, cos, exp, integrate, pi, simplify, sin

from ..core import check_equivalent, generator, latex_answer, latex_prompt, pick_distractors

_X = Symbol("x", real=True)


def _fmt(expr) -> str:
    return str(expr).replace("**", "^")


def _cyclic(rng: random.Random) -> dict:
    a, b = rng.choice([(1, 2), (2, 1), (3, 1), (3, 2), (1, 3), (2, 3), (4, 1), (4, 3), (1, 4), (3, 4), (2, 5)])
    use_sin = rng.random() < 0.5
    fn = sin if use_sin else cos
    hi = pi / b if use_sin else pi / (2 * b)
    integrand = exp(a * _X) * fn(b * _X)
    answer = simplify(integrate(integrand, (_X, 0, hi)))
    # Hand derivation of the same value: I (1 + a^2/b^2) = boundary terms after two applications.
    if use_sin:
        boundary = (-exp(a * _X) * cos(b * _X) / b + a * exp(a * _X) * sin(b * _X) / b**2).subs(_X, hi) - (
            -exp(0) * cos(0) / b
        )
    else:
        boundary = (exp(a * _X) * sin(b * _X) / b + a * exp(a * _X) * cos(b * _X) / b**2).subs(_X, hi) - (
            a * exp(0) / b**2
        )
    check_equivalent(answer, boundary / (1 + Rational(a * a, b * b)))
    one_step = (
        (-exp(a * _X) * cos(b * _X) / b).subs(_X, hi) - (-exp(0) * cos(0) / b)
        if use_sin
        else (exp(a * _X) * sin(b * _X) / b).subs(_X, hi)
    )
    wrong = [
        boundary,  # did not divide by (1 + a^2/b^2) after the integral reappeared
        one_step,  # stopped after one application, dropping the remaining integral
    ]
    wrong.append(boundary / (1 - Rational(a * a, b * b)))  # added the reappearing integral instead of subtracting
    name = "sin" if use_sin else "cos"
    bound_text = f"π/{b}" if use_sin and b > 1 else ("π" if use_sin else (f"π/{2 * b}"))
    text = (
        (f"Find ∫ from x = 0 to x = {bound_text} of e^({a}x)·{name}({b}x) dx. Give your answer as an exact value.")
        .replace("e^(1x)", "e^x")
        .replace("(1x)", "(x)")
    )
    text = text.replace(f"{name}(1x)", f"{name}(x)")
    return _pack(
        text, answer, wrong, "Apply integration by parts twice; the original integral reappears, so solve for it"
    )


def _poly(rng: random.Random) -> dict:
    k = rng.choice([1, 2, 3, 4, -1, -2, -3, -4])
    n = rng.choice([1, 2])  # upper limit
    integrand = _X**2 * exp(k * _X)
    answer = simplify(integrate(integrand, (_X, 0, n)))
    # two applications: I = [x^2 e^{kx}/k] - (2/k)[x e^{kx}/k - e^{kx}/k^2]
    antider = exp(k * _X) * (_X**2 / k - 2 * _X / k**2 + 2 / Rational(k) ** 3)
    check_equivalent(answer, antider.subs(_X, n) - antider.subs(_X, 0))
    tail = n * exp(k * n) / k - (exp(k * n) - 1) / k**2  # integral of x e^(kx) over [0, n]
    wrong = [
        antider.subs(_X, n),  # evaluated at the upper limit only
        n**2 * exp(k * n) / k,  # stopped after the first application, dropping the remaining integral
        n**2 * exp(k * n) / k + 2 * tail / k,  # sign slip: added the second integral instead of subtracting it
    ]
    kx = "x" if k == 1 else ("-x" if k == -1 else f"{k}x")
    text = f"Find ∫ from x = 0 to x = {n} of x^2·e^({kx}) dx. Give your answer as an exact value.".replace(
        "e^(x)", "e^x"
    )
    return _pack(text, answer, wrong, "Apply integration by parts twice, taking u = x^2 and then u = x")


def _pack(text: str, answer, wrong, first_step: str) -> dict:
    answer_text = _fmt(answer)
    return {
        "text": text,
        "latex": latex_prompt(text),
        "answer": answer_text,
        "answer_text": answer_text,
        "answer_latex": latex_answer(answer),
        "distractors": pick_distractors(answer_text, [_fmt(simplify(w)) for w in wrong]),
        "working": [first_step, f"Substitute the limits: {answer_text}"],
    }


@generator("exd_x_c4_by_parts_cyclic", level=17, presets={"exam": dict(tier="exam")})
def exd_x_c4_by_parts_cyclic(rng: random.Random, tier: str = "exam") -> dict:
    """Definite integrals that need integration by parts twice (cyclic, or a squared polynomial factor)."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return _poly(rng) if rng.random() < 0.35 else _cyclic(rng)
