"""Exam source for `integrate-trig`: a squared sine or cosine taken through its half-angle form to a definite integral.

The stem gives f(x) = c sin^2(kx) or c cos^2(kx). The three parts are the steps a Specialist or first-year exam
sets on it:

(a) rewrite f(x) as p + q cos(2kx) and give q (the identity, with its sign and its factor of one half);
(b) integrate the oscillating term q cos(2kx) between x = 0 and x = m pi / (4k), m = 1 or 3, using the q from (a);
(c) the definite integral of f itself, which adds the constant term's contribution c m pi / (8k) to (b), to 3 d.p.

Parts (b) and (c) carry `follow_through`, so a wrong sign or factor in (a) is paid for once.

The key is exact. It is also checked against Simpson's rule on the stated f(x), which does not use the identity.
"""

from __future__ import annotations

import math
from fractions import Fraction

from mathgen.core import attempts, check, generator, round_half_up

from .exd_x_r3c_parts import fraction_part, number_part
from ._exd_r3_distinct import distinct_part_answers

PI = Fraction(math.pi)  # exact value of the double nearest pi: far finer than the 3 d.p. asked
_MARGIN = Fraction(1, 50)  # keep the 3 d.p. key this far (in thousandths) from a rounding boundary
_FUNCTIONS = (("sin", -1), ("cos", 1))  # sin^2 = (1 - cos 2t)/2, cos^2 = (1 + cos 2t)/2
_SIGMA = {1: 1, 3: -1}  # sin(m pi / 2) for the two upper limits

_STEMS = [
    "Let f(x) = {f}.",
    "The function f is defined by f(x) = {f}.",
    "Consider the function f(x) = {f}.",
]


def _simpson(c, name, k, upper, steps=240):
    fn = math.sin if name == "sin" else math.cos
    h = upper / steps
    total = c * fn(0) ** 2 + c * fn(k * upper) ** 2
    for i in range(1, steps):
        total += (4 if i % 2 else 2) * c * fn(k * i * h) ** 2
    return total * h / 3


def _pi_over(num, den):
    """The text of num*pi/den in lowest terms, such as "pi/8" or "3pi/16"."""
    r = Fraction(num, den)
    top = "π" if r.numerator == 1 else f"{r.numerator}π"
    return top if r.denominator == 1 else f"{top}/{r.denominator}"


@generator("exd_x_r3c_trig_identity_integral", level=16, presets={"exam": {"tier": "exam"}})
@distinct_part_answers
def exd_x_r3c_trig_identity_integral(rng, tier="exam"):
    """c sin^2(kx) or c cos^2(kx): the half-angle form, its oscillating part integrated, then the whole integral."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(500, "a definite integral clear of a rounding boundary"):
        name, s = rng.choice(_FUNCTIONS)
        c = rng.randint(1, 9)
        k = rng.randint(1, 5)
        m = rng.choice((1, 3))
        q = Fraction(s * c, 2)
        osc = q * _SIGMA[m] / (2 * k)  # integral of q cos(2kx) from 0 to m pi / (4k)
        total = Fraction(c * m, 8 * k) * PI + osc
        if abs((total * 1000) % 1 - Fraction(1, 2)) > _MARGIN:
            break
    upper = Fraction(m, 4 * k)
    simpson = _simpson(c, name, k, float(upper) * math.pi)
    check(abs(simpson - float(total)) < 1e-6, "the identity route disagrees with Simpson's rule on f")
    check(round_half_up(total, 3) != round_half_up(Fraction(c * m, 8 * k) * PI, 3), "the cosine term vanishes")

    f_text = f"{c if c != 1 else ''}{name}²({k if k != 1 else ''}x)"
    arg = f"{2 * k}x"
    limit = _pi_over(m, 4 * k)
    stem = rng.choice(_STEMS).format(f=f_text)

    part_a = fraction_part(
        f"The function can be written in the form f(x) = p + q cos({arg}), where p and q are constants. Find q.",
        q,
        marks=2,
        working=[
            f"{name}²θ = (1 {'-' if s < 0 else '+'} cos 2θ)/2, so f(x) = {c}/2 {'-' if s < 0 else '+'} ({c}/2) cos({arg}).",
            f"So q = {q}.",
        ],
        wrongs=[
            (-q, "half-angle-sign-swapped"),
            (2 * q, "half-angle-factor-dropped"),
            (q / 2, "half-angle-halved-twice"),
        ],
    )
    part_b = fraction_part(
        f"Using your value of q, evaluate the definite integral of q cos({arg}) with respect to x from x = 0 to x = {limit}.",
        osc,
        marks=2,
        working=[
            f"The integral of q cos({arg}) is q sin({arg})/{2 * k}.",
            f"At x = {limit}, sin({arg}) = {_SIGMA[m]}, so the value is {osc}.",
        ],
        wrongs=[
            (q * _SIGMA[m], "integral-inner-coefficient-not-divided"),
            (-osc, "integral-limit-sign-slip"),
            (q * _SIGMA[m] * 2 * k, "integral-inner-coefficient-multiplied"),
            (q * _SIGMA[m] / k, "integral-inner-coefficient-not-divided"),
        ],
        follow_through=f"part_a*{_SIGMA[m]}/{2 * k}",
    )
    const = Fraction(c * m, 8 * k) * PI
    part_c = number_part(
        f"Hence find the value of the definite integral of f(x) from x = 0 to x = {limit}, correct to 3 decimal places.",
        total,
        dp=3,
        marks=2,
        sensible_range=(0, None),
        working=[
            f"The constant term {c}/2 contributes ({c}/2) × {limit} = {_pi_over(c * m, 8 * k)}.",
            f"Adding the value from (b), the integral is {round_half_up(total, 3)} (3 d.p.).",
        ],
        wrongs=[
            (const, "dropped-oscillating-term"),
            (const - osc, "integral-limit-sign-slip"),
            (2 * const + osc, "half-angle-factor-dropped"),
            (const + 2 * osc, "half-angle-factor-dropped"),
        ],
        follow_through=f"{c * m}*pi/{8 * k} + part_b",
    )
    return dict(text=stem, parts=[part_a, part_b, part_c])
