"""Exam question on the Fourier transform of a Gaussian pulse and of the same pulse on a cosine carrier.

The stem gives f(x) = A exp(-a x^2) and g(x) = f(x) cos(w0 x), with the ordinary convention
F(w) = integral of f(x) exp(-i w x) dx. Four parts:

(a) F(w) = A sqrt(pi/a) exp(-w^2/(4a))                                  expression, 2 marks
(b) the integral of f(x) cos(w0 x) over the real line, = F(w0)          number (3 d.p.), 2 marks, follow_through
(c) G(w0), the transform of g at the carrier frequency, by the
    modulation property: (F(0) + F(2 w0)) / 2                           number (3 d.p.), 2 marks, follow_through
(d) G(w0) / G(0) = (e^x + e^(-3x)) / 2, x = w0^2 / (4a)                                  number (3 d.p.), 1 mark,  follow_through

The follow-through of (b) and (c) evaluates the learner's own F(w), so a slip in (a) is carried through; (d)
divides the learner's (c) by their (b). Parts (b) to (d) are numeric so the question stays cheap to make and to
mark, while (a) is the symbolic step an exam would set.

Every key is computed in 30-digit arithmetic from the closed form and then checked against a trapezoid-rule
quadrature of the defining integrals (which converges geometrically for a Gaussian), so a slip in the formula
cannot survive. The draw keeps each numeric answer well away from 0 so that rounding to 3 places leaves
several significant figures.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import mpmath
import sympy

from ..core import GRADER_TOLERANCE, attempts, check, generator, phrase, round_half_up
from .exd_x_r2h_support import build_part, expression_part

W = sympy.Symbol("w")
_AMPLITUDES = (-6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7)
_WIDTHS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 12)
_CARRIERS = (1, 2, 3, 4, 5, 6)
_DP = 3
_MIN_MAGNITUDE = Fraction(1, 10)

_STEM = (
    "Use the ordinary Fourier transform convention F(w) = the integral over all real x of f(x) exp(-i*w*x) dx. "
    "A pulse is modelled by f(x) = {amp} exp(-{wid}x^2), and g(x) = f(x) cos({wx}) is the same pulse carried on a "
    "wave of angular frequency {w0}. Let G(w) be the transform of g(x).",
    "Let F(w) be the ordinary Fourier transform of f(x) = {amp} exp(-{wid}x^2), that is F(w) = the integral over "
    "all real x of f(x) exp(-i*w*x) dx. The modulated signal is g(x) = f(x) cos({wx}), with transform G(w).",
    "A signal has envelope f(x) = {amp} exp(-{wid}x^2) and is multiplied by a carrier to give g(x) = f(x) "
    "cos({wx}). With the convention F(w) = the integral over all real x of f(x) exp(-i*w*x) dx, write G(w) for "
    "the transform of g.",
)
_PROMPT_A = ("Find F(w).", "Find F(w), the transform of f.")
_PROMPT_B = (
    "Hence find the value of the integral of f(x) cos({wx}) over all real x, correct to 3 decimal places.",
    "Use F(w) to evaluate the integral over all real x of f(x) cos({wx}) dx, correct to 3 decimal places.",
)
_PROMPT_C = (
    "Using the modulation (frequency shift) property and your F(w), find G({w0}), correct to 3 decimal places.",
    "Use the modulation property to find the value of G({w0}), correct to 3 decimal places.",
)
_PROMPT_D = (
    "Hence find the ratio G({w0}) / G(0), correct to 3 decimal places.",
    "Express G({w0}) as a multiple of G(0), correct to 3 decimal places.",
)


def _trapezoid(func, a_width: int) -> float:
    """Trapezoid rule for the integral of `func` over the real line, for an integrand with a factor exp(-a x^2)."""
    h = 0.04
    half = math.sqrt(45.0 / a_width)
    n = int(half / h) + 1
    return h * sum(func(i * h) for i in range(-n, n + 1))


def _quadrature_check(amp: int, wid: int, w0: int, key_f, key_b: Fraction, key_c: Fraction) -> None:
    """The defining integrals, by quadrature, against the closed forms (f is even, so F is real)."""

    def f(x):
        return amp * math.exp(-wid * x * x)

    probe = 0.7
    q_f = _trapezoid(lambda x: f(x) * math.cos(probe * x), wid)
    check(abs(q_f - float(key_f(probe))) < 1e-9, "F(w) disagrees with quadrature")
    q_b = _trapezoid(lambda x: f(x) * math.cos(w0 * x), wid)
    check(abs(q_b - float(key_b)) < 1e-9, "the integral disagrees with quadrature")
    q_c = _trapezoid(lambda x: f(x) * math.cos(w0 * x) ** 2, wid)
    check(abs(q_c - float(key_c)) < 1e-9, "G(w0) disagrees with quadrature")


def _values(amp: int, wid: int, w0: int):
    """Keys and tagged wrong values (as 30-digit numbers) for parts (b), (c) and (d)."""

    def big_f(omega):
        return amp * mpmath.sqrt(mpmath.pi / wid) * mpmath.exp(-(mpmath.mpf(omega) ** 2) / (4 * wid))

    wmp, wid_mp = mpmath.mpf(w0), mpmath.mpf(wid)
    exact_b = big_f(w0)
    exact_c = (big_f(0) + big_f(2 * w0)) / 2
    ratio = exact_c / exact_b
    b_wrong = [
        (big_f(0), "modulation-ignored"),
        (amp * mpmath.sqrt(mpmath.pi / wid_mp) * mpmath.exp(-(wmp**2) / (2 * wid_mp)), "gaussian-exponent-2a"),
        (amp * mpmath.exp(-(wmp**2) / (4 * wid_mp)), "gaussian-prefactor-dropped"),
        (amp * mpmath.sqrt(wid_mp / mpmath.pi) * mpmath.exp(-(wmp**2) / (4 * wid_mp)), "gaussian-prefactor-inverted"),
    ]
    c_wrong = [
        ((big_f(0) + big_f(w0)) / 2, "sideband-distance-halved"),
        (big_f(0) / 2, "modulation-one-sideband-only"),
        (big_f(0) + big_f(2 * w0), "modulation-half-forgotten"),
        (big_f(0), "modulation-ignored"),
    ]
    d_wrong = [
        (1 / ratio, "inverted-ratio"),
        (2 * ratio, "modulation-half-forgotten"),
        ((big_f(0) + big_f(w0)) / 2 / exact_b, "sideband-distance-halved"),
        (mpmath.exp(wmp**2 / (4 * wid_mp)), "modulation-ignored"),
    ]
    return big_f, (exact_b, b_wrong), (exact_c, c_wrong), (ratio, d_wrong)


def _rounded(value) -> Fraction:
    return round_half_up(Fraction(mpmath.nstr(value, 28, strip_zeros=False)), _DP)


def _usable(key, wrong) -> list:
    """The wrong values, rounded, that are not too small to be an answer and that the grader would not accept."""
    key = _rounded(key)
    out = []
    for value, tag in wrong:
        shown = _rounded(value)
        if abs(shown) >= Fraction(1, 100) and abs(shown - key) > GRADER_TOLERANCE:
            out.append((shown, tag))
    return out


def _build(rng: random.Random) -> dict:
    mpmath.mp.dps = 30
    for _ in attempts(1000, "a pulse with answers well away from zero"):
        amp = rng.choice(_AMPLITUDES)
        wid = rng.choice(_WIDTHS)
        w0 = rng.choice(_CARRIERS)
        big_f, (exact_b, b_wrong), (exact_c, c_wrong), (ratio, d_wrong) = _values(amp, wid, w0)
        if (
            abs(float(exact_b)) >= 0.1
            and abs(float(exact_c)) >= 0.1
            and abs(_rounded(exact_c) / _rounded(exact_b) - _rounded(ratio)) <= Fraction(1, 10**_DP)
            and all(
                len({d for d, _ in _usable(k, w)}) >= 3
                for k, w in ((exact_b, b_wrong), (exact_c, c_wrong), (ratio, d_wrong))
            )
        ):
            break
    _quadrature_check(amp, wid, w0, big_f, Fraction(str(exact_b)), Fraction(str(exact_c)))
    x_arg = mpmath.mpf(w0) ** 2 / (4 * wid)
    check(abs(ratio - (mpmath.exp(x_arg) + mpmath.exp(-3 * x_arg)) / 2) < mpmath.mpf("1e-25"), "ratio disagrees")

    a_int = sympy.Integer(wid)
    f_key = amp * sympy.sqrt(sympy.pi / a_int) * sympy.exp(-(W**2) / (4 * a_int))
    sqrt_part = amp * sympy.sqrt(sympy.pi / a_int)
    f_wrong = [
        (amp * sympy.exp(-(W**2) / (4 * a_int)), "gaussian-prefactor-dropped"),
        (sqrt_part * sympy.exp(-(W**2) / (2 * a_int)), "gaussian-exponent-2a"),
        (sqrt_part * sympy.exp(-a_int * W**2), "gaussian-exponent-not-inverted"),
        (amp * sympy.sqrt(a_int / sympy.pi) * sympy.exp(-(W**2) / (4 * a_int)), "gaussian-prefactor-inverted"),
    ]
    key_b, key_c, key_d = _rounded(exact_b), _rounded(exact_c), _rounded(ratio)
    wx = "x" if w0 == 1 else f"{w0}x"
    stem = phrase(rng, _STEM, amp=amp, wid="" if wid == 1 else wid, w0=w0, wx=wx)
    prefactor = f"{amp} sqrt(pi/{wid})"
    parts = [
        expression_part(
            phrase(rng, _PROMPT_A),
            f_key,
            marks=2,
            working=[
                "For f(x) = A exp(-a x^2) the transform is F(w) = A sqrt(pi/a) exp(-w^2/(4a)).",
                f"With A = {amp} and a = {wid}: F(w) = {prefactor} exp(-w^2/{4 * wid}).",
            ],
            wrong=f_wrong,
        ),
        build_part(
            phrase(rng, _PROMPT_B, w0=w0, wx=wx),
            key_b,
            kind="number",
            marks=2,
            dp=_DP,
            working=[f"The integral of f(x) cos({wx}) dx is the real part of F({w0}), which is F({w0}) = {key_b}."],
            wrong=_usable(exact_b, b_wrong),
            follow_through=f"Subs(part_a, (w, z), ({w0}, 0))",
        ),
        build_part(
            phrase(rng, _PROMPT_C, w0=w0, wx=wx),
            key_c,
            kind="number",
            marks=2,
            dp=_DP,
            working=[
                "cos(w0 x) = (e^(i w0 x) + e^(-i w0 x))/2, so G(w) = (F(w - w0) + F(w + w0))/2.",
                f"G({w0}) = (F(0) + F({2 * w0}))/2 = {key_c}.",
            ],
            wrong=_usable(exact_c, c_wrong),
            follow_through=f"(Subs(part_a, (w, z), (0, 0)) + Subs(part_a, (w, z), ({2 * w0}, 0)))/2",
        ),
        build_part(
            phrase(rng, _PROMPT_D, w0=w0, wx=wx),
            key_d,
            kind="number",
            marks=1,
            dp=_DP,
            working=[f"G({w0}) / G(0) = ({key_c}) / ({key_b}) = {key_d}."],
            wrong=_usable(ratio, d_wrong),
            follow_through="part_c/part_b",
            sensible_range=(0, None),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r2h_gaussian_modulation", level=19, presets={"exam": dict(tier="exam")})
def exd_x_r2h_gaussian_modulation(rng: random.Random, tier: str = "exam") -> dict:
    """The transform of a Gaussian, then three values of the pulse on a cosine carrier."""
    check(tier == "exam", "exd_x_r2h_gaussian_modulation: only the exam tier exists")
    return _build(rng)
