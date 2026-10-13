"""Exam question: a probability density function with an unknown constant, taken through to its variance.

A density g(x) k on a stated interval is given with k unknown. Part (a) finds k from the requirement that the total
probability is 1, part (b) finds a probability over a stated sub-interval, part (c) finds E(X), and part (d) finds
Var(X) = E(X^2) - E(X)^2. Parts (b) to (d) carry `follow_through`, so a wrong k still earns the later parts for using
it correctly. All answers are exact fractions.

Three shapes: k x^n on [0, b]; k (x - a)^n on [a, b] (a shifted start); and k x (b - x) on [0, b] (an arch). The shapes
are polynomials, integrated exactly with Fractions, and the constant is re-checked with SymPy.
"""

from __future__ import annotations

import random
from fractions import Fraction
from math import comb

import sympy

from ..core import attempts, check, generator
from .exd_x_r2k_support import fraction_part

_X = sympy.Symbol("x")

_TEXT_STEMS = (
    "The continuous random variable X has probability density function f(x) = {f} for {dom}, and f(x) = 0 otherwise.",
    "A continuous random variable X has probability density function f(x) = {f} for {dom} (and f(x) = 0 elsewhere).",
    "Let X be a continuous random variable with probability density function f(x) = {f} on {dom}, and f(x) = 0 outside it.",
)


def _poly(shape: str, a: int, b: int, n: int) -> tuple[dict[int, Fraction], str, str]:
    """(coefficients of g by power, g as text with k in front, the domain as text)."""
    if shape == "power":
        return {n: Fraction(1)}, f"kx^{n}" if n > 1 else "kx", f"0 ≤ x ≤ {b}"
    if shape == "shifted":
        coeffs = {j: Fraction(comb(n, j) * (-a) ** (n - j)) for j in range(n + 1)}
        return coeffs, f"k(x - {a})^{n}" if n > 1 else f"k(x - {a})", f"{a} ≤ x ≤ {b}"
    return {1: Fraction(b), 2: Fraction(-1)}, f"kx({b} - x)", f"0 ≤ x ≤ {b}"


def _integral(coeffs: dict[int, Fraction], lo: int, hi: int, extra: int = 0) -> Fraction:
    """Integral of x^extra * g(x) from lo to hi, exactly."""
    return sum(
        (c * (Fraction(hi) ** (j + extra + 1) - Fraction(lo) ** (j + extra + 1)) / (j + extra + 1))
        for j, c in coeffs.items()
    )


def _frac(v: Fraction) -> str:
    return f"({v.numerator}/{v.denominator})"


@generator("exd_x_r2k_density", level=13, presets={"exam": dict(tier="exam")})
def exd_x_r2k_density(rng: random.Random, tier: str = "exam") -> dict:
    """k, a probability, the mean and the variance of a polynomial density."""
    check(tier == "exam", "exd_x_r2k_density: only the exam tier exists")
    for _ in attempts(200, "a density whose every part has distractors"):
        built = _build(rng)
        if built is not None:
            return built
    raise AssertionError("unreachable")


def _build(rng: random.Random) -> dict | None:
    for _ in attempts(1000, "a density with a usable sub-interval"):
        shape = rng.choice(("power", "power", "shifted", "arch"))
        n = rng.randint(1, 3) if shape == "power" else rng.randint(1, 2)
        a = rng.randint(1, 4) if shape == "shifted" else 0
        b = a + rng.randint(2, 7) if shape == "shifted" else rng.randint(2, 8)
        coeffs, ftext, dom = _poly(shape, a, b, n)
        lo, hi = a, b
        c1 = rng.randint(lo, hi - 1)
        c2 = rng.randint(c1 + 1, hi)
        if c1 == lo and c2 == hi:
            continue
        total = _integral(coeffs, lo, hi)
        if total <= 0:
            continue
        break
    k = 1 / total
    prob = k * _integral(coeffs, c1, c2)
    mean = k * _integral(coeffs, lo, hi, 1)
    second = k * _integral(coeffs, lo, hi, 2)
    var = second - mean**2

    # Independent check with SymPy polynomials: integrate g, x g and x^2 g and compare with the Fraction route.
    g = sympy.Poly(sum(sympy.Rational(c.numerator, c.denominator) * _X**j for j, c in coeffs.items()), _X)

    def area(poly: sympy.Poly, left: int, right: int):
        anti = poly.integrate()
        return anti.eval(right) - anti.eval(left)

    xg = g * sympy.Poly(_X, _X)
    x2g = xg * sympy.Poly(_X, _X)
    k_sym = 1 / area(g, lo, hi)
    check(k_sym == sympy.Rational(k.numerator, k.denominator), "density: k disagrees with SymPy")
    check(k_sym * area(g, c1, c2) == sympy.Rational(prob.numerator, prob.denominator), "density: P disagrees")
    ex = k_sym * area(xg, lo, hi)
    check(ex == sympy.Rational(mean.numerator, mean.denominator), "density: E(X) disagrees")
    check(
        k_sym * area(x2g, lo, hi) - ex**2 == sympy.Rational(var.numerator, var.denominator),
        "density: Var(X) disagrees",
    )
    check(0 < prob < 1 and var > 0 and lo < mean < hi, "density: implausible values")

    stem = rng.choice(_TEXT_STEMS).format(f=ftext, dom=dom) + " k is a constant."
    pa = fraction_part(
        "Find the value of k.",
        k,
        2,
        [f"The total probability is 1, so k x (integral of the shape from {lo} to {hi}) = 1.", f"k = {k}."],
        [
            (Fraction(1, hi - lo), "assumed-uniform-density"),
            (Fraction(n + 1, hi) if shape == "power" else None, "forgot-to-raise-the-power-in-k"),
            (total, "inverted-normalising-constant"),
        ],
        sensible_range=(0, None),
    )
    pb = fraction_part(
        f"Hence find P({c1} < X < {c2}).",
        prob,
        2,
        [f"P = k x (integral of the shape from {c1} to {c2}) = {prob}."],
        [
            (1 - prob, "gave-complement-for-event"),
            (_integral(coeffs, c1, c2), "forgot-normalising-constant"),
            (k * _integral(coeffs, lo, c2), "used-upper-bound-only"),
            (Fraction(c2 - c1, hi - lo), "assumed-uniform-density"),
        ],
        follow_through=f"part_a*{_frac(_integral(coeffs, c1, c2))}",
        sensible_range=(0, 1),
    )
    pc = fraction_part(
        "Hence find E(X).",
        mean,
        2,
        ["E(X) = integral of x f(x) over the whole interval.", f"E(X) = {mean}."],
        [
            (Fraction(lo + hi, 2), "assumed-uniform-mean"),
            (second, "squared-instead-of-mean"),
            (_integral(coeffs, lo, hi, 1), "forgot-normalising-constant"),
        ],
        follow_through=f"part_a*{_frac(_integral(coeffs, lo, hi, 1))}",
        sensible_range=(lo, hi),
    )
    pd = fraction_part(
        "Hence find Var(X).",
        var,
        2,
        [f"E(X^2) = integral of x^2 f(x) = {second}.", f"Var(X) = E(X^2) - E(X)^2 = {second} - ({mean})^2 = {var}."],
        [
            (second, "variance-as-second-moment"),
            (mean**2 - second, "subtracted-in-wrong-order"),
            (second - mean, "forgot-to-square-the-mean"),
        ],
        follow_through=f"part_a*{_frac(_integral(coeffs, lo, hi, 2))} - part_c**2",
        sensible_range=(0, None),
    )
    if None in (pa, pb, pc, pd):
        return None
    return dict(text=stem, parts=[pa, pb, pc, pd])
