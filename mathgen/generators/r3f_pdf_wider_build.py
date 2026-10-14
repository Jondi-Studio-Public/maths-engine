"""A wider exam chain for the probability-density-function concept skill: `exd_x_r3f_pdf_wider`.

The skill's older exam chain (`exd_x_r2k_density`) uses power-law densities. This chain uses a density of two linear
pieces, so the cumulative distribution function is built piece by piece and the median depends on which piece it lies
in. It is appended as an extra exam source; the older chain is unchanged.

Two shapes, each with an unknown constant k:

* ramp then plateau -- f(x) = kx for 0 <= x <= a, f(x) = ka for a <= x <= b;
* plateau then fall -- f(x) = k for 0 <= x <= a, f(x) = k(b - x)/(b - a) for a <= x <= b.

The chain is (a) k, (b) P(X <= a), the cumulative probability at the join, (c) the median, (d) a probability across the
join. Parts (b) to (d) carry `follow_through`. Every answer is an exact fraction, from exact integration, and is
re-derived by summing the area of the pieces with Fractions (trapezoids), which uses no antiderivative.
"""

from __future__ import annotations

import random
from fractions import Fraction

from ..core import attempts, check, generator, phrase
from ._r3f_kit import build_until, fraction_part


def _trapezoid(f, lo: Fraction, hi: Fraction) -> Fraction:
    """Area under a straight-line piece of f between lo and hi, exactly."""
    return (hi - lo) * (f(lo) + f(hi)) / 2


class TwoPiece:
    def __init__(self, shape: str, a: int, b: int):
        self.shape, self.a, self.b = shape, Fraction(a), Fraction(b)
        self.k = 2 / (self.a * (2 * self.b - self.a)) if shape == "ramp" else 2 / (self.a + self.b)
        check(self.area(0, self.b) == 1, "the density does not have area 1")

    def f(self, x: Fraction) -> Fraction:
        x = Fraction(x)
        if self.shape == "ramp":
            return self.k * x if x <= self.a else self.k * self.a
        return self.k if x <= self.a else self.k * (self.b - x) / (self.b - self.a)

    def area(self, lo, hi) -> Fraction:
        """Exact area between lo and hi, split at the join so each piece is a straight line."""
        lo, hi = Fraction(lo), Fraction(hi)
        total = Fraction(0)
        if lo < self.a:
            total += _trapezoid(self.f, lo, min(hi, self.a))
        if hi > self.a:
            total += _trapezoid(self.f, max(lo, self.a), hi)
        return total

    def cdf(self, x) -> Fraction:
        return self.area(0, x)

    def median(self) -> Fraction:
        """The median, found on the plateau (the draw guarantees it lies there) by the area up to the join."""
        base = self.cdf(self.a)
        plateau = self.k * self.a if self.shape == "ramp" else self.k
        check(base < Fraction(1, 2) if self.shape == "ramp" else base >= Fraction(1, 2), "median is not on the plateau")
        if self.shape == "ramp":
            m = self.a + (Fraction(1, 2) - base) / plateau
        else:
            m = Fraction(1, 2) / self.k
        check(self.cdf(m) == Fraction(1, 2), "median does not satisfy F(m) = 1/2")
        return m


def _text(shape: str, a: int, b: int) -> str:
    if shape == "ramp":
        return f"f(x) = kx for 0 ≤ x ≤ {a}, f(x) = {a}k for {a} ≤ x ≤ {b}, and f(x) = 0 otherwise"
    return f"f(x) = k for 0 ≤ x ≤ {a}, f(x) = k({b} - x)/{b - a} for {a} ≤ x ≤ {b}, and f(x) = 0 otherwise"


def _draw(rng: random.Random):
    for _ in attempts(500, "two linear pieces with a median on the plateau"):
        shape = rng.choice(("ramp", "fall"))
        a = rng.randint(2, 6)
        b = a + rng.randint(2, 8)
        if shape == "ramp" and not 3 * a < 2 * b:
            continue
        if shape == "fall" and not 3 * a >= b:
            continue
        dens = TwoPiece(shape, a, b)
        c1 = rng.randint(1, a - 1)
        c2 = rng.randint(a + 1, b - 1) if b - a >= 2 else None
        if c2 is None:
            continue
        return dens, c1, c2
    raise AssertionError("unreachable")


def _exam(rng: random.Random) -> dict:
    dens, c1, c2 = _draw(rng)
    a, b, k = int(dens.a), int(dens.b), dens.k
    fa = dens.cdf(a)
    m = dens.median()
    p = dens.area(c1, c2)
    stem = phrase(
        rng,
        [
            "A continuous random variable X has probability density function {f}, where k is a constant.",
            "The probability density function of a continuous random variable X is {f}, where k is a constant.",
            "Let X be a continuous random variable with probability density function {f}, where k is a constant.",
        ],
        f=_text(dens.shape, a, b),
    )
    if dens.shape == "ramp":
        k_work = [
            f"Area = (1/2) x {a} x {a}k + {a}k x ({b} - {a}) = 1, so k({a * a}/2 + {a * (b - a)}) = 1.",
            f"k = {k}.",
        ]
        k_slips = [
            (Fraction(2, a * a), "r3f-pdf-two-piece-area-ignores-the-plateau"),
            (Fraction(1, a * b), "r3f-pdf-two-piece-area-as-a-rectangle"),
            (Fraction(1, a * (b - a)), "r3f-pdf-two-piece-area-ignores-the-ramp"),
            (Fraction(a * (2 * b - a), 2), "r3f-k-inverted"),
        ]
        fa_ft = f"part_a*{a}**2/2"
        med_ft = f"{a}+(1/2-part_b)/(part_a*{a})"
        p_ft = f"part_a*({a * a}/2+{a}*({c2}-{a})-{c1 * c1}/2)"
        m_slips = [
            (Fraction(b, 2), "r3f-median-taken-as-midpoint"),
            (Fraction(1, 2) / k, "r3f-median-uses-the-plateau-height-only"),
            (Fraction(a + b, 2), "r3f-median-taken-as-midpoint"),
            (a + Fraction(1, 2) / (k * a), "r3f-pdf-two-piece-median-forgets-the-ramp-area"),
        ]
        m_work = [
            f"P(X ≤ {a}) = {fa} < 1/2, so the median is on the plateau.",
            f"{fa} + {a}k(m - {a}) = 1/2 gives m = {m}.",
        ]
    else:
        k_work = [
            f"Area = {a}k + (1/2) x {b - a} x k = 1, so k({a} + {b - a}/2) = 1.",
            f"k = {k}.",
        ]
        k_slips = [
            (Fraction(1, a), "r3f-pdf-two-piece-area-ignores-the-fall"),
            (Fraction(1, b), "r3f-pdf-two-piece-area-as-a-rectangle"),
            (Fraction(2, b - a), "r3f-pdf-two-piece-area-ignores-the-plateau"),
            (Fraction(a + b, 2), "r3f-k-inverted"),
        ]
        fa_ft = f"part_a*{a}"
        med_ft = "1/(2*part_a)"
        p_ft = f"part_a*({a}-{c1})+part_a*(({b}-{a})**2-({b}-{c2})**2)/(2*({b}-{a}))"
        m_slips = [
            (Fraction(b, 2), "r3f-median-taken-as-midpoint"),
            (Fraction(a, 2), "r3f-median-halves-the-plateau"),
            (Fraction(a + b, 2), "r3f-median-taken-as-midpoint"),
            (dens.cdf(a) / 2, "r3f-median-reports-a-probability"),
        ]
        m_work = [f"P(X ≤ {a}) = {fa} ≥ 1/2, so the median is on the plateau.", f"km = 1/2 gives m = {m}."]
    parts = [
        fraction_part("Find the value of k.", k, 2, k_work, k_slips, sensible_range=(0, 1)),
        fraction_part(
            f"Hence find P(X ≤ {a}).",
            fa,
            2,
            [f"The area up to x = {a} is one piece: {fa}."],
            [
                (1 - fa, "r3f-cdf-gives-complement"),
                (Fraction(a, b), "r3f-cdf-assumes-uniform"),
                (k * a, "r3f-pdf-two-piece-reads-the-height-as-the-area"),
                (fa / 2, "r3f-pdf-two-piece-halves-the-area"),
            ],
            follow_through=fa_ft,
            sensible_range=(0, 1),
        ),
        fraction_part(
            "Find the median of X.",
            m,
            3,
            m_work,
            m_slips,
            follow_through=med_ft,
            sensible_range=(0, b),
        ),
        fraction_part(
            f"Find P({c1} < X < {c2}).",
            p,
            3,
            [f"The interval crosses x = {a}, so add the areas of the two pieces: {p}."],
            [
                (1 - p, "r3f-cdf-gives-complement"),
                (Fraction(c2 - c1, b), "r3f-cdf-assumes-uniform"),
                (dens.cdf(c2), "r3f-cdf-probability-uses-upper-value-only"),
                (dens.cdf(c2) + dens.cdf(c1), "r3f-cdf-adds-the-values"),
            ],
            follow_through=p_ft,
            sensible_range=(0, 1),
        ),
    ]
    return dict(text=stem, parts=parts)


@generator("exd_x_r3f_pdf_wider", level=13, presets={"exam": dict(tier="exam")})
def exd_x_r3f_pdf_wider(rng: random.Random, tier: str = "exam") -> dict:
    """k, the cumulative probability at the join, the median, then a probability across the join."""
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    return build_until(rng, _exam)
