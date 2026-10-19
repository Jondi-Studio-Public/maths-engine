"""Piecewise-linear functions on a symmetric or half interval and their Fourier coefficients, for the ``s2c3_*`` builds.

A `Fn` is a function on ``[a_first L, a_last L]`` made of pieces ``c0 + c1 x`` whose end points are rational
multiples of the half-period ``L`` (an integer or pi). Coefficients come from the closed-form antiderivative of
``(c0 + c1 x) cos(kx)`` and ``(c0 + c1 x) sin(kx)``; `partial_sum` is the independent check, built from composite
Simpson quadrature instead of the closed form.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ._s2c3_common import linear_text

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# (value, plain text, TeX) of the half-period L
HALF_PERIODS = (
    (1.0, "1", "1"),
    (2.0, "2", "2"),
    (3.0, "3", "3"),
    (4.0, "4", "4"),
    (5.0, "5", "5"),
    (math.pi, "π", r"\pi"),
)


def coefficient_name(kind: str, n: int) -> tuple[str, str]:
    """('b₃', 'b_{3}') for kind 'b', n 3."""
    return f"{kind}{str(n).translate(SUB)}", f"{kind}_{{{n}}}"


def multiple_of_L(a: Fraction, L_text: str, L_tex: str) -> tuple[str, str]:
    """a L written for the learner: with L = 3, 1/2 -> '3/2'; with L = pi, 1/2 -> 'π/2'. (plain, tex)."""
    a = Fraction(a)
    if a == 0:
        return "0", "0"
    sign = "-" if a < 0 else ""
    a = abs(a)
    if L_text == "π":
        num = "π" if a.numerator == 1 else f"{a.numerator}π"
        plain = num if a.denominator == 1 else f"{num}/{a.denominator}"
        tnum = r"\pi" if a.numerator == 1 else rf"{a.numerator}\pi"
        tex = tnum if a.denominator == 1 else rf"\frac{{{tnum}}}{{{a.denominator}}}"
        return sign + plain, sign + tex
    value = a * int(L_text)
    if value.denominator == 1:
        return f"{sign}{value.numerator}", f"{sign}{value.numerator}"
    return f"{sign}{value.numerator}/{value.denominator}", rf"{sign}\frac{{{value.numerator}}}{{{value.denominator}}}"


@dataclass(frozen=True)
class Piece:
    a: Fraction  # left end, in units of L
    b: Fraction  # right end, in units of L
    c0: int
    c1: int


class Fn:
    def __init__(self, L: float, L_text: str, L_tex: str, pieces: list[Piece]):
        self.L, self.L_text, self.L_tex, self.pieces = L, L_text, L_tex, pieces

    # -- evaluation --------------------------------------------------------------------------------------------
    def piece_value(self, piece: Piece, x: float) -> float:
        return piece.c0 + piece.c1 * x

    def at_left_end(self, piece: Piece) -> float:
        return self.piece_value(piece, float(piece.a) * self.L)

    def at_right_end(self, piece: Piece) -> float:
        return self.piece_value(piece, float(piece.b) * self.L)

    def breakpoints(self) -> list[Fraction]:
        return [p.b for p in self.pieces[:-1]]

    def value_at(self, a: Fraction) -> float:
        """f at x = a L inside the domain; at an interior break the right-hand piece."""
        for p in self.pieces:
            if p.a <= a < p.b:
                return self.piece_value(p, float(a) * self.L)
        return self.at_right_end(self.pieces[-1])

    def array(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        for p in self.pieces:
            mask = (x >= float(p.a) * self.L) & (x <= float(p.b) * self.L)
            out[mask] = p.c0 + p.c1 * x[mask]
        return out

    # -- coefficients ------------------------------------------------------------------------------------------
    def integral(self, kind: str, k: float, lo: Fraction | None = None, hi: Fraction | None = None) -> float:
        """Integral of f(x) cos(kx) (kind 'a') or f(x) sin(kx) (kind 'b') over the part of the domain in [lo L, hi L]."""
        total = 0.0
        for p in self.pieces:
            a = p.a if lo is None else max(p.a, lo)
            b = p.b if hi is None else min(p.b, hi)
            if a >= b:
                continue
            total += _trig_integral(p.c0, p.c1, float(a) * self.L, float(b) * self.L, k, kind)
        return total

    def coefficient(self, kind: str, n: int) -> float:
        """Full-period coefficient (1/L) * integral over [-L, L] of f cos(n pi x / L) or f sin(n pi x / L)."""
        return self.integral(kind, n * math.pi / self.L) / self.L

    def check_coefficient(self, kind: str, n: int, norm: float, lo: Fraction, hi: Fraction) -> float:
        """The same integral by Simpson quadrature, piece by piece (an independent route)."""
        k = n * math.pi / self.L
        total = 0.0
        for p in self.pieces:
            a, b = max(p.a, lo), min(p.b, hi)
            if a >= b:
                continue
            x = np.linspace(float(a) * self.L, float(b) * self.L, 4001)
            y = (p.c0 + p.c1 * x) * (np.cos(k * x) if kind == "a" else np.sin(k * x))
            h = (x[-1] - x[0]) / 4000
            total += h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())
        return norm * total

    # -- the series ----------------------------------------------------------------------------------------------
    def series_value(self, a: Fraction) -> tuple[float, float, float, str]:
        """The sum of the full-period Fourier series at x = a L (period 2L, domain [-L, L)): (value, left limit,
        right limit, how). `how` is 'continuous', 'jump' or 'endpoint'."""
        r = a - 2 * math.floor((a + 1) / 2)  # reduced into [-1, 1)
        first, last = self.pieces[0], self.pieces[-1]
        if r == first.a:
            left, right = self.at_right_end(last), self.at_left_end(first)
            return (left + right) / 2, left, right, "endpoint"
        for left_piece, right_piece in zip(self.pieces, self.pieces[1:]):
            if r == left_piece.b:
                left, right = self.at_right_end(left_piece), self.at_left_end(right_piece)
                if left == right:
                    return left, left, right, "continuous"
                return (left + right) / 2, left, right, "jump"
        v = self.value_at(r)
        return v, v, v, "continuous"

    def reduced(self, a: Fraction) -> Fraction:
        return a - 2 * math.floor((a + 1) / 2)

    def partial_sum(self, a: Fraction, terms: int = 160) -> float:
        """S_N at x = a L from Simpson-integrated coefficients (independent of the closed forms)."""
        x0 = float(a) * self.L
        n = np.arange(1, terms + 1)[:, None]
        k = n * math.pi / self.L
        a0 = 0.0
        acc = np.zeros((terms, 1))
        bcc = np.zeros((terms, 1))
        for p in self.pieces:
            x = np.linspace(float(p.a) * self.L, float(p.b) * self.L, 2001)
            y = p.c0 + p.c1 * x
            h = (x[-1] - x[0]) / 2000
            w = np.ones_like(x)
            w[1:-1:2] = 4
            w[2:-1:2] = 2
            wy = w * y * h / 3
            a0 += float(wy.sum())
            acc += (np.cos(k * x[None, :]) * wy[None, :]).sum(axis=1, keepdims=True)
            bcc += (np.sin(k * x[None, :]) * wy[None, :]).sum(axis=1, keepdims=True)
        total = a0 / (2 * self.L) + float(((acc * np.cos(k * x0) + bcc * np.sin(k * x0)) / self.L).sum())
        return total

    # -- text ----------------------------------------------------------------------------------------------------
    def describe(self, close_last: bool = False, name: str = "f(x)", bare_single: bool = False) -> tuple[str, str]:
        """('f(x) = 2 for -3 ≤ x < 0 and f(x) = 4 - x for 0 ≤ x < 3', its TeX)."""
        plain, tex = [], []
        if bare_single and len(self.pieces) == 1:
            body = linear_text(self.pieces[0].c0, self.pieces[0].c1)
            return f"{name} = {body}", f"{name} = {body}"
        for i, p in enumerate(self.pieces):
            lo_p, lo_t = multiple_of_L(p.a, self.L_text, self.L_tex)
            hi_p, hi_t = multiple_of_L(p.b, self.L_text, self.L_tex)
            body = linear_text(p.c0, p.c1)
            last = close_last and i == len(self.pieces) - 1
            plain.append(f"{name} = {body} for {lo_p} ≤ x {'≤' if last else '<'} {hi_p}")
            relation = r"\le" if last else "<"
            tex.append(rf"{name} = {body} \text{{ for }} {lo_t} \le x {relation} {hi_t}")
        return " and ".join(plain), r" \text{ and } ".join(tex)


def _trig_integral(c0: float, c1: float, lo: float, hi: float, k: float, kind: str) -> float:
    if kind == "a":
        if abs(k) < 1e-12:
            return c0 * (hi - lo) + c1 * (hi * hi - lo * lo) / 2

        def F(x):
            return (c0 + c1 * x) * math.sin(k * x) / k + c1 * math.cos(k * x) / (k * k)
    else:
        if abs(k) < 1e-12:
            return 0.0

        def F(x):
            return -(c0 + c1 * x) * math.cos(k * x) / k + c1 * math.sin(k * x) / (k * k)

    return F(hi) - F(lo)


# -- half-range series ---------------------------------------------------------------------------------------------


def half_range_extension(fn: Fn, series: str, a: Fraction, side: int) -> float:
    """The periodic odd ('b', sine series) or even ('a', cosine series) extension of f from [0, L], at x = a L, taken
    from the right (side 1) or the left (side -1) by stepping 10^-9 of L either way."""
    t = Fraction(a) + side * Fraction(1, 10**9)
    t = t - 2 * math.floor((t + 1) / 2)  # into [-1, 1)
    sign = 1.0
    if t < 0:
        t = -t
        sign = -1.0 if series == "b" else 1.0
    return sign * fn.value_at(t)


def half_range_value(fn: Fn, series: str, a: Fraction) -> tuple[float, float, float]:
    """(value the series converges to, left limit, right limit) at x = a L."""
    left = half_range_extension(fn, series, a, -1)
    right = half_range_extension(fn, series, a, 1)
    return (left + right) / 2, left, right


def half_range_partial_sum(fn: Fn, series: str, a: Fraction, terms: int = 160) -> float:
    """S_N at x = a L of the half-range series, from Simpson-integrated coefficients (independent of closed forms)."""
    x0 = float(a) * fn.L
    n = np.arange(0 if series == "a" else 1, terms + 1)[:, None]
    k = n * math.pi / fn.L
    acc = np.zeros((len(n), 1))
    for p in fn.pieces:
        x = np.linspace(float(p.a) * fn.L, float(p.b) * fn.L, 2001)
        y = p.c0 + p.c1 * x
        h = (x[-1] - x[0]) / 2000
        w = np.ones_like(x)
        w[1:-1:2] = 4
        w[2:-1:2] = 2
        trig = np.cos(k * x[None, :]) if series == "a" else np.sin(k * x[None, :])
        acc += (trig * (w * y * h / 3)[None, :]).sum(axis=1, keepdims=True)
    coeff = acc * 2 / fn.L
    if series == "a":
        coeff[0] /= 2
        return float((coeff * np.cos(k * x0)).sum())
    return float((coeff * np.sin(k * x0)).sum())
