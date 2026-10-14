"""Independent oracle for the rectangle questions: reads the plate and its edge profiles back out of the wording, then
solves Laplace's equation by Simpson-integrated Fourier coefficients (edge by edge, then added) and, separately, by
Jacobi iteration on a grid."""

from __future__ import annotations

import math
import re

import numpy as np
import sympy

from _s2c3_oracle import X, expr

_TERM = re.compile(r"([+-]?)\s*(\d*)\s*sin\((\d*)π[xy]/(\d+)\)")


def _profile(body: str):
    """The edge profile as a vectorised function of the position s along the edge."""
    body = body.replace(" °C", "")
    if "sin(" in body:
        terms = [
            ((-1 if sign == "-" else 1) * int(amp or 1), int(n or 1), int(length))
            for sign, amp, n, length in _TERM.findall(body)
        ]
        return lambda s: sum(c * np.sin(n * np.pi * s / length) for c, n, length in terms)
    if " for " in body:
        m = re.match(r"(.+?) for 0 ≤ [xy] ≤ ([\d.]+) and (.+?) for [\d.]+ ≤ [xy] ≤ ([\d.]+)", body)
        first, mid, second = m.group(1), float(m.group(2)), m.group(3)
        f1, f2 = _lambda(first), _lambda(second)
        return lambda s: np.where(s <= mid, f1(s), f2(s))
    return _lambda(body)


def _lambda(text: str):
    f = sympy.lambdify(X, expr(text, local={"y": X}), "numpy")
    return lambda s: np.asarray(f(s), dtype=float) * np.ones_like(s)


class Plate:
    def __init__(self, text: str):
        self.a, self.b = (int(v) for v in re.search(r"0 ≤ x ≤ (\d+), 0 ≤ y ≤ (\d+)", text).groups())
        self.edges = {}
        self._cache: dict = {}
        pattern = r"u\((x, \d+|\d+, y|0, y)\) = (.+?)(?=, the edge| and the edge|\. (?:Find|Write))"
        for args, body in re.findall(pattern, text):
            if args == "0, y":
                edge = "left"
            elif args.endswith(", y"):
                edge = "right"
            elif args == "x, 0":
                edge = "bottom"
            else:
                edge = "top"
            self.edges[edge] = _profile(body)

    def _geometry(self, edge: str, x: float, y: float):
        """(s along the edge, r from the opposite edge, length of the edge, depth across it)."""
        if edge == "top":
            return x, y, self.a, self.b
        if edge == "bottom":
            return x, self.b - y, self.a, self.b
        if edge == "right":
            return y, x, self.b, self.a
        return y, self.a - x, self.b, self.a

    def coefficient(self, edge: str, n: int) -> float:
        key = (edge, n)
        if key not in self._cache:
            self._cache[key] = self._coefficient(edge, n)
        return self._cache[key]

    def _coefficient(self, edge: str, n: int) -> float:
        _, _, ell, _ = self._geometry(edge, 0.0, 0.0)
        s = np.linspace(0.0, ell, 8001)
        mid = (s[:-1] + s[1:]) / 2
        return float((self.edges[edge](mid) * np.sin(n * np.pi * mid / ell)).sum() * (s[1] - s[0]) * 2 / ell)

    def u(self, x: float, y: float, terms: int = 60) -> float:
        total = 0.0
        for edge in self.edges:
            s, r, ell, d = self._geometry(edge, x, y)
            for n in range(1, terms + 1):
                ratio = (
                    math.sinh(n * math.pi * r / ell) / math.sinh(n * math.pi * d / ell)
                    if n * math.pi * d / ell < 600
                    else math.exp(n * math.pi * (r - d) / ell)
                )
                total += self.coefficient(edge, n) * math.sin(n * math.pi * s / ell) * ratio
        return total

    def jacobi(self, x0: float, y0: float, cells: int = 12, sweeps: int = 20000) -> float:
        nx, ny = cells * self.a, cells * self.b
        xs, ys = np.linspace(0.0, self.a, nx + 1), np.linspace(0.0, self.b, ny + 1)
        grid = np.zeros((nx + 1, ny + 1))
        if "top" in self.edges:
            grid[:, -1] = self.edges["top"](xs)
        if "bottom" in self.edges:
            grid[:, 0] = self.edges["bottom"](xs)
        if "right" in self.edges:
            grid[-1, :] = self.edges["right"](ys)
        if "left" in self.edges:
            grid[0, :] = self.edges["left"](ys)
        for corner in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            grid[corner] = 0.0
        hx, hy = (self.a / nx) ** 2, (self.b / ny) ** 2
        wx, wy = 1 / hx, 1 / hy
        for _ in range(sweeps):
            grid[1:-1, 1:-1] = (wx * (grid[2:, 1:-1] + grid[:-2, 1:-1]) + wy * (grid[1:-1, 2:] + grid[1:-1, :-2])) / (
                2 * (wx + wy)
            )
        i = round(x0 / self.a * nx)
        j = round(y0 / self.b * ny)
        return float(grid[i, j])
