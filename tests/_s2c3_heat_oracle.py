"""Independent oracle for the heat-equation questions: reads the rod problem back out of the wording and solves it
by Simpson-integrated Fourier coefficients (never the generator's closed forms) and, separately, by an explicit
finite-difference scheme."""

from __future__ import annotations

import math
import re

import numpy as np

import sympy

from _s2c3_oracle import X, expr, pieces

_TRIG = re.compile(r"([+-]?)\s*(\d*)\s*(sin|cos)\((\d*)πx/(\d+)\)")


def _vectorised(body):
    f = sympy.lambdify(X, body, "numpy")
    return lambda x: np.asarray(f(x), dtype=float) * np.ones_like(x)


class Rod:
    def __init__(self, text: str):
        self._cache: dict[int, float] = {}
        self.L = int(re.search(r"rod (\d+) cm long", text).group(1))
        self.k = float(re.search(r"k = ([\d.]+) cm²/s", text).group(1))
        m = re.search(r"end x = 0 is held at (\d+) °C and the end x = \d+ at (\d+) °C", text)
        if m:
            self.bc, self.t1, self.t2 = "dirichlet", float(m.group(1)), float(m.group(2))
        elif "insulated" in text:
            self.bc, self.t1, self.t2 = "neumann", 0.0, 0.0
        else:
            self.bc, self.t1, self.t2 = "dirichlet", 0.0, 0.0
        self.finite = None
        self.constant = 0.0
        ic = re.search(r"Initially u\(x, 0\) = ([^.]*(?:sin|cos)[^.]*)\. ", text)
        if ic:
            body = ic.group(1)
            self.finite = []
            for sign, amp, fn, n, length in _TRIG.findall(body):
                self.finite.append((fn, int(n or 1), (-1 if sign == "-" else 1) * int(amp or 1)))
            lead = re.match(r"\s*(-?\d+)\s*(?:[+-]|$)", body)
            self.constant = float(lead.group(1)) if lead else 0.0
            self.ic_fn = None
        else:
            whole = re.search(r"whole rod is at (\d+) °C", text)
            mid = re.search(r"middle half of the rod \(\S+ ≤ x ≤ \S+\) is at (\d+) °C", text)
            if whole:
                value = float(whole.group(1))
                self.ic_fn = lambda x: np.full_like(x, value)
            elif mid:
                value = float(mid.group(1))
                self.ic_fn = lambda x: np.where((x >= self.L / 4) & (x <= 3 * self.L / 4), value, 0.0)
            else:
                ps = pieces(text, r"u\(x, 0\)")
                if ps:
                    funcs = [(_vectorised(body), float(lo), float(hi)) for body, lo, hi in ps]
                else:
                    bare = re.search(r"u\(x, 0\) = ([^.]*?)\. (?:Find|Write)", text)
                    funcs = [(_vectorised(expr(bare.group(1))), 0.0, float(self.L))]
                self.ic_fn = lambda x: sum(np.where((x >= lo) & (x <= hi), f(x), 0.0) for f, lo, hi in funcs)

    def initial(self, x: np.ndarray) -> np.ndarray:
        if self.finite is not None:
            out = np.full_like(x, self.constant)
            for fn, n, amp in self.finite:
                out += amp * (np.sin if fn == "sin" else np.cos)(n * math.pi * x / self.L)
            return out
        return self.ic_fn(x)

    def steady(self, x):
        return self.t1 + (self.t2 - self.t1) * x / self.L

    def family(self) -> str:
        return "cos" if self.bc == "neumann" else "sin"

    def coefficient(self, n: int) -> float:
        if n not in self._cache:
            self._cache[n] = self._coefficient(n)
        return self._cache[n]

    def _coefficient(self, n: int) -> float:
        """(2/L) integral of (u(x,0) - steady) against the family's mode n, by Simpson on a fine grid; for n = 0 the mean."""
        x = np.linspace(0.0, self.L, 6001)
        # nudge the grid off the jumps by evaluating the initial temperature at cell centres
        mid = (x[:-1] + x[1:]) / 2
        w = self.initial(mid) - self.steady(mid)
        if n == 0:
            return float(w.sum() * (x[1] - x[0]) / self.L)
        mode = np.sin if self.family() == "sin" else np.cos
        return float((w * mode(n * math.pi * mid / self.L)).sum() * (x[1] - x[0]) * 2 / self.L)

    def u(self, x0: float, t0: float, terms: int = 150) -> float:
        if self.finite is not None:
            total = self.constant
            for fn, n, amp in self.finite:
                lam = self.k * (n * math.pi / self.L) ** 2
                total += amp * math.exp(-lam * t0) * (math.sin if fn == "sin" else math.cos)(n * math.pi * x0 / self.L)
            return total
        total = self.steady(x0)
        if self.family() == "cos":
            total += self.coefficient(0)
        mode = math.sin if self.family() == "sin" else math.cos
        for n in range(1, terms + 1):
            lam = self.k * (n * math.pi / self.L) ** 2
            total += self.coefficient(n) * math.exp(-lam * t0) * mode(n * math.pi * x0 / self.L)
        return total

    def finite_difference(self, x0: float, t0: float, cells: int = 240) -> float:
        """Explicit finite differences for u_t = k u_xx with the stated ends and initial temperature."""
        x = np.linspace(0.0, self.L, cells + 1)
        dx = x[1] - x[0]
        dt = 0.4 * dx * dx / self.k
        steps = int(round(t0 / dt))
        dt = t0 / steps
        r = self.k * dt / (dx * dx)
        u = self.initial(x).astype(float)
        for _ in range(steps):
            new = u.copy()
            new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
            if self.bc == "dirichlet":
                new[0], new[-1] = self.t1, self.t2
            else:
                new[0] = u[0] + 2 * r * (u[1] - u[0])
                new[-1] = u[-1] + 2 * r * (u[-2] - u[-1])
            u = new
        return float(np.interp(x0, x, u))
