"""Small helpers for the `gt_f_squared` and `gt_modulus` builds (quadratic and linear graphs read for their features).

Not a generator module (the leading underscore keeps the package loader from importing it). It holds the plane that
the drawn questions use, a numeric scan for turning points (the second route for every key), and the typed spelling
of an inequality and its wrong versions.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import sympy as sp

from .. import figures
from ..answers import Distractor
from ..core import check
from ..grading.inequality import read_solution_set, set_text

LO, HI = -8, 8
_STEP = 12


def plane(fn, label: str, alt: str) -> dict:
    """The graph of `fn` on a unit grid from -8 to 8 on both axes, every integer x an exact sample."""
    pieces = figures.sample_curve(fn, LO, HI, (LO, HI), n=(HI - LO) * _STEP + 1)
    return figures.cartesian(
        x_range=(LO, HI),
        y_range=(LO, HI),
        alt=alt,
        grid=1,
        equal_scale=True,
        curves=[{"pieces": pieces, "label": label}],
    )


def curve_points(fig: dict, label: str = "f") -> list[tuple[float, float]]:
    curve = next(c for c in fig["curves"] if c.get("label") == label)
    return [(x, y) for piece in curve["pieces"] for x, y in piece]


def scan_turning_points(fn, lo: float, hi: float, n: int = 40000):
    """Turning points of `fn` (which must accept a numpy array) on (lo, hi), from sign changes of the slope:
    [(x, y, "max" or "min"), ...]."""
    xs = np.linspace(lo, hi, n + 1)
    ys = np.asarray(fn(xs), dtype=float)
    d = np.diff(ys)
    found = []
    for i in np.nonzero(d[:-1] * d[1:] < 0)[0] + 1:
        found.append((float(xs[i]), float(ys[i]), "max" if d[i - 1] > 0 else "min"))
    return found


def interval_set(spec) -> sp.Set:
    """A sympy set from ``("ge", a)`` (x >= a), ``("le", a)``, ``("outside", a, b)`` or ``("between", a, b)``."""
    kind = spec[0]
    r = lambda v: sp.Rational(Fraction(v).numerator, Fraction(v).denominator)  # noqa: E731
    if kind == "ge":
        return sp.Interval(r(spec[1]), sp.oo)
    if kind == "le":
        return sp.Interval(-sp.oo, r(spec[1]))
    if kind == "outside":
        return sp.Union(sp.Interval(-sp.oo, r(spec[1])), sp.Interval(r(spec[2]), sp.oo))
    if kind == "between":
        return sp.Interval(r(spec[1]), r(spec[2]))
    raise ValueError(spec)


def set_has(s: sp.Set, x: float) -> bool:
    """Whether the float x lies in a union of intervals (comparing floats, since sympy's own test is slow)."""
    if s == sp.S.EmptySet:
        return False
    pieces = s.args if isinstance(s, sp.Union) else (s,)
    return any(float(p.start) <= x <= float(p.end) for p in pieces)


def ineq_text(s: sp.Set, var: str = "x") -> str:
    text = set_text(s, var)
    back = read_solution_set(text)
    check(back is not None and back[0] == s, f"{text!r} does not read back as the set")
    return text


def ineq_dist(s: sp.Set, tag: str, var: str = "x") -> Distractor:
    text = ineq_text(s, var)
    return Distractor(text, tag, text)


def non_negative_set(fn, lo: float, hi: float, n: int = 4000) -> list[tuple[float, float]]:
    """The runs of x in [lo, hi] where fn(x) >= 0 (fn accepts a numpy array): the second route for a domain."""
    xs = np.linspace(lo, hi, n + 1)
    ok = np.asarray(fn(xs), dtype=float) >= -1e-12
    runs, start = [], None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = float(xs[i])
        if not flag and start is not None:
            runs.append((start, float(xs[i - 1])))
            start = None
    if start is not None:
        runs.append((start, float(hi)))
    return runs


def quad_text(a: int, b: int, c: int) -> str:
    """a x^2 + b x + c as a stem writes it: x^2 - 6x + 5."""
    from ..core import join_terms

    return join_terms([(a, "x^2"), (b, "x"), (c, "")])


def para(a, h, k):
    """The function a(x - h)^2 + k, which accepts numbers and numpy arrays."""
    return lambda x: a * (x - h) ** 2 + k


def verify_square(fn, expect_points) -> None:
    """The turning points of fn(x)^2, found by a scan, are exactly `expect_points` [(x, y, "max" or "min")]."""
    lo = min(p[0] for p in expect_points) - 3.0
    hi = max(p[0] for p in expect_points) + 3.0
    found = scan_turning_points(lambda x: fn(x) ** 2, lo, hi)
    check(len(found) == len(expect_points), f"{len(found)} turning points found, {len(expect_points)} expected")
    for (x, y, kind), (ex, ey, ek) in zip(found, sorted(expect_points)):
        check(abs(x - ex) < 2e-2 and abs(y - ey) < 2e-2 * (1 + abs(ey)) and kind == ek, "turning point differs")


def verify_domain(fn, spec_set, lo: float, hi: float) -> None:
    """The set where fn is not negative, from a scan, agrees with `spec_set` away from its end points."""
    runs = non_negative_set(fn, lo, hi)
    for x in [lo + (hi - lo) * i / 199 for i in range(200)]:
        inside = any(a - 1e-9 <= x <= b + 1e-9 for a, b in runs)
        if abs(fn(x)) > 0.1:
            check(inside == set_has(spec_set, x), f"the domain disagrees with the scan at x = {x}")


def vertex_of(pts):
    """The extreme point of a drawn parabola: its lowest point if the ends are above the middle, else its highest."""
    ys = [y for _, y in pts]
    lo, hi = min(ys), max(ys)
    target = lo if (pts[0][1] + pts[-1][1]) / 2 > (lo + hi) / 2 else hi
    return min(((x, y) for x, y in pts if abs(y - target) < 1e-9), key=lambda p: p[0])
