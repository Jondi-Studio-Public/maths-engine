"""This module registers NO generator topic. Nothing here reads a generator parameter: the ``read_*`` functions
take a figure spec and nothing else, so a reader built on them re-derives the answer from what the learner
sees.

Coordinates are handled in GRID UNITS ``(u, v)`` inside a :class:`Window`, and
converted to real coordinates ``x = u * xs``, ``y = v * ys`` only when the
figure is built. A line ``v = M u + C`` in grid units is therefore
``y = (M * ys / xs) x + C * ys`` in real units. Every ``through`` point of a
drawn line is a lattice point (integer ``u`` and ``v``), so every value a key
needs is read off a grid intersection.

Plausibility: the
figure renderer labels at most about ten ticks per
axis and thins the rest, so a :class:`Window` keeps every grid line labelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

from .. import figures

# The figure renderer keeps at most ~10 tick labels per axis (ceil(n_ticks / 10) is the
# label stride). A window with more ticks than this would leave some grid lines
# unlabelled and turn "read the value" into "count the squares".
MAX_TICKS = 10


@dataclass(frozen=True)
class Window:
    """A drawing window in grid units, with the real units per grid step."""

    u0: int
    u1: int
    v0: int
    v1: int
    xs: int = 1  # real x per grid step
    ys: int = 1  # real y per grid step

    def __post_init__(self):
        if not (self.u0 < self.u1 and self.v0 < self.v1):
            raise ValueError("window must have u0 < u1 and v0 < v1")
        if self.u1 - self.u0 + 1 > MAX_TICKS or self.v1 - self.v0 + 1 > MAX_TICKS:
            raise ValueError(f"window has more than {MAX_TICKS} ticks on an axis")
        if self.xs <= 0 or self.ys <= 0:
            raise ValueError("grid steps must be positive")

    @property
    def x_range(self) -> tuple[int, int]:
        return (self.u0 * self.xs, self.u1 * self.xs)

    @property
    def y_range(self) -> tuple[int, int]:
        return (self.v0 * self.ys, self.v1 * self.ys)

    @property
    def shows_y_axis(self) -> bool:
        """True when x = 0 lies inside the window (the y-axis is drawn)."""
        return self.u0 <= 0 <= self.u1

    @property
    def shows_x_axis(self) -> bool:
        return self.v0 <= 0 <= self.v1


def lattice_points(win: Window, slope: Fraction, intercept: Fraction) -> list[tuple[int, int]]:
    """Every grid intersection of the window that lies on ``v = slope * u + intercept``."""
    slope, intercept = Fraction(slope), Fraction(intercept)
    pts = []
    for u in range(win.u0, win.u1 + 1):
        v = slope * u + intercept
        if v.denominator == 1 and win.v0 <= v <= win.v1:
            pts.append((u, int(v)))
    return pts


def pick_two_points(rng, pts: Sequence[tuple[int, int]], min_du: int = 2) -> tuple[tuple[int, int], tuple[int, int]]:
    """Two lattice points at least ``min_du`` grid columns apart (so the line is
    unambiguously determined and the rise/run is a real reading). Raises if none."""
    pairs = [(a, b) for i, a in enumerate(pts) for b in pts[i + 1 :] if abs(b[0] - a[0]) >= min_du]
    if not pairs:
        raise ValueError("no two lattice points far enough apart")
    a, b = rng.choice(pairs)
    return a, b


def line_figure(
    win: Window,
    lines: Iterable[tuple[tuple[int, int], tuple[int, int]]],
    alt: str,
    *,
    labels: Sequence[str | None] | None = None,
    points: Iterable[dict] = (),
    equal_scale: bool | None = None,
    x_label: str = "x",
    y_label: str = "y",
    title: str = "",
) -> dict:
    """A cartesian figure of straight lines, each given by two lattice points (grid units).

    ``alt`` must name the object only. ``equal_scale`` defaults to True when the
    two grid steps are equal and False otherwise (a contextual graph).
    """
    if equal_scale is None:
        equal_scale = win.xs == win.ys
    drawn = []
    for i, (p, q) in enumerate(lines):
        for u, v in (p, q):
            if not (win.u0 <= u <= win.u1 and win.v0 <= v <= win.v1):
                raise ValueError(f"through point {(u, v)} is outside the window")
        entry = {"through": [[p[0] * win.xs, p[1] * win.ys], [q[0] * win.xs, q[1] * win.ys]]}
        if labels and labels[i]:
            entry["label"] = labels[i]
        drawn.append(entry)
    return figures.cartesian(
        x_range=win.x_range,
        y_range=win.y_range,
        alt=alt,
        grid=1,
        x_grid=win.xs,
        y_grid=win.ys,
        equal_scale=equal_scale,
        lines=drawn,
        points=points,
        x_label=x_label,
        y_label=y_label,
        title=title,
    )


# ---------------------------------------------------------------- readers
# Each takes the figure spec ONLY (never a generator parameter or the key).


def read_line(spec: dict, index: int = 0) -> tuple[Fraction, Fraction]:
    """``(m, c)`` of the ``index``-th drawn line, in real units, from its two through points."""
    (x1, y1), (x2, y2) = spec["lines"][index]["through"]
    if x1 == x2:
        raise ValueError("a vertical line has no gradient")
    m = Fraction(y2 - y1) / Fraction(x2 - x1)
    return m, Fraction(y1) - m * Fraction(x1)


def read_intersection(spec: dict, i: int = 0, j: int = 1) -> tuple[Fraction, Fraction]:
    """Where line ``i`` meets line ``j`` (real units); raises if they are parallel."""
    m1, c1 = read_line(spec, i)
    m2, c2 = read_line(spec, j)
    if m1 == m2:
        raise ValueError("parallel lines do not meet")
    x = (c2 - c1) / (m1 - m2)
    return x, m1 * x + c1


def read_y_at(spec: dict, x, index: int = 0) -> Fraction:
    m, c = read_line(spec, index)
    return m * Fraction(x) + c


def read_x_at(spec: dict, y, index: int = 0) -> Fraction:
    m, c = read_line(spec, index)
    if m == 0:
        raise ValueError("a horizontal line never reaches another y value")
    return (Fraction(y) - c) / m


def real_line(win: Window, slope: Fraction, intercept: Fraction) -> tuple[Fraction, Fraction]:
    """A grid-unit line ``v = slope u + intercept`` as real ``(m, c)``."""
    return Fraction(slope) * win.ys / win.xs, Fraction(intercept) * win.ys
