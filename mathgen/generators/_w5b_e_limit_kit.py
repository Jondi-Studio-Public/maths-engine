"""Shared arithmetic for the e-as-a-limit skill: tables of (a^h - 1)/h and the choices built from them.

Nothing here registers a topic. A table cell is the quotient evaluated in 40-digit decimal arithmetic and rounded half
up to four places, the way a calculator would show it; `check_cell` confirms it against a second route (`expm1`).
The limit as h tends to 0 is ln a, so e is the base whose limit is exactly 1.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction

from ..core import check

ROWS_ALL = ("0.1", "0.01", "0.001")
ROWS_FINE = ("0.01", "0.001")


def quotient(base: Fraction, h: str) -> Fraction:
    """(base^h - 1) / h at h, rounded half up to four places, as a calculator table would show it."""
    with localcontext() as ctx:
        ctx.prec = 40
        value = (Decimal(base.numerator) / Decimal(base.denominator)) ** Decimal(h)
        cell = ((value - 1) / Decimal(h)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    return Fraction(cell)


def check_cell(base: Fraction, h: str, cell: Fraction) -> None:
    second = math.expm1(float(h) * math.log(float(base))) / float(h)
    check(abs(float(cell) - second) < 6e-5, f"table cell for a={base}, h={h} disagrees with expm1")


def limit(base: Fraction) -> float:
    """The limit of (a^h - 1)/h as h tends to 0: the natural logarithm of the base."""
    return math.log(float(base))


def cell_text(cell: Fraction) -> str:
    return f"{float(cell):.4f}"


def base_text(base: Fraction, places: int) -> str:
    return f"{float(base):.{places}f}"


def row_text(base: Fraction, rows: tuple[str, ...]) -> tuple[str, dict[str, Fraction]]:
    cells = {h: quotient(base, h) for h in rows}
    for h, cell in cells.items():
        check_cell(base, h, cell)
    return ", ".join(f"h = {h}: {cell_text(cells[h])}" for h in rows), cells


def table_text(bases, rows, places) -> tuple[str, dict]:
    """One entry per base, ``a = 2.70 (h = 0.01: 0.9979, h = 0.001: 0.9940)``, separated by semicolons; also the cells by base."""
    lines, grid = [], {}
    for base in bases:
        line, grid[base] = row_text(base, rows)
        lines.append(f"a = {base_text(base, places)} ({line})")
    return "; ".join(lines), grid


def nearest_to_one(bases, grid, h):
    """The base whose cell at `h` is nearest 1, and the gap between the best and the second best distance."""
    ordered = sorted(bases, key=lambda b: abs(grid[b][h] - 1))
    gap = abs(grid[ordered[1]][h] - 1) - abs(grid[ordered[0]][h] - 1)
    return ordered[0], float(gap)


def places_for(bases) -> int:
    return 1 if all((b * 10).denominator == 1 for b in bases) else 2


TARGETS = tuple(Fraction(n, 10) for n in (5, 6, 7, 8, 12, 14, 15, 16, 18, 20))


def target_text(target: Fraction) -> str:
    return f"{float(target):g}"


def draw_bracket(rng):
    """(L, p, q): a target limit L other than 1, and two bases p < e^L < q at two decimal places."""
    target = rng.choice(TARGETS)
    centre = math.exp(float(target))
    lo = Fraction(round(centre * rng.uniform(0.82, 0.95) * 100), 100)
    hi = Fraction(round(centre * rng.uniform(1.05, 1.18) * 100), 100)
    return target, lo, hi


def away_from_e(candidates):
    """Drop wrong values that happen to sit on 2.72, so no option can be picked by recalling e."""
    return [(mid, v) for mid, v in candidates if v is None or abs(float(v) - math.e) > 0.04]
