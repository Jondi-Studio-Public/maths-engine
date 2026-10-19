"""Shared helpers for the second-year linear programming and discrete mathematics skills (`r3l_*` generators).

Everything here is exact: coordinates, tableaux and costs are `Fraction`s, so a generator can compare its answer with
an independent route without rounding.

* two-variable linear programs: constraint lines, corner points and objective values;
* the simplex tableau: the initial tableau, the entering and leaving rules (most negative objective entry, then
  smallest ratio, lowest index on a tie), one pivot and the run to optimality;
* wording: subscripted variable names and signed linear expressions.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from ..core import VerificationError, attempts

_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

#: One sentence of the pivot rule, quoted in every simplex stem so the answer is a single normal form.
PIVOT_RULE = (
    "Choose the entering column as the most negative entry in the objective row (the lowest column on a tie), and the "
    "pivot row by the minimum ratio test (the lowest row on a tie)."
)


def sub(name: str, i: int) -> str:
    """`x`, 2 -> `x₂`."""
    return f"{name}{str(i).translate(_SUBSCRIPT)}"


def num(v) -> str:
    """A Fraction, int or SymPy rational as the text a student writes: 3, -3/2."""
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def linear_text(coefs, names) -> str:
    """`[3, -1, 0], [x, y, z]` -> `3x − y`: a signed linear expression with a true minus sign and no zero terms."""
    out = ""
    for c, name in zip(coefs, names):
        c = Fraction(c)
        if c == 0:
            continue
        mag = abs(c)
        body = name if mag == 1 else f"{num(mag)}{name}"
        if not out:
            out = ("−" if c < 0 else "") + body
        else:
            out += (" − " if c < 0 else " + ") + body
    return out or "0"


def decimal_matrix(rows) -> list[list[Fraction]]:
    return [[Fraction(v) for v in row] for row in rows]


# -- two-variable linear programs ------------------------------------------------------------------------------------


def line_intersection(c1, c2):
    """The point where a1 x + b1 y = r1 meets a2 x + b2 y = r2, or None when the lines are parallel."""
    (a1, b1, r1), (a2, b2, r2) = c1, c2
    det = a1 * b2 - a2 * b1
    if det == 0:
        return None
    return (Fraction(r1 * b2 - r2 * b1, det), Fraction(a1 * r2 - a2 * r1, det))


def feasible(point, constraints) -> bool:
    """`constraints` are (a, b, r, sense) with sense "<=" or ">="; x, y >= 0 is always imposed."""
    x, y = point
    if x < 0 or y < 0:
        return False
    for a, b, r, sense in constraints:
        value = a * x + b * y
        if (sense == "<=" and value > r) or (sense == ">=" and value < r):
            return False
    return True


def all_intersections(constraints):
    """Every pairwise intersection of the constraint lines and the two axes, feasible or not, in a fixed order."""
    lines = [(a, b, r) for a, b, r, _ in constraints] + [(1, 0, 0), (0, 1, 0)]
    seen, out = set(), []
    for l1, l2 in combinations(lines, 2):
        p = line_intersection(l1, l2)
        if p is not None and p not in seen:
            seen.add(p)
            out.append(p)
    return sorted(out)


def corner_points(constraints):
    """The feasible intersections of the constraint lines and the axes, sorted."""
    return [p for p in all_intersections(constraints) if feasible(p, constraints)]


def objective(coefs, point) -> Fraction:
    return coefs[0] * point[0] + coefs[1] * point[1]


# -- the simplex tableau ---------------------------------------------------------------------------------------------


def initial_tableau(a, b, c):
    """Rows [A | I | b] for the constraints, then the objective row [-c | 0 | 0]."""
    m, n = len(a), len(c)
    rows = []
    for i in range(m):
        rows.append([Fraction(v) for v in a[i]] + [Fraction(int(i == j)) for j in range(m)] + [Fraction(b[i])])
    rows.append([Fraction(-v) for v in c] + [Fraction(0)] * m + [Fraction(0)])
    return rows


def entering_column(tab):
    """The most negative objective-row entry among the variable columns (lowest column on a tie), or None."""
    obj = tab[-1][:-1]
    low = min(obj)
    if low >= 0:
        return None
    return obj.index(low)


def leaving_row(tab, col):
    """The row of the smallest ratio RHS / entry over rows with a positive entry (lowest row on a tie), or None."""
    best, best_row = None, None
    for i, row in enumerate(tab[:-1]):
        if row[col] > 0:
            ratio = row[-1] / row[col]
            if best is None or ratio < best:
                best, best_row = ratio, i
    return best_row


def ratios(tab, col):
    """The ratio RHS / entry for each row with a positive entry in `col` (None elsewhere)."""
    return [(row[-1] / row[col] if row[col] > 0 else None) for row in tab[:-1]]


def pivot(tab, r, c):
    """The tableau after pivoting on entry (r, c): scale row r so the pivot is 1, clear the rest of column c."""
    p = tab[r][c]
    if p == 0:
        raise VerificationError("pivot on a zero entry")
    out = [list(row) for row in tab]
    out[r] = [v / p for v in tab[r]]
    for i, row in enumerate(tab):
        if i != r and row[c] != 0:
            f = row[c]
            out[i] = [v - f * w for v, w in zip(row, out[r])]
    return out


def basis_of(tab, n_vars):
    """For each constraint row, the index of the variable whose column is the unit vector with its 1 in that row (or
    None). `n_vars` is the number of variable columns (decision plus slack)."""
    out = []
    for i in range(len(tab) - 1):
        found = None
        for j in range(n_vars):
            col = [tab[k][j] for k in range(len(tab))]
            if col[i] == 1 and all(v == 0 for k, v in enumerate(col) if k != i):
                found = j
                break
        out.append(found)
    return out


def run_simplex(tab, limit=12):
    """Pivot with the stated rules until optimal; return (the list of tableaux, the pivots as (row, col)).

    Raises when the problem is unbounded or the run exceeds `limit` pivots."""
    tabs, pivots = [tab], []
    for _ in range(limit):
        col = entering_column(tabs[-1])
        if col is None:
            return tabs, pivots
        row = leaving_row(tabs[-1], col)
        if row is None:
            raise VerificationError("unbounded linear program")
        pivots.append((row, col))
        tabs.append(pivot(tabs[-1], row, col))
    raise VerificationError("simplex did not finish")


def unique_entering_and_leaving(tab) -> bool:
    """True when the most negative objective entry and the minimum ratio are both unique (no tie to break)."""
    obj = tab[-1][:-1]
    low = min(obj)
    if low >= 0 or obj.count(low) > 1:
        return False
    col = obj.index(low)
    rs = [r for r in ratios(tab, col) if r is not None]
    return bool(rs) and rs.count(min(rs)) == 1


def tableau_is_tidy(tab, max_den=24, max_abs=400) -> bool:
    """Small enough denominators and entries that a learner can follow the arithmetic by hand."""
    return all(v.denominator <= max_den and abs(v) <= max_abs for row in tab for v in row)


def _has_parallel_rows(a):
    """True when two constraint rows are proportional, so one of the constraints could never matter."""
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if all(a[i][k] * a[j][l] == a[i][l] * a[j][k] for k in range(len(a[i])) for l in range(len(a[i]))):
                return True
    return False


def draw_lp(rng, n, m, *, b_range=(8, 40), a_max=6, c_range=(2, 9), tries=4000):
    """Random data (a, b, c) for max c.x subject to A x <= b, x >= 0, whose first simplex pivot has no tie to break
    and whose run to the optimum is finite, non-degenerate and short."""
    for _ in attempts(tries, "a tidy simplex instance"):
        a = [[rng.randint(0, a_max) for _ in range(n)] for _ in range(m)]
        if any(sum(row) == 0 for row in a) or any(all(a[i][j] == 0 for i in range(m)) for j in range(n)):
            continue
        if sum(1 for row in a for v in row if v == 0) > 1:
            continue
        if _has_parallel_rows(a):
            continue  # one of two parallel constraints would always be redundant
        b = [rng.randint(*b_range) for _ in range(m)]
        c = [rng.randint(*c_range) for _ in range(n)]
        if len(set(c)) < n:
            continue
        tab = initial_tableau(a, b, c)
        if not unique_entering_and_leaving(tab):
            continue
        try:
            tabs, pivots = run_simplex(tab)
        except VerificationError:
            continue
        if not 2 <= len(pivots) <= 4 or not all(tableau_is_tidy(t) for t in tabs):
            continue
        final = tabs[-1]
        # non-degenerate: every basic variable positive, every non-basic objective entry strictly positive
        basis = basis_of(final, n + m)
        if None in basis or any(final[i][-1] <= 0 for i in range(m)):
            continue
        nonbasic = [j for j in range(n + m) if j not in basis]
        if any(final[-1][j] <= 0 for j in nonbasic):
            continue
        if not all(unique_entering_and_leaving(t) for t in tabs[:-1]):
            continue
        return a, b, c, tabs, pivots
    raise VerificationError("no simplex instance")  # pragma: no cover
