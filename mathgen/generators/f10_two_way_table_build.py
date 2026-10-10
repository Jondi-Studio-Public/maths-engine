"""Generates `f10-two-way-table-relationship` questions (AC9M10ST04): complete the blank cells of a two-way
frequency table of counts. `answer: table`; every blank is an integer.

Scope (the narrowed kernel): the learner fills in counts and totals for two categorical variables, each with
a small, named, fixed set of categories. Comparing conditional or marginal proportions and discussing a
relationship are not asked. Every blank is determined by what is given: a small linear solver in this module
(`_solve`) proves it on every question, and each question's blank positions are chosen at random, so the
answer cannot be guessed from the layout.

Ladder (reasoning escalation, not size):
  easy:   2 x 2. The paired observations are listed, one per person, and the learner tabulates them: cells,
          row totals, column totals and the grand total, with a random subset left blank.
  medium: 2 x 3 or 3 x 2. No list: some cells and totals are given and the learner works the rest out from
          the row, column and grand totals. The blanks are chosen so that filling any line that has just one
          unknown, again and again, finishes the table.
  hard:   3 x 3. That filling in alone gets stuck. A stated condition ("the total for Bus is 6 more than the
          total for Walk", or "twice") links two blank totals, and the learner has to combine it with the
          totals to finish. The generator proves the table is NOT determined without the condition and IS
          determined with it.
"""

from __future__ import annotations

import random
from fractions import Fraction

from mathgen.core import check, generator, phrase

from ._f10_table_kit import build_answer, finish, grid_text, pick_blanks

TIERS = ("easy", "medium", "hard")

# row variable, its categories, column variable, its categories, who is counted
_CONTEXTS = [
    dict(
        noun="students",
        rowvar="year level",
        rows=["Year 4", "Year 5", "Year 6"],
        colvar="way of travelling to school",
        cols=["Walk", "Bus", "Bike"],
    ),
    dict(
        noun="people at a pool",
        rowvar="age group",
        rows=["Child", "Teen", "Adult"],
        colvar="activity",
        cols=["Laps", "Lessons", "Free swim"],
    ),
    dict(
        noun="library visitors",
        rowvar="visitor type",
        rows=["Adult", "Child", "Student"],
        colvar="activity",
        cols=["Borrowed", "Returned", "Studied"],
    ),
    dict(
        noun="cafe customers",
        rowvar="time of day",
        rows=["Morning", "Afternoon", "Evening"],
        colvar="drink ordered",
        cols=["Hot", "Cold", "Juice"],
    ),
    dict(
        noun="games played by a team",
        rowvar="venue",
        rows=["Home", "Away", "Neutral"],
        colvar="result",
        cols=["Won", "Drew", "Lost"],
    ),
    dict(
        noun="households",
        rowvar="type of home",
        rows=["House", "Unit", "Farm"],
        colvar="pet",
        cols=["Dog", "Cat", "None"],
    ),
]


# ---------------------------------------------------------------------------
# Table arithmetic and the uniqueness solver
# ---------------------------------------------------------------------------


def _full_grid(interior: list[list[int]]) -> list[list[int]]:
    """Interior counts with a row-total column and a column-total row (grand total in the corner)."""
    rows = [list(r) + [sum(r)] for r in interior]
    rows.append([sum(r[j] for r in interior) for j in range(len(interior[0]))] + [sum(map(sum, interior))])
    return rows


def _lines(n_rows: int, n_cols: int):
    """Every sum relation of a two-way table, as lists of cells (row, col) that add to a total cell.
    Each yields (parts, total)."""
    out = []
    for i in range(n_rows + 1):
        out.append(([(i, j) for j in range(n_cols)], (i, n_cols)))
    for j in range(n_cols + 1):
        out.append(([(i, j) for i in range(n_rows)], (n_rows, j)))
    return out


def _propagate(n_rows: int, n_cols: int, unknown: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Fill-in by single unknowns: repeatedly resolve any sum line with exactly one unknown cell.
    Returns the cells that stay unresolved (empty means the table can be finished by that method alone)."""
    unknown = set(unknown)
    changed = True
    while changed:
        changed = False
        for parts, total in _lines(n_rows, n_cols):
            cells = parts + [total]
            missing = [c for c in cells if c in unknown]
            if len(missing) == 1:
                unknown.discard(missing[0])
                changed = True
    return unknown


def _solve(n_rows, n_cols, grid, unknown, relations=()):
    """(rank, solution) for the unknown cells given every other cell of `grid`.
    `relations` are extra linear equations as ({cell: coefficient}, constant). The solution is a dict
    cell -> Fraction when the unknowns are uniquely determined, else None."""
    cells = sorted(unknown)
    index = {c: k for k, c in enumerate(cells)}
    equations = []
    for parts, total in _lines(n_rows, n_cols):
        coeffs = [Fraction(0)] * len(cells)
        constant = Fraction(0)
        for c in parts:
            if c in index:
                coeffs[index[c]] += 1
            else:
                constant -= grid[c[0]][c[1]]
        if total in index:
            coeffs[index[total]] -= 1
        else:
            constant += grid[total[0]][total[1]]
        equations.append(coeffs + [constant])
    for coeff_map, constant in relations:
        coeffs = [Fraction(0)] * len(cells)
        const = Fraction(constant)
        for c, v in coeff_map.items():
            if c in index:
                coeffs[index[c]] += v
            else:
                const -= v * grid[c[0]][c[1]]
        equations.append(coeffs + [const])
    # Gauss-Jordan elimination over the rationals.
    rank, n = 0, len(cells)
    for col in range(n):
        pivot = next((r for r in range(rank, len(equations)) if equations[r][col] != 0), None)
        if pivot is None:
            continue
        equations[rank], equations[pivot] = equations[pivot], equations[rank]
        lead = equations[rank][col]
        equations[rank] = [v / lead for v in equations[rank]]
        for r in range(len(equations)):
            if r != rank and equations[r][col] != 0:
                factor = equations[r][col]
                equations[r] = [a - factor * b for a, b in zip(equations[r], equations[rank])]
        rank += 1
    if any(all(v == 0 for v in row[:-1]) and row[-1] != 0 for row in equations):
        return rank, None  # inconsistent
    if rank < n:
        return rank, None
    return rank, {cells[col]: equations[col][-1] for col in range(n)}


def _interior(rng: random.Random, n_rows: int, n_cols: int, lo: int, hi: int) -> list[list[int]]:
    return [[rng.randint(lo, hi) for _ in range(n_cols)] for _ in range(n_rows)]


# ---------------------------------------------------------------------------
# Wording
# ---------------------------------------------------------------------------


def _headers(ctx, rows, cols):
    corner = f"{ctx['rowvar']} / {ctx['colvar']}"
    return [corner, *cols, "Total"], [*rows, "Total"]


def _build(ctx, rows, cols, grid, blank_cells):
    columns, row_labels = _headers(ctx, rows, cols)
    table_rows = [[row_labels[i], *grid[i]] for i in range(len(grid))]
    return build_answer(columns, table_rows, {(r, c + 1) for r, c in blank_cells})


def _blank_key(cell):
    return f"{cell[0]},{cell[1] + 1}"


def _wrong_tables(rng, grid, blank_cells, n_rows, n_cols, extra=()):
    """Misconception edits, as (note id, {"r,c": wrong value}), in the order they are tried."""
    edits = list(extra)
    interior = [c for c in sorted(blank_cells) if c[0] < n_rows and c[1] < n_cols]
    totals = [c for c in sorted(blank_cells) if c[0] == n_rows or c[1] == n_cols]
    grand = grid[n_rows][n_cols]
    rng.shuffle(interior)
    rng.shuffle(totals)
    for r, c in interior:
        x = grid[r][c]
        row_total = grid[r][n_cols]
        edits.append(("adds-instead-of-subtracting", {_blank_key((r, c)): 2 * row_total - x}))
        edits.append(("subtracts-from-the-grand-total", {_blank_key((r, c)): x + grand - row_total}))
        break
    pairs = [(a, b) for i, a in enumerate(interior) for b in interior[i + 1 :] if grid[a[0]][a[1]] != grid[b[0]][b[1]]]
    if pairs:
        a, b = rng.choice(pairs)
        edits.append(
            ("swaps-two-cells", {_blank_key(a): grid[b[0]][b[1]], _blank_key(b): grid[a[0]][a[1]]}),
        )
    for r, c in totals:
        if r == n_rows and c == n_cols:
            edits.append(("adds-row-and-column-totals-together", {_blank_key((r, c)): 2 * grand}))
            continue
        parts = [grid[r][j] for j in range(n_cols)] if c == n_cols else [grid[i][c] for i in range(n_rows)]
        edits.append(("leaves-out-a-cell-from-the-total", {_blank_key((r, c)): sum(parts) - parts[0]}))
        break
    k = rng.choice(sorted(blank_cells))
    edits.append(("miscounts-by-one", {_blank_key(k): grid[k[0]][k[1]] + 1}))
    k2 = rng.choice(sorted(blank_cells))
    edits.append(("miscounts-by-one", {_blank_key(k2): grid[k2[0]][k2[1]] - 1}))
    return edits


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------


def _easy(rng: random.Random) -> dict:
    ctx = rng.choice(_CONTEXTS)
    rows, cols = rng.sample(ctx["rows"], 2), rng.sample(ctx["cols"], 2)
    for _ in range(200):
        interior = _interior(rng, 2, 2, 1, 6)
        if 10 <= sum(map(sum, interior)) <= 18 and len({v for r in interior for v in r}) >= 3:
            break
    grid = _full_grid(interior)
    records = [(rows[i], cols[j]) for i in range(2) for j in range(2) for _ in range(interior[i][j])]
    rng.shuffle(records)
    # Recount the records the long way as an independent check on the table.
    recount = [[sum(1 for rec in records if rec == (rw, cl)) for cl in cols] for rw in rows]
    check(recount == interior, "records disagree with the interior counts")
    cells = [(i, j) for i in range(3) for j in range(3)]
    blank_cells = pick_blanks(rng, cells, 4, 7)
    answer = _build(ctx, rows, cols, grid, blank_cells)
    listing = "; ".join(f"({a}, {b})" for a, b in records)
    intro = phrase(
        rng,
        [
            "{n} {noun} each had their {rv} and {cv} recorded as ({rv}, {cv}): {listing}.",
            "For each of {n} {noun}, the {rv} and the {cv} were noted as ({rv}, {cv}): {listing}.",
        ],
        n=len(records),
        noun=ctx["noun"],
        rv=ctx["rowvar"],
        cv=ctx["colvar"],
        listing=listing,
    )
    text = (
        f"{intro} Complete the two-way table of counts, including the totals. Fill in each ? with a number. "
        f"{grid_text(answer)}"
    )
    edits = _wrong_tables(rng, grid, blank_cells, 2, 2)
    return _pack(answer, edits, grid, ctx, rows, cols, text)


def _medium(rng: random.Random) -> dict:
    ctx = rng.choice(_CONTEXTS)
    n_rows, n_cols = rng.choice([(2, 3), (3, 2)])
    rows, cols = rng.sample(ctx["rows"], n_rows), rng.sample(ctx["cols"], n_cols)
    cells = [(i, j) for i in range(n_rows + 1) for j in range(n_cols + 1)]
    for _ in range(400):
        interior = _interior(rng, n_rows, n_cols, 2, 14)
        grid = _full_grid(interior)
        blank_cells = pick_blanks(rng, cells, 4, 6)
        inner = [c for c in blank_cells if c[0] < n_rows and c[1] < n_cols]
        if len(inner) < 2 or len(inner) == len(blank_cells):
            continue
        if _propagate(n_rows, n_cols, blank_cells):
            continue
        rank, solution = _solve(n_rows, n_cols, grid, blank_cells)
        if solution is None or any(solution[c] != grid[c[0]][c[1]] for c in blank_cells):
            continue
        break
    else:  # pragma: no cover - plenty of patterns qualify
        raise ValueError("no solvable two-way table found")
    answer = _build(ctx, rows, cols, grid, blank_cells)
    text = (
        f"A survey of {ctx['noun']} recorded the {ctx['rowvar']} and the {ctx['colvar']} "
        "of each. Some of the counts are in the two-way table, including row totals, column totals and "
        f"the grand total. Work out the missing counts. Fill in each ? with a number. {grid_text(answer)}"
    )
    edits = _wrong_tables(rng, grid, blank_cells, n_rows, n_cols)
    return _pack(answer, edits, grid, ctx, rows, cols, text)


def _hard(rng: random.Random) -> dict:
    ctx = rng.choice(_CONTEXTS)
    n_rows = n_cols = 3
    rows, cols = rng.sample(ctx["rows"], 3), rng.sample(ctx["cols"], 3)
    cells = [(i, j) for i in range(4) for j in range(4)]
    for _ in range(3000):
        interior = _interior(rng, 3, 3, 2, 12)
        grid = _full_grid(interior)
        by_column = rng.random() < 0.5
        idx_a, idx_b = rng.sample(range(3), 2)
        if by_column:
            ta, tb, side = (3, idx_a), (3, idx_b), "column"
        else:
            ta, tb, side = (idx_a, 3), (idx_b, 3), "row"
        va, vb = grid[ta[0]][ta[1]], grid[tb[0]][tb[1]]
        if va == vb:
            continue
        if va < vb:
            ta, tb, va, vb = tb, ta, vb, va
        if va == 2 * vb:
            kind, amount = "times", 2
        elif va == 3 * vb:
            kind, amount = "times", 3
        else:
            kind, amount = "more", va - vb
        blank_cells = pick_blanks(rng, cells, 7, 9, must=[ta, tb])
        if not _propagate(3, 3, blank_cells):
            continue
        if kind == "times":
            relation = ({ta: Fraction(1), tb: Fraction(-amount)}, 0)  # ta = amount * tb
        else:
            relation = ({ta: Fraction(1), tb: Fraction(-1)}, amount)  # ta - tb = amount
        rank_without, solution_without = _solve(3, 3, grid, blank_cells)
        if solution_without is not None:
            continue
        rank_with, solution = _solve(3, 3, grid, blank_cells, relations=[relation])
        if solution is None or any(solution[c] != grid[c[0]][c[1]] for c in blank_cells):
            continue
        break
    else:  # pragma: no cover - plenty of patterns qualify
        raise ValueError("no condition-linked two-way table found")
    labels = cols if by_column else rows
    name_a, name_b = labels[ta[1] if by_column else ta[0]], labels[tb[1] if by_column else tb[0]]
    statement = (
        f"{amount} more than the total for {name_b}"
        if kind == "more"
        else f"{'twice' if amount == 2 else 'three times'} the total for {name_b}"
    )
    answer = _build(ctx, rows, cols, grid, blank_cells)
    text = (
        f"A survey of {ctx['noun']} recorded the {ctx['rowvar']} and the {ctx['colvar']} of each. Some of the "
        f"counts are in the two-way table, including totals. You are also told that the total for {name_a} is "
        f"{statement}. Use this to work out the missing counts. Fill in each ? with a number. {grid_text(answer)}"
    )
    base = grid[tb[0]][tb[1]]
    if kind == "times":
        misread = ("reads-times-as-more", {_blank_key(ta): base + amount})
    else:
        misread = ("reads-more-as-times", {_blank_key(ta): base * amount})
    edits = [misread, *_wrong_tables(rng, grid, blank_cells, 3, 3)]
    return _pack(answer, edits, grid, ctx, rows, cols, text)


def _pack(answer, edits, grid, ctx, rows, cols, text):
    answer_text, distractors, notes = finish(answer, edits, max_value=grid[-1][-1])
    check(len(distractors) >= 2, "a two-way table question needs at least two tagged wrong tables")
    working = [
        f"{rows[i] if i < len(rows) else 'Total'}: " + ", ".join(str(v) for v in grid[i]) for i in range(len(grid))
    ]
    return dict(
        text=text, answer=answer, answer_text=answer_text, working=working, distractors=distractors, option_notes=notes
    )


@generator(
    "f10_two_way_table",
    level=10,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def f10_two_way_table(rng: random.Random, tier: str = "easy") -> dict:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    return {"easy": _easy, "medium": _medium, "hard": _hard}[tier](rng)
