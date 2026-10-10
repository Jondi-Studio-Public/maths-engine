"""Generates `f10-two-way-table-relationship` questions (AC9M10ST04): tabulate a list of paired categorical
observations into a two-way frequency table. `answer: table`; every blank is an integer.

Scope (the narrowed kernel): the paired observations are always given as a list across exactly two named
variables, each with a small, named, fixed set of categories. The learner tabulates the joint counts, every
cell plus every row and column total. Comparing proportions, stating conditions and discussing a relationship
are not asked. Every blank is determined by the list: the generator recounts the list a second way and checks
every given cell and every blank against it. The blank positions are random, so the answer cannot be guessed
from the layout.

Ladder (reasoning escalation, not size):
  easy:   2 x 2 table from 10-18 records, each written (row category, column category).
  medium: 2 x 3 or 3 x 2 table from a longer list (20-30 records). The grand total and several totals are
          blank, so the learner tabulates cells and then adds rows and columns.
  hard:   3 x 3 table from 26-40 records written in a mixed form: some pairs are written the other way round,
          (column category, row category), and some entries are a repeat such as "3 x (Year 5, Bus)" meaning
          that many identical records. Counting entries instead of records, or misfiling a reversed pair,
          gives a wrong table; the grand total and further marginals are blank.
"""

from __future__ import annotations

import random

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


def _full_grid(interior: list[list[int]]) -> list[list[int]]:
    """Interior counts with a row-total column and a column-total row (grand total in the corner)."""
    rows = [list(r) + [sum(r)] for r in interior]
    rows.append([sum(r[j] for r in interior) for j in range(len(interior[0]))] + [sum(map(sum, interior))])
    return rows


def _interior(rng: random.Random, n_rows: int, n_cols: int, lo: int, hi: int, total_lo: int, total_hi: int):
    for _ in range(500):
        interior = [[rng.randint(lo, hi) for _ in range(n_cols)] for _ in range(n_rows)]
        flat = [v for r in interior for v in r]
        if total_lo <= sum(flat) <= total_hi and len(set(flat)) >= 3:
            return interior
    raise ValueError("no counts in range")  # pragma: no cover - ranges are wide enough


def _blank_key(cell):
    return f"{cell[0]},{cell[1] + 1}"


def _entries(rng, rows, cols, interior, mixed: bool):
    """The list entries (repeat, row category, column category, written_reversed) for the counts."""
    entries = []
    for i, rw in enumerate(rows):
        for j, cl in enumerate(cols):
            n = interior[i][j]
            if mixed and n >= 3 and rng.random() < 0.55:
                k = rng.randint(2, n)
                entries.append((k, rw, cl))
                n -= k
            entries.extend((1, rw, cl) for _ in range(n))
    rng.shuffle(entries)
    return [(k, rw, cl, mixed and rng.random() < 0.4) for k, rw, cl in entries]


def _render(entries) -> str:
    parts = []
    for k, rw, cl, reversed_ in entries:
        pair = f"({cl}, {rw})" if reversed_ else f"({rw}, {cl})"
        parts.append(f"{k} x {pair}" if k > 1 else pair)
    return "; ".join(parts)


def _recount(entries, rows, cols):
    """Tabulate the entries the long way (a second route to the interior counts)."""
    counts = [[0] * len(cols) for _ in rows]
    for k, rw, cl, _rev in entries:
        counts[rows.index(rw)][cols.index(cl)] += k
    return counts


def _wrong_tables(rng, grid, entries, rows, cols, blank_cells):
    n_rows, n_cols = len(rows), len(cols)
    grand = grid[n_rows][n_cols]
    interior = [c for c in sorted(blank_cells) if c[0] < n_rows and c[1] < n_cols]
    totals = [c for c in sorted(blank_cells) if c[0] == n_rows or c[1] == n_cols]
    edits = []
    # Hard only: a repeat entry read as a single record.
    once = _recount([(1, rw, cl, r) for _k, rw, cl, r in entries], rows, cols)
    for r, c in interior:
        if once[r][c] != grid[r][c]:
            edits.append(("reads-a-repeat-as-one-record", {_blank_key((r, c)): once[r][c]}))
            break
    rng.shuffle(interior)
    if interior:
        r, c = interior[0]
        edits.append(("misses-a-record", {_blank_key((r, c)): grid[r][c] - 1}))
        r, c = interior[-1]
        edits.append(("double-counts-a-record", {_blank_key((r, c)): grid[r][c] + 1}))
    pairs = [(a, b) for i, a in enumerate(interior) for b in interior[i + 1 :] if grid[a[0]][a[1]] != grid[b[0]][b[1]]]
    if pairs:
        a, b = rng.choice(pairs)
        edits.append(("swaps-two-cells", {_blank_key(a): grid[b[0]][b[1]], _blank_key(b): grid[a[0]][a[1]]}))
    for r, c in totals:
        if r == n_rows and c == n_cols:
            edits.append(("adds-row-and-column-totals-together", {_blank_key((r, c)): 2 * grand}))
            continue
        parts = [grid[r][j] for j in range(n_cols)] if c == n_cols else [grid[i][c] for i in range(n_rows)]
        edits.append(("leaves-out-a-cell-from-the-total", {_blank_key((r, c)): sum(parts) - parts[0]}))
        break
    spare = sorted(blank_cells)
    rng.shuffle(spare)
    for k in spare:
        edits.append(("miscounts-by-one", {_blank_key(k): grid[k[0]][k[1]] + rng.choice([-1, 1])}))
    return edits


def _build(rng, tier, n_rows, n_cols, count_range, blanks_range, mixed):
    ctx = rng.choice(_CONTEXTS)
    rows, cols = rng.sample(ctx["rows"], n_rows), rng.sample(ctx["cols"], n_cols)
    interior = _interior(rng, n_rows, n_cols, 1, 6 if tier != "hard" else 7, *count_range)
    grid = _full_grid(interior)
    entries = _entries(rng, rows, cols, interior, mixed)
    check(_recount(entries, rows, cols) == interior, "list entries disagree with the interior counts")
    check(sum(k for k, *_ in entries) == grid[n_rows][n_cols], "list length disagrees with the grand total")
    cells = [(i, j) for i in range(n_rows + 1) for j in range(n_cols + 1)]
    must = [(n_rows, n_cols)] if tier != "easy" else []
    blank_cells = pick_blanks(rng, cells, *blanks_range, must=must)
    columns = [f"{ctx['rowvar']} / {ctx['colvar']}", *cols, "Total"]
    labels = [*rows, "Total"]
    answer = build_answer(
        columns, [[labels[i], *grid[i]] for i in range(n_rows + 1)], {(r, c + 1) for r, c in blank_cells}
    )
    # Every blank and every given cell must agree with the recounted list.
    for i in range(n_rows + 1):
        for j in range(n_cols + 1):
            value = answer["blanks"].get(f"{i},{j + 1}", answer["rows"][i][j + 1])
            expected = recount_total(interior, i, j, n_rows, n_cols)
            check(value == expected, "table cell disagrees with the recounted list")
    listing = _render(entries)
    note = ""
    if mixed:
        note = (
            f" Some pairs are written the other way round, as ({ctx['colvar']}, {ctx['rowvar']}), and "
            '"3 x" in front of a pair means that many identical records.'
        )
    intro = phrase(
        rng,
        [
            "The {rv} and the {cv} of each of the {noun} were recorded as ({rv}, {cv}): {listing}.{note}",
            "For each of the {noun}, the {rv} and the {cv} were noted as ({rv}, {cv}): {listing}.{note}",
        ],
        noun=ctx["noun"],
        rv=ctx["rowvar"],
        cv=ctx["colvar"],
        listing=listing,
        note=note,
    )
    text = (
        f"{intro} Complete the two-way table of counts, including the totals. Fill in each ? with a number. "
        f"{grid_text(answer)}"
    )
    edits = _wrong_tables(rng, grid, entries, rows, cols, blank_cells)
    answer_text, distractors, notes = finish(answer, edits, max_value=grid[n_rows][n_cols])
    check(len(distractors) >= 2, "a two-way table question needs at least two tagged wrong tables")
    working = [f"{labels[i]}: " + ", ".join(str(v) for v in grid[i]) for i in range(n_rows + 1)]
    return dict(
        text=text, answer=answer, answer_text=answer_text, working=working, distractors=distractors, option_notes=notes
    )


def recount_total(interior, i, j, n_rows, n_cols):
    """Cell (i, j) of the full grid, computed straight from the interior counts."""
    if i < n_rows and j < n_cols:
        return interior[i][j]
    if i < n_rows:
        return sum(interior[i])
    if j < n_cols:
        return sum(interior[r][j] for r in range(n_rows))
    return sum(map(sum, interior))


@generator(
    "f10_two_way_table",
    level=10,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def f10_two_way_table(rng: random.Random, tier: str = "easy") -> dict:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    if tier == "easy":
        return _build(rng, tier, 2, 2, (10, 18), (4, 7), mixed=False)
    if tier == "medium":
        n_rows, n_cols = rng.choice([(2, 3), (3, 2)])
        return _build(rng, tier, n_rows, n_cols, (20, 30), (6, 9), mixed=False)
    return _build(rng, tier, 3, 3, (26, 40), (8, 11), mixed=True)
