"""Matrices for storing and displaying information
(``senior-use-matrices-for-storing-and-displaying-information-that``, ACMGM013, domain matrices, answer:
number).

Skill ``name``: "Use matrices for storing and displaying
information that can be presented in rows and columns; for example,
databases, links in social or road networks".

Scope -- narrower than the name alone suggests: "Given a stated small
dataset (with context) represented as a matrix of rows and columns, read off the stated entry at a specified
row and column, or compute the stated row or column total." The core is EXACTLY those two operations --
read a stated entry, or compute a stated row/column total -- over a generator-supplied small dataset with a
stated context.

The four shipping matrix neighbours (``matrix-multiplication``,
``matrix-determinant-2x2``, ``matrix-inverse-2x2``, ``matrix-solve-system``)
are all ALGEBRAIC operations on a matrix of bare numbers with no row/column
labels and no dataset context. None of them ever reads a labelled entry or
sums a labelled row/column of a stated dataset -- this skill's core is a
plain lookup/sum over a LABELLED dataset, genuinely simpler in the
arithmetic sense than its own prerequisite (``matrix-multiplication``), and
escalates by DECISION (which row/column is meant; row-sum vs column-sum on
an asymmetric matrix) rather than by size.

Answer space: every tier's answer is a small non-negative
integer -- an entry value, or a row/column sum of small non-negative
integers -- drawn from a matrix whose entries the generator controls.
Distractors are constructed BY DESIGN so at least two survive
``check_pairwise_distinct`` on (at least) 85% of draws, every tier:

- easy (``entry_lookup``): the matrix's own entries are sampled WITHOUT
  REPLACEMENT from a range wide enough to hold every cell, so every cell is
  a genuinely different value from every other cell, always. The three
  distractor positions (wrong row/same column, wrong column/same row, the
  transposed cell) are proven pairwise-distinct POSITIONS from the target
  and from each other purely by index arithmetic (see ``_draw_easy``), so
  all three distractors clear ``check_pairwise_distinct`` on every single
  draw (100%, not merely 85%).
- medium (``row_or_column_total``): the matrix is resampled until every row
  sum is distinct from every other row sum, and every column sum is
  distinct from every other column sum, so a wrong-axis-index total is
  always a genuinely different value. Two further distractors (one entry
  omitted from the total, one entry double-counted) are only accepted when
  their nonzero source entries keep every candidate pairwise-distinct from
  the answer and from each other -- verified directly, not assumed.
- hard/exam (``directed_adjacency``/``directed_adjacency_larger``): the
  network is resampled until (a) every row sum is pairwise distinct, (b)
  every column sum is pairwise distinct, and (c) the row sum and column sum
  at the QUERIED node are themselves different (the matrix is genuinely
  asymmetric at that node) -- this is verified directly by
  ``_verify_asymmetric_at_node``, which is also exercised against a
  deliberately-corrupted (forced-symmetric) input in the test file to prove
  it actually discriminates.

Every draw's independent check re-reads the RAW matrix passed in (never a
running total the construction loop already accumulated) via
``_verify_matrix_lookup_or_total``. The companion oracle
 parses only the rendered
``question.text``/``answer_text`` and never imports this module.
"""

from __future__ import annotations

from ..core import (
    check,
    check_pairwise_distinct,
    generator,
    phrase,
    pick_distractors,
)

_RESAMPLE_LIMIT = 400

# Row/column label pools -- every pool is large enough that `rng.sample`
# never repeats a label within one draw, and no two labels in a pool are
# visually confusable (plausibility: "no two labels that could
# be confused, e.g. no 'Store A' and 'Store B' both abbreviated
# identically").
STORE_NAMES = ("Riverside", "Hillcrest", "Oakville", "Bayview", "Meadowbrook", "Fairfield")
PRODUCT_NAMES = ("Bats", "Balls", "Nets", "Hoops", "Cones", "Bibs")
TOWN_NAMES = ("Milbrook", "Ashford", "Denton", "Rosewood", "Carrington", "Fairview")
NEIGHBOURHOOD_NAMES = (
    "Riverside Estate",
    "Hillcrest Heights",
    "Oakville Green",
    "Bayview Point",
    "Meadowbrook Fields",
    "Fairfield Corner",
    "Kingsford Reserve",
)


# ---------------------------------------------------------------------------
# Independent checks -- each re-reads the RAW matrix passed in, never a
# running total the construction loop already accumulated.
# ---------------------------------------------------------------------------


def _verify_matrix_lookup_or_total(matrix, row_labels, col_labels, query, claimed_value):
    """Re-derive an entry or a row/column total straight from `matrix`
    (a plain list-of-lists) and the label lists, and assert it matches
    `claimed_value` exactly (`int` equality -- these are always small
    integers, no tolerance needed)."""
    kind = query["kind"]
    if kind == "entry":
        i = row_labels.index(query["row"])
        j = col_labels.index(query["col"])
        recomputed = matrix[i][j]
    elif kind == "row_total":
        i = row_labels.index(query["row"])
        recomputed = sum(matrix[i])
    elif kind == "col_total":
        j = col_labels.index(query["col"])
        recomputed = sum(row[j] for row in matrix)
    else:
        raise ValueError(f"unknown query kind: {kind!r}")
    check(recomputed == claimed_value, f"matrix_table: {kind} mismatch ({recomputed} vs {claimed_value})")
    return recomputed


def _verify_asymmetric_at_node(matrix, node: int):
    """Assert the row total and column total at `node` genuinely differ --
    the load-bearing check for the directed-adjacency tiers (a symmetric
    node would make the 'opposite direction' distractor collide with the
    answer). Fired directly on a deliberately-corrupted symmetric matrix in
    the test file to prove it discriminates."""
    row_sum = sum(matrix[node])
    col_sum = sum(row[node] for row in matrix)
    check(
        row_sum != col_sum,
        f"matrix_table: expected asymmetric row/col totals at node {node} (row={row_sum}, col={col_sum})",
    )
    return row_sum, col_sum


# ---------------------------------------------------------------------------
# Draw + reject loops, one per tier.
# ---------------------------------------------------------------------------


def _draw_easy(rng) -> dict:
    for _ in range(_RESAMPLE_LIMIT):
        n = rng.choice([3, 4])
        row_labels = rng.sample(STORE_NAMES, n)
        col_labels = rng.sample(PRODUCT_NAMES, n)
        pool = rng.sample(range(1, 61), n * n)
        matrix = [pool[i * n : (i + 1) * n] for i in range(n)]

        target_row = rng.randrange(n)
        target_col = rng.randrange(n)
        if target_row == target_col:
            # A transposed cell (target_col, target_row) would be the SAME
            # cell as (target_row, target_col) when the two indices are
            # equal -- resample rather than let that distractor collide.
            continue
        wrong_row = (target_row + 1) % n
        wrong_col = (target_col + 1) % n

        answer = matrix[target_row][target_col]
        cand_wrong_row = matrix[wrong_row][target_col]
        cand_wrong_col = matrix[target_row][wrong_col]
        cand_transposed = matrix[target_col][target_row]

        return dict(
            n=n,
            row_labels=row_labels,
            col_labels=col_labels,
            matrix=matrix,
            target_row=target_row,
            target_col=target_col,
            answer=answer,
            cands=[cand_wrong_row, cand_wrong_col, cand_transposed],
        )
    check(False, "matrix_table easy: no valid draw found")


def _draw_medium(rng) -> dict:
    for _ in range(_RESAMPLE_LIMIT):
        n_rows = rng.choice([3, 4])
        n_cols = rng.choice([3, 4])
        row_labels = rng.sample(STORE_NAMES, n_rows)
        col_labels = rng.sample(PRODUCT_NAMES, n_cols)
        matrix = [[rng.randint(0, 25) for _ in range(n_cols)] for _ in range(n_rows)]

        row_sums = [sum(r) for r in matrix]
        col_sums = [sum(matrix[i][j] for i in range(n_rows)) for j in range(n_cols)]
        if len(set(row_sums)) != n_rows or len(set(col_sums)) != n_cols:
            continue  # a coincidental equal total -- resample the whole matrix

        axis = rng.choice(["row", "col"])
        if axis == "row":
            idx = rng.randrange(n_rows)
            answer = row_sums[idx]
            wrong_axis_total = row_sums[(idx + 1) % n_rows]
            line = matrix[idx]
        else:
            idx = rng.randrange(n_cols)
            answer = col_sums[idx]
            wrong_axis_total = col_sums[(idx + 1) % n_cols]
            line = [matrix[i][idx] for i in range(n_rows)]

        nonzero_positions = [k for k, v in enumerate(line) if v > 0]
        if len(nonzero_positions) < 2:
            continue  # need two distinct nonzero entries: one to omit, one to double-count
        omitted_total = answer - line[nonzero_positions[0]]
        doubled_total = answer + line[nonzero_positions[-1]]

        cands = [wrong_axis_total, omitted_total, doubled_total]
        if len({answer, *cands}) != 4:
            continue  # a collision -- resample rather than ship a duplicate option

        return dict(
            n_rows=n_rows,
            n_cols=n_cols,
            row_labels=row_labels,
            col_labels=col_labels,
            matrix=matrix,
            axis=axis,
            idx=idx,
            answer=answer,
            cands=cands,
        )
    check(False, "matrix_table medium: no valid draw found")


def _draw_adjacency(rng, n_choices, hi: int, labels_pool) -> dict:
    for _ in range(_RESAMPLE_LIMIT):
        n = rng.choice(n_choices)
        labels = rng.sample(labels_pool, n)
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = rng.randint(0, hi)

        row_sums = [sum(r) for r in matrix]
        col_sums = [sum(matrix[i][j] for i in range(n)) for j in range(n)]
        if len(set(row_sums)) != n or len(set(col_sums)) != n:
            continue  # a coincidental equal total -- resample the whole network

        node = rng.randrange(n)
        if row_sums[node] == col_sums[node]:
            continue  # not asymmetric at the queried node -- resample

        direction = rng.choice(["from", "to"])
        if direction == "from":
            answer = row_sums[node]
            opposite = col_sums[node]
            adjacent_same_dir = row_sums[(node + 1) % n]
            line = matrix[node]
        else:
            answer = col_sums[node]
            opposite = row_sums[node]
            adjacent_same_dir = col_sums[(node + 1) % n]
            line = [matrix[i][node] for i in range(n)]

        nonzero_positions = [k for k, v in enumerate(line) if v > 0]
        if not nonzero_positions:
            continue
        omitted_total = answer - line[nonzero_positions[0]]

        cands = [opposite, adjacent_same_dir, omitted_total]
        if len({answer, *cands}) != 4:
            continue

        return dict(
            n=n,
            labels=labels,
            matrix=matrix,
            node=node,
            direction=direction,
            answer=answer,
            cands=cands,
        )
    check(False, "matrix_table adjacency: no valid draw found")


# ---------------------------------------------------------------------------
# Wording -- 3 structurally distinct phrasings per tier, sharing the same
# fixed "below: {table}." connector so the independent oracle can locate the
# table with one anchored regex regardless of which intro was drawn.
# ---------------------------------------------------------------------------


def _render_matrix(header: str, col_labels, row_labels, matrix) -> str:
    header_row = header + " | " + " | ".join(col_labels)
    data_rows = "; ".join(row_labels[i] + " | " + " | ".join(str(v) for v in matrix[i]) for i in range(len(row_labels)))
    return f"{header_row}; {data_rows}"


_EASY_TEMPLATES = (
    "A sports store chain records sales of several products across several stores, "
    "with each row a store and each column a product, shown in the table below: "
    "{table}. Find the number of {col} sold by {row}.",
    "The number of units each store sold of each product is shown in the table "
    "below: {table}. How many {col} did {row} sell?",
    "Each row of the table below is a store and each column a product; the entries "
    "are units sold, shown in the table below: {table}. Find how many {col} {row} sold.",
)


def _easy_tier(rng) -> dict:
    draw = _draw_easy(rng)
    row_labels, col_labels, matrix = draw["row_labels"], draw["col_labels"], draw["matrix"]
    target_row, target_col, answer = draw["target_row"], draw["target_col"], draw["answer"]
    row, col = row_labels[target_row], col_labels[target_col]

    query = {"kind": "entry", "row": row, "col": col}
    _verify_matrix_lookup_or_total(matrix, row_labels, col_labels, query, answer)

    table = _render_matrix("Store", col_labels, row_labels, matrix)
    question_text = phrase(rng, _EASY_TEMPLATES, table=table, col=col, row=row)

    answer_text = str(answer)
    check_pairwise_distinct(
        answer, *draw["cands"], msg="matrix_table easy: a distractor collides with the answer or another option"
    )
    distractors = pick_distractors(answer_text, [str(c) for c in draw["cands"]])
    working = [
        f"Read the row for {row} and the column for {col} from the table.",
        f"The entry in row {row}, column {col} is {answer}.",
    ]
    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


_MEDIUM_ROW_TEMPLATES = (
    "A sports store chain records sales of several products across several stores, "
    "shown in the table below: {table}. Find the total number of items sold by {row}.",
    "Each row of the table below is a store and each column a product, with entries "
    "the number of units sold, shown in the table below: {table}. How many items in "
    "total did {row} sell?",
    "The table below shows units sold by store and product, shown in the table "
    "below: {table}. Find the total number of items {row} sold across every product.",
)

_MEDIUM_COL_TEMPLATES = (
    "A sports store chain records sales of several products across several stores, "
    "shown in the table below: {table}. Find the total number of {col} sold across all stores.",
    "Each row of the table below is a store and each column a product, with entries "
    "the number of units sold, shown in the table below: {table}. How many {col} were "
    "sold in total across every store?",
    "The table below shows units sold by store and product, shown in the table "
    "below: {table}. Find the total number of {col} sold, summed over every store.",
)


def _medium_tier(rng) -> dict:
    draw = _draw_medium(rng)
    row_labels, col_labels, matrix = draw["row_labels"], draw["col_labels"], draw["matrix"]
    axis, idx, answer = draw["axis"], draw["idx"], draw["answer"]

    table = _render_matrix("Store", col_labels, row_labels, matrix)
    if axis == "row":
        row = row_labels[idx]
        query = {"kind": "row_total", "row": row, "col": None}
        _verify_matrix_lookup_or_total(matrix, row_labels, col_labels, query, answer)
        question_text = phrase(rng, _MEDIUM_ROW_TEMPLATES, table=table, row=row)
        working = [
            f"{row}'s row: " + " + ".join(str(v) for v in matrix[idx]) + f" = {answer}.",
        ]
    else:
        col = col_labels[idx]
        query = {"kind": "col_total", "row": None, "col": col}
        _verify_matrix_lookup_or_total(matrix, row_labels, col_labels, query, answer)
        question_text = phrase(rng, _MEDIUM_COL_TEMPLATES, table=table, col=col)
        column_values = [row[idx] for row in matrix]
        working = [
            f"{col}'s column: " + " + ".join(str(v) for v in column_values) + f" = {answer}.",
        ]

    answer_text = str(answer)
    check_pairwise_distinct(
        answer, *draw["cands"], msg="matrix_table medium: a distractor collides with the answer or another option"
    )
    distractors = pick_distractors(answer_text, [str(c) for c in draw["cands"]])
    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


_HARD_FROM_TEMPLATES = (
    "A regional bus company records the number of direct daily routes between "
    "{n} towns, where the entry in row X, column Y is the number of routes "
    "running directly from X to Y (this can differ from Y to X), shown in the "
    "table below: {table}. Find the total number of direct routes leaving {node}.",
    "The table below is a directed network of daily bus routes between {n} "
    "towns: the entry in row X, column Y counts the direct routes from X to Y, "
    "shown in the table below: {table}. How many direct routes leave {node} in total?",
    "Each entry in the table below counts the direct bus routes running from the "
    "row town to the column town among {n} towns, shown in the table below: "
    "{table}. Find the total number of routes leaving {node}.",
)

_HARD_TO_TEMPLATES = (
    "A regional bus company records the number of direct daily routes between "
    "{n} towns, where the entry in row X, column Y is the number of routes "
    "running directly from X to Y (this can differ from Y to X), shown in the "
    "table below: {table}. Find the total number of direct routes arriving at {node}.",
    "The table below is a directed network of daily bus routes between {n} "
    "towns: the entry in row X, column Y counts the direct routes from X to Y, "
    "shown in the table below: {table}. How many direct routes arrive at {node} in total?",
    "Each entry in the table below counts the direct bus routes running from the "
    "row town to the column town among {n} towns, shown in the table below: "
    "{table}. Find the total number of routes arriving at {node}.",
)

_EXAM_FROM_TEMPLATES = (
    "A regional transport authority tracks direct minibus routes connecting {n} "
    "neighbourhoods for its annual network audit. The entry in row X, column Y is "
    "the number of direct routes running from X to Y, which can differ from the "
    "number running from Y to X, shown in the table below: {table}. Find the "
    "total number of direct routes leaving {node}.",
    "As part of a council transport review of {n} neighbourhoods, every direct "
    "minibus route is recorded in a directed network: the entry in row X, "
    "column Y counts the routes from X to Y, shown in the table below: {table}. "
    "How many direct routes leave {node} in total, across the whole network?",
    "A council audits its minibus network across {n} neighbourhoods each year. "
    "The table below records, for each ordered pair of neighbourhoods, the "
    "number of direct routes from the row neighbourhood to the column "
    "neighbourhood, shown in the table below: {table}. Find the total number of "
    "direct routes leaving {node}.",
)

_EXAM_TO_TEMPLATES = (
    "A regional transport authority tracks direct minibus routes connecting {n} "
    "neighbourhoods for its annual network audit. The entry in row X, column Y is "
    "the number of direct routes running from X to Y, which can differ from the "
    "number running from Y to X, shown in the table below: {table}. Find the "
    "total number of direct routes arriving at {node}.",
    "As part of a council transport review of {n} neighbourhoods, every direct "
    "minibus route is recorded in a directed network: the entry in row X, "
    "column Y counts the routes from X to Y, shown in the table below: {table}. "
    "How many direct routes arrive at {node} in total, across the whole network?",
    "A council audits its minibus network across {n} neighbourhoods each year. "
    "The table below records, for each ordered pair of neighbourhoods, the "
    "number of direct routes from the row neighbourhood to the column "
    "neighbourhood, shown in the table below: {table}. Find the total number of "
    "direct routes arriving at {node}.",
)


def _adjacency_tier(rng, n_choices, hi, labels_pool, header, from_templates, to_templates) -> dict:
    draw = _draw_adjacency(rng, n_choices, hi, labels_pool)
    labels, matrix, node, direction, answer = (
        draw["labels"],
        draw["matrix"],
        draw["node"],
        draw["direction"],
        draw["answer"],
    )
    node_label = labels[node]

    row_sum, col_sum = _verify_asymmetric_at_node(matrix, node)
    query = {
        "kind": "row_total" if direction == "from" else "col_total",
        "row": node_label if direction == "from" else None,
        "col": node_label if direction == "to" else None,
    }
    _verify_matrix_lookup_or_total(matrix, labels, labels, query, answer)

    table = _render_matrix(header, labels, labels, matrix)
    templates = from_templates if direction == "from" else to_templates
    question_text = phrase(rng, templates, table=table, n=draw["n"], node=node_label)

    answer_text = str(answer)
    check_pairwise_distinct(
        answer, *draw["cands"], msg="matrix_table adjacency: a distractor collides with the answer or another option"
    )
    distractors = pick_distractors(answer_text, [str(c) for c in draw["cands"]])
    verb = "leaving" if direction == "from" else "arriving at"
    line = matrix[node] if direction == "from" else [row[node] for row in matrix]
    working = [
        f"Direct routes {verb} {node_label}: " + " + ".join(str(v) for v in line) + f" = {answer}.",
    ]
    return dict(
        text=question_text,
        latex=f"plain:{question_text}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=working,
        distractors=distractors,
    )


def _hard_tier(rng) -> dict:
    return _adjacency_tier(rng, [4, 5], 9, TOWN_NAMES, "Town", _HARD_FROM_TEMPLATES, _HARD_TO_TEMPLATES)


def _exam_tier(rng) -> dict:
    return _adjacency_tier(
        rng, [5, 6], 9, NEIGHBOURHOOD_NAMES, "Neighbourhood", _EXAM_FROM_TEMPLATES, _EXAM_TO_TEMPLATES
    )


@generator(
    "p14a_matrix_table",
    level=13,
    presets={
        "easy": dict(variant="entry_lookup"),
        "medium": dict(variant="row_or_column_total"),
        "hard": dict(variant="directed_adjacency"),
        "exam": dict(variant="directed_adjacency_larger"),
    },
)
def p14a_matrix_table(rng, variant="entry_lookup"):
    """Use matrices for storing and displaying information that can be
    presented in rows and columns (ACMGM013). Scope: read a
    stated entry at a specified row and column, or compute a stated row or
    column total -- exactly two operations, nothing else."""
    if variant == "entry_lookup":
        return _easy_tier(rng)
    if variant == "row_or_column_total":
        return _medium_tier(rng)
    if variant == "directed_adjacency":
        return _hard_tier(rng)
    if variant == "directed_adjacency_larger":
        return _exam_tier(rng)
    raise ValueError(f"unknown variant: {variant}")
