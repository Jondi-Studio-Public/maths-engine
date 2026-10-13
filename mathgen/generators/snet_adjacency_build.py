"""Generates `snet_adjacency` questions (easy, medium, hard) and the exam chain `snet_adjacency_chain`.

``graph-adjacency-matrices`` (``financial-discrete``, answer ``number``): the adjacency matrix of a network has a 1
in row i, column j when vertices i and j are joined by an edge and a 0 otherwise (rows and columns in alphabetical
order). Its row sums are the degrees, its entries add to twice the number of edges, and the entry in row i, column j
of A squared is the number of walks of length 2 from i to j.

* easy: from a stated matrix the degree of one vertex, or the number of edges; or, from an edge list, how many
  entries of the adjacency matrix equal 1;
* medium: the number of walks of length 2 from one vertex to another (an entry of A squared) from a stated matrix;
* hard: the number of walks of length 3 between two vertices (an entry of A cubed), or how many walks of length 2
  start at one vertex (a row sum of A squared);
* exam (``snet_adjacency_chain``): a network is given by its edges. (a) its adjacency matrix (a ``matrix`` answer);
  (b) how many walks of length 2 start at a named vertex; (c) how many of them end back at that vertex; (d) how many end
  at a different vertex, following through from (b) and (c).

Every matrix is built from the edge list, checked symmetric with a zero diagonal, and every power of it is computed
with SymPy and again by summing over the middle vertex of the walk.
"""

from __future__ import annotations

import sympy

from ..answers import Distractor
from ..core import check, generator, phrase
from ..grading.matrix import matrices_match, matrix_latex, matrix_text
from ._snet_kit import (
    LETTERS,
    adjacency,
    degrees,
    describe_network,
    int_distractors,
    pick_context,
    random_connected,
)

_ORDER = "with rows and columns in the order {order}"


def _order(n):
    return ", ".join(LETTERS[:n])


def _check_matrix(n, edges, rows):
    sym = sympy.Matrix(rows)
    check(sym == sym.T and all(sym[i, i] == 0 for i in range(n)), "an adjacency matrix is symmetric with zero diagonal")
    check(sum(sym) == 2 * len(edges), "the entries do not add to twice the number of edges")
    return sym


def _walks2(rows, i, j):
    return sum(rows[i][k] * rows[k][j] for k in range(len(rows)))


def _walks3(rows, i, j):
    n = len(rows)
    return sum(rows[i][k] * rows[k][m] * rows[m][j] for k in range(n) for m in range(n))


def _number(text, answer, working, distractors):
    return dict(
        text=text,
        answer_kind="number",
        answer=answer,
        answer_text=str(answer),
        answer_latex=str(answer),
        working=working,
        distractors=distractors,
    )


# -- easy -------------------------------------------------------------------------------------------------------

_DEGREE_STEMS = (
    "A network has the adjacency matrix {m}, {order}. What is the degree of vertex {x}?",
    "The adjacency matrix of a network is {m}, {order}. Find the degree of vertex {x}.",
)
_EDGES_STEMS = (
    "A network has the adjacency matrix {m}, {order}. How many {es} does the network have?",
    "The adjacency matrix of a network is {m}, {order}. How many edges does the network have?",
)
_ONES_STEMS = (
    "{net} The adjacency matrix of the network is written with a 1 for each pair of {vs} that are joined and a 0 "
    "otherwise. How many of its entries are equal to 1?",
    "{net} How many entries of the network's adjacency matrix are 1?",
)


def _easy(rng):
    ctx = pick_context(rng)
    n = rng.randint(4, 6)
    edges = random_connected(rng, n, rng.randint(n, min(n + 3, n * (n - 1) // 2)))
    rows = adjacency(n, edges)
    sym = _check_matrix(n, edges, rows)
    deg = degrees(n, edges)
    shape = rng.choice(["degree", "edges", "ones"])
    m_text = matrix_text(rows)
    order = _ORDER.format(order=_order(n))
    if shape == "degree":
        x = rng.randrange(n)
        answer = deg[x]
        check(answer == sum(sym.row(x)) == sum(sym.col(x)), "row sum and column sum differ")
        text = phrase(rng, _DEGREE_STEMS, m=m_text, order=order, x=LETTERS[x])
        candidates = [
            (len(edges), "net-adjacency-edges-for-degree"),
            (sum(sym), "net-adjacency-sum-for-degree"),
            (answer + 1, "net-degree-miscount"),
            (answer - 1, "net-degree-miscount"),
            (n - 1, "net-adjacency-vertices-for-degree"),
        ]
        working = [
            f"The entries in row {LETTERS[x]} are {rows[x]}.",
            f"They add to {answer}, so the degree of {LETTERS[x]} is {answer}.",
        ]
    elif shape == "edges":
        answer = len(edges)
        text = phrase(rng, _EDGES_STEMS, m=m_text, order=order, es=ctx["es"])
        candidates = [
            (sum(sym), "net-adjacency-sum-as-edges"),
            (sum(sym) // 2 + 1, "net-handshake-slip"),
            (sum(sym) // 2 - 1, "net-handshake-slip"),
            (n, "net-vertices-as-edges"),
        ]
        working = [
            f"The entries add to {sum(sym)}.",
            f"Each edge puts a 1 in two places, so there are {sum(sym)} / 2 = {answer} edges.",
        ]
    else:
        answer = 2 * len(edges)
        text = phrase(rng, _ONES_STEMS, net=describe_network(ctx, n, edges), vs=ctx["vs"])
        candidates = [
            (len(edges), "net-adjacency-one-entry-per-edge"),
            (n * n, "net-adjacency-all-entries-counted"),
            (answer + 2, "net-handshake-slip"),
            (answer - 2, "net-handshake-slip"),
        ]
        working = [
            f"The network has {len(edges)} edges.",
            f"Each edge gives a 1 in row i, column j and another in row j, column i, so {2 * len(edges)} entries are 1.",
        ]
    distractors = int_distractors(answer, candidates, lo=1)
    check(len(distractors) >= 2, "too few distractors")
    return _number(text, answer, working, distractors)


# -- medium -----------------------------------------------------------------------------------------------------

_WALK2_STEMS = (
    "A network has the adjacency matrix {m}, {order}. A walk of length 2 uses two edges, one after the other. "
    "How many walks of length 2 are there from {x} to {y}?",
    "A network has the adjacency matrix {m}, {order}. Find the number of walks of length 2 from {x} to {y}.",
)


def _medium(rng):
    for _ in range(200):
        n = rng.randint(4, 6)
        edges = random_connected(rng, n, rng.randint(n + 1, min(n + 4, n * (n - 1) // 2)))
        rows = adjacency(n, edges)
        sym = _check_matrix(n, edges, rows)
        x = rng.randrange(n)
        y = x if rng.random() < 0.25 else rng.choice([v for v in range(n) if v != x])
        answer = _walks2(rows, x, y)
        if 1 <= answer <= 4 and answer != rows[x][y]:
            break
    else:
        raise AssertionError  # pragma: no cover
    check(answer == (sym**2)[x, y], "SymPy's A squared entry differs")
    deg = degrees(n, edges)
    candidates = [
        (rows[x][y], "net-adjacency-walk-length-confused"),
        (rows[x][y] ** 2, "net-adjacency-squared-entrywise"),
        (sum(rows[x]) + sum(rows[k][y] for k in range(n)), "net-adjacency-row-column-added"),
        (deg[x] * deg[y], "net-adjacency-degrees-multiplied"),
        (answer + 1, "net-handshake-slip"),
        (answer - 1, "net-handshake-slip"),
    ]
    distractors = int_distractors(answer, candidates, lo=0)
    check(len(distractors) >= 2, "too few distractors")
    middles = [LETTERS[k] for k in range(n) if rows[x][k] and rows[k][y]]
    working = [
        f"A walk {LETTERS[x]} to k to {LETTERS[y]} needs {LETTERS[x]}k and k{LETTERS[y]} both to be edges.",
        f"The middle vertex k can be {', '.join(middles)}, so the number of such walks is {answer}.",
        f"(This is the entry in row {LETTERS[x]}, column {LETTERS[y]} of the matrix squared.)",
    ]
    text = phrase(
        rng, _WALK2_STEMS, m=matrix_text(rows), order=_ORDER.format(order=_order(n)), x=LETTERS[x], y=LETTERS[y]
    )
    return _number(text, answer, working, distractors)


# -- hard -------------------------------------------------------------------------------------------------------

_WALK3_STEMS = (
    "A network has the adjacency matrix {m}, {order}. How many walks of length 3 are there from {x} to {y}?",
    "A network has the adjacency matrix {m}, {order}. A walk of length 3 uses three edges one after the other, and may "
    "revisit vertices. Find the number of walks of length 3 from {x} to {y}.",
)
_START2_STEMS = (
    "A network has the adjacency matrix {m}, {order}. A walk of length 2 uses two edges, one after the other, and may "
    "return to where it started. How many walks of length 2 start at {x}?",
    "A network has the adjacency matrix {m}, {order}. Find the number of walks of length 2 that start at vertex {x}, "
    "ending at any vertex.",
)


def _hard(rng):
    n = rng.randint(4, 5)
    edges = random_connected(rng, n, rng.randint(n + 1, min(n + 3, n * (n - 1) // 2)))
    rows = adjacency(n, edges)
    sym = _check_matrix(n, edges, rows)
    deg = degrees(n, edges)
    order = _ORDER.format(order=_order(n))
    x = rng.randrange(n)
    if rng.random() < 0.5:
        for _ in range(100):
            y = rng.randrange(n)
            answer = _walks3(rows, x, y)
            if 2 <= answer <= 12 and answer != _walks2(rows, x, y):
                break
        else:
            raise AssertionError  # pragma: no cover
        check(answer == (sym**3)[x, y], "SymPy's A cubed entry differs")
        candidates = [
            (_walks2(rows, x, y), "net-adjacency-walk-length-confused"),
            (rows[x][y] ** 3, "net-adjacency-squared-entrywise"),
            ((sym**2)[x, y] * deg[y] if deg[y] else -1, "net-adjacency-degrees-multiplied"),
            (answer + 1, "net-handshake-slip"),
            (answer - 1, "net-handshake-slip"),
        ]
        text = phrase(rng, _WALK3_STEMS, m=matrix_text(rows), order=order, x=LETTERS[x], y=LETTERS[y])
        working = [
            f"The number of walks of length 3 from {LETTERS[x]} to {LETTERS[y]} is the entry in row {LETTERS[x]}, column {LETTERS[y]} of the matrix cubed.",
            f"Multiplying the matrix by itself three times gives {answer}.",
        ]
    else:
        answer = sum(_walks2(rows, x, y) for y in range(n))
        check(answer == sum((sym**2).row(x)), "SymPy's row sum of A squared differs")
        check(answer == sum(deg[k] for k in range(n) if rows[x][k]), "sum of neighbours' degrees differs")
        candidates = [
            (deg[x], "net-adjacency-one-step-only"),
            (sum(sum((sym**2).row(i)) for i in range(n)), "net-adjacency-all-starts-counted"),
            (answer + 1, "net-handshake-slip"),
            (answer - 1, "net-handshake-slip"),
            (sum(deg), "net-degree-sum-as-walks"),
        ]
        text = phrase(rng, _START2_STEMS, m=matrix_text(rows), order=order, x=LETTERS[x])
        nbrs = [LETTERS[k] for k in range(n) if rows[x][k]]
        working = [
            f"A walk of length 2 from {LETTERS[x]} goes to a neighbour ({', '.join(nbrs)}) and then along any edge from there.",
            "So the number of walks is the sum of the neighbours' degrees: "
            + " + ".join(str(deg[k]) for k in range(n) if rows[x][k])
            + f" = {answer}.",
        ]
    distractors = int_distractors(answer, candidates, lo=0)
    check(len(distractors) >= 2, "too few distractors")
    return _number(text, answer, working, distractors)


@generator(
    "snet_adjacency",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def snet_adjacency(rng, tier="easy"):
    """Adjacency matrices of networks and walks counted with their powers; the tiers are in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


def _matrix_distractors(rng, n, edges, rows):
    """Wrong adjacency matrices, each tagged: ones on the diagonal, one entry per edge, an edge missed, two vertices
    swapped. Any the matrix grader would accept (the same matrix) is dropped."""
    wrong: list[Distractor] = []

    def add(candidate, tag):
        if matrices_match(candidate, rows) or any(matrices_match(candidate, d.value) for d in wrong):
            return
        wrong.append(Distractor(matrix_text(candidate), tag, candidate))

    add([[1 if i == j else rows[i][j] for j in range(n)] for i in range(n)], "net-adjacency-diagonal-ones")
    add([[rows[i][j] if i < j else 0 for j in range(n)] for i in range(n)], "net-adjacency-one-entry-per-edge")
    u, v = rng.choice(edges)
    missed = [row[:] for row in rows]
    missed[u][v] = missed[v][u] = 0
    add(missed, "net-adjacency-edge-missed")
    a, b = rng.sample(range(n), 2)
    perm = list(range(n))
    perm[a], perm[b] = perm[b], perm[a]
    add([[rows[perm[i]][perm[j]] for j in range(n)] for i in range(n)], "net-adjacency-wrong-vertex-order")
    return wrong


@generator("snet_adjacency_chain", level=12, presets={"exam": dict(tier="exam")})
def snet_adjacency_chain(rng, tier="exam"):
    """Exam chain on one network: its adjacency matrix, then the walks of length 2 from a vertex."""
    check(tier == "exam", f"unknown tier {tier!r}")
    ctx = pick_context(rng)
    for _ in range(200):
        n = rng.randint(5, 6)
        edges = random_connected(rng, n, rng.randint(n + 1, n + 3))
        rows = adjacency(n, edges)
        sym = _check_matrix(n, edges, rows)
        deg = degrees(n, edges)
        x = rng.choice([v for v in range(n) if deg[v] >= 2])
        total = sum(_walks2(rows, x, y) for y in range(n))
        closed = _walks2(rows, x, x)
        if closed == deg[x] and total - closed >= 3:
            break
    else:
        raise AssertionError  # pragma: no cover
    check(total == sum((sym**2).row(x)) and closed == (sym**2)[x, x], "SymPy's A squared differs")
    other = total - closed
    xn = LETTERS[x]
    stem = (
        f"{describe_network(ctx, n, edges)} The adjacency matrix A of the network has rows and columns in the order "
        f"{_order(n)}. A walk of length 2 uses two edges, one after the other."
    )
    b_dist = int_distractors(
        total,
        [
            (deg[x], "net-adjacency-one-step-only"),
            (sum(sum((sym**2).row(i)) for i in range(n)), "net-adjacency-all-starts-counted"),
            (total + 1, "net-handshake-slip"),
            (total - 1, "net-handshake-slip"),
        ],
    )
    c_dist = int_distractors(
        closed,
        [
            (total, "net-adjacency-ends-not-restricted"),
            (other, "net-adjacency-ends-elsewhere"),
            (closed + 1, "net-handshake-slip"),
            (closed - 1, "net-handshake-slip"),
        ],
        lo=1,
    )
    d_dist = int_distractors(
        other,
        [
            (total, "net-closed-walks-not-subtracted"),
            (closed, "net-adjacency-ends-elsewhere"),
            (other + 1, "net-handshake-slip"),
            (other - 1, "net-handshake-slip"),
        ],
        lo=1,
    )
    a_dist = _matrix_distractors(rng, n, edges, rows)
    check(all(len(x_) >= 2 for x_ in (a_dist, b_dist, c_dist, d_dist)), "too few distractors")
    nbrs = [LETTERS[k] for k in range(n) if rows[x][k]]
    return dict(
        text=stem,
        parts=[
            dict(
                prompt="Write down the adjacency matrix A of the network.",
                answer_kind="matrix",
                answer=rows,
                answer_text=matrix_text(rows),
                answer_latex=matrix_latex(rows),
                marks=2,
                working=[
                    "Put a 1 in row i, column j when vertices i and j are joined by an edge, and a 0 otherwise:",
                    matrix_text(rows),
                ],
                distractors=a_dist,
            ),
            dict(
                prompt=f"How many walks of length 2 start at {xn}? A walk may end at any vertex.",
                answer_kind="number",
                answer=total,
                answer_text=str(total),
                answer_latex=str(total),
                marks=2,
                sensible_range=(0, None),
                working=[
                    f"The row for {xn} in the matrix A squared adds to the number of walks of length 2 from {xn}.",
                    f"Going to a neighbour of {xn} ({', '.join(nbrs)}) and then along any edge: "
                    + " + ".join(str(deg[k]) for k in range(n) if rows[x][k])
                    + f" = {total}.",
                ],
                distractors=b_dist,
            ),
            dict(
                prompt=f"How many of the walks of length 2 that start at {xn} end back at {xn}?",
                answer_kind="number",
                answer=closed,
                answer_text=str(closed),
                answer_latex=str(closed),
                marks=1,
                sensible_range=(0, None),
                working=[
                    f"Such a walk goes out along an edge at {xn} and straight back, so there is one for each edge at {xn}: {closed}."
                ],
                distractors=c_dist,
            ),
            dict(
                prompt=f"Hence find how many walks of length 2 that start at {xn} end at a vertex other than {xn}.",
                answer_kind="number",
                answer=other,
                answer_text=str(other),
                answer_latex=str(other),
                marks=1,
                sensible_range=(0, None),
                follow_through="part_b - part_c",
                working=[
                    f"{total} walks start at {xn} and {closed} of them end at {xn}, so {total} - {closed} = {other} end elsewhere."
                ],
                distractors=d_dist,
            ),
        ],
    )
