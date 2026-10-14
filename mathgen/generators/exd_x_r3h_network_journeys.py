"""Exam tier for storing information in a matrix: a three-part chain on a directed bus network.

The table is the network's adjacency matrix A: entry (i, j) is the number of direct bus services per day from town i
to town j. The practice tiers read one row or column total from it. Here the matrix is used to count journeys:

(a) the total number of direct services leaving a town P (a row total);
(b) the number of ways to travel from P to a town Q using exactly two services, which is entry (P, Q) of A squared;
(c) the number of ways to travel from P to Q using at most two services, the direct services plus the answer to (b).

Part (c) follows through from the learner's (b).
"""

from __future__ import annotations

from fractions import Fraction

import sympy

from ..core import attempts, check, generator
from ._exd_distinct_parts import distinct_part_answers
from ._exd_x_r3h_kit import number_part

_TOWNS = [
    "Ashford",
    "Bendigo Creek",
    "Carrington",
    "Denton",
    "Fairview",
    "Glenrock",
    "Milbrook",
    "Rosewood",
    "Wattle Bay",
    "Yarrow",
]


def _render(names, matrix) -> str:
    header = "Town | " + " | ".join(names)
    rows = "; ".join(f"{names[i]} | " + " | ".join(str(v) for v in matrix[i]) for i in range(len(names)))
    return f"{header}; {rows}"


@generator("exd_x_r3h_network_journeys", level=14, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3h_network_journeys(rng, tier="exam"):
    if tier != "exam":
        raise ValueError(f"unknown tier {tier!r}")
    for _ in attempts(2000, "a network with distinct, non-trivial journey counts"):
        n = rng.choice((4, 5))
        names = rng.sample(_TOWNS, n)
        matrix = [[0 if i == j else rng.choice((0, 0, 1, 1, 2, 3, 4)) for j in range(n)] for i in range(n)]
        p, q = rng.sample(range(n), 2)
        direct = matrix[p][q]
        two_stage = sum(matrix[p][k] * matrix[k][q] for k in range(n))
        row_p = sum(matrix[p])
        col_p = sum(matrix[i][p] for i in range(n))
        if direct == 0 or matrix[q][p] == direct or two_stage < 3 or row_p == col_p:
            continue
        others = [k for k in range(n) if k not in (p, q)]
        slips_a = {row_p, col_p, sum(matrix[q]), row_p - max(matrix[p])}
        reverse = sum(matrix[q][k] * matrix[k][p] for k in range(n))
        added = sum(matrix[p][k] + matrix[k][q] for k in others)
        best = max(matrix[p][k] * matrix[k][q] for k in others)
        rowrow = sum(matrix[p][k] * matrix[q][k] for k in range(n))
        slips_b = {two_stage, reverse, added, best, rowrow}
        if len(slips_a) < 4 or len(slips_b) < 4 or direct * two_stage in (two_stage, direct + two_stage):
            continue
        break
    at_most = direct + two_stage
    check((sympy.Matrix(matrix) ** 2)[p, q] == two_stage, "two-stage count is the (P, Q) entry of A squared")
    P, Q = names[p], names[q]
    stem = (
        f"A regional bus company runs direct services between {n} towns. The table gives the number of direct "
        f"services per day from the town in each row to the town in each column: {_render(names, matrix)}. "
        "A journey may use several services, changing buses at intermediate towns."
    )
    return dict(
        text=stem,
        latex=f"plain:{stem}",
        parts=[
            dict(
                **number_part(
                    f"Find the total number of direct services per day leaving {P}.",
                    Fraction(row_p),
                    0,
                    1,
                    [f"Add the entries in the row for {P}: {' + '.join(str(v) for v in matrix[p])} = {row_p}."],
                    [
                        (col_p, "opposite-direction-total"),
                        (sum(matrix[q]), "read-wrong-row-or-column"),
                        (row_p - max(matrix[p]), "omitted-entry-in-total"),
                    ],
                    sensible_range=(0, None),
                )
            ),
            dict(
                **number_part(
                    f"Find the number of different ways to travel from {P} to {Q} using exactly two services "
                    "(one change of bus), counting each bus service separately.",
                    Fraction(two_stage),
                    0,
                    3,
                    [
                        f"For each intermediate town, multiply the services from {P} to it by the services from it to {Q}.",
                        f"Sum over the towns: {two_stage}.",
                    ],
                    [
                        (reverse, "reversed-direction"),
                        (added, "added-instead-of-multiplied"),
                        (best, "counted-only-the-best-intermediate-town"),
                        (rowrow, "multiplied-row-by-row-not-row-by-column"),
                    ],
                    sensible_range=(0, None),
                )
            ),
            dict(
                **number_part(
                    f"Hence find the number of different ways to travel from {P} to {Q} using at most two services.",
                    Fraction(at_most),
                    0,
                    2,
                    [
                        f"Direct services {direct} plus the two-service journeys from (b): {direct} + {two_stage} = {at_most}."
                    ],
                    [
                        (two_stage, "left-out-direct-services"),
                        (direct * two_stage, "multiplied-instead-of-added"),
                        (direct, "left-out-two-service-journeys"),
                        (two_stage + matrix[q][p], "reversed-direction"),
                    ],
                    follow_through=f"part_b + {direct}",
                    sensible_range=(0, None),
                )
            ),
        ],
    )
