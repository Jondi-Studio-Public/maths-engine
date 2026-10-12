"""Generates `exd_x_r3j_matrix_square_parameter` questions: a matrix whose size depends on a parameter.

Skill: `senior-recognise-different-types-of-matrices-row-column-square` (exam tier). The practice tiers classify a
stated matrix and read off its size. The exam gives a matrix by a rule for a(i,j) and ranges for i and j that
depend on a parameter m, and says the matrix is square; the learner has to use that classification:

(a) the value of m that makes A square                                   number, 1 mark
(b) the number of rows (= columns) of A                                  number, 1 mark,  m + p
(c) the sum of the diagonal entries                                      number, 2 marks, T(part_b)
(d) the sum of all the entries                                           number, 2 marks, S(part_b)

## Construction

Rows run 1 to m + p and columns 1 to 2m - q, so A is square exactly when m = p + q and then has p + q + p = 2p + q
rows. Two rule families are used, a linear rule c i + d j + e and a product rule k i j + e, whose diagonal and total
sums have closed forms in the order n:

* linear:   T(n) = (c + d) n (n + 1) / 2 + e n          S(n) = ((c + d)(n + 1) / 2 + e) n^2
* product:  T(n) = k n (n + 1) (2n + 1) / 6 + e n       S(n) = k (n (n + 1) / 2)^2 + e n^2

Every canonical answer is also summed entry by entry as an independent check.
"""

from __future__ import annotations

from ..core import attempts, check, generator, join_terms
from ._exd_x_r3_distinct import distinct_answers
from ._exd_x_r3j_kit import number_part

_STEMS = [
    "A matrix A has entries a(i,j) = {rule} for i = 1, ..., {rows} and j = 1, ..., {cols}, where m is a "
    "positive integer, and A is a square matrix.",
    "Let A be the matrix with a(i,j) = {rule}, where i runs from 1 to {rows} and j runs from 1 to {cols} for a "
    "positive integer m. A is known to be a square matrix.",
    "The entries of a matrix A are given by a(i,j) = {rule}, for i = 1, ..., {rows} and j = 1, ..., {cols} "
    "(m a positive integer). A is square.",
]


def _linear_sums(c: int, d: int, e: int, n: int) -> tuple[int, int]:
    trace = (c + d) * n * (n + 1) // 2 + e * n
    total = ((c + d) * (n + 1) * n * n) // 2 + e * n * n
    return trace, total


def _product_sums(k: int, e: int, n: int) -> tuple[int, int]:
    trace = k * n * (n + 1) * (2 * n + 1) // 6 + e * n
    total = k * (n * (n + 1) // 2) ** 2 + e * n * n
    return trace, total


@generator("exd_x_r3j_matrix_square_parameter", level=11, presets={"exam": {}})
@distinct_answers
def exd_x_r3j_matrix_square_parameter(rng):
    p, q = rng.randint(1, 3), rng.randint(1, 4)
    m = p + q
    n = m + p  # = 2m - q
    check(n == 2 * m - q, "rows and columns disagree at the solution")
    family = rng.choice(["linear", "linear", "product"])
    if family == "linear":
        for _ in attempts(500, "a linear rule with a non-trivial diagonal"):
            c, d, e = rng.randint(-3, 4), rng.randint(-3, 4), rng.randint(-5, 5)
            if c != 0 and d != 0 and c + d != 0:
                break
        rule = join_terms([(c, "i"), (d, "j"), (e, "")])
        entry = lambda i, j: c * i + d * j + e  # noqa: E731
        trace, total = _linear_sums(c, d, e, n)
        ft_trace = f"({c + d})*(part_b)*(part_b + 1)/2 + ({e})*part_b"
        ft_total = f"(({c + d})*(part_b + 1)/2 + ({e}))*part_b**2"
        trace_slips = [
            ((c + d) * n * (n + 1) // 2, "dropped-constant-term"),
            ((c + d) * n * (n + 1) + e * n, "dropped-factor-of-two"),
            (n * entry(n, n), "all-entries-taken-as-last-entry"),
            ((c + d) * (n - 1) * n // 2 + e * n, "off-by-one-count"),
        ]
        total_slips = [
            (total - e * n * n, "dropped-constant-term"),
            (trace, "gave-trace-for-total"),
            (n * n * entry(n, n), "all-entries-taken-as-last-entry"),
            (trace * n, "multiplied-trace-by-order"),
        ]
    else:
        k = rng.randint(1, 3)
        e = rng.randint(-5, 5)
        rule = join_terms([(k, "i·j"), (e, "")])
        entry = lambda i, j: k * i * j + e  # noqa: E731
        trace, total = _product_sums(k, e, n)
        ft_trace = f"{k}*part_b*(part_b + 1)*(2*part_b + 1)/6 + ({e})*part_b"
        ft_total = f"{k}*(part_b*(part_b + 1)/2)**2 + ({e})*part_b**2"
        trace_slips = [
            (k * n * (n + 1) * (2 * n + 1) // 6, "dropped-constant-term"),
            (k * (n * (n + 1) // 2) + e * n, "summed-i-not-i-squared-on-diagonal"),
            (n * entry(n, n), "all-entries-taken-as-last-entry"),
            (k * n * (n + 1) * (2 * n + 1) // 3 + e * n, "dropped-factor-of-two"),
        ]
        total_slips = [
            (total - e * n * n, "dropped-constant-term"),
            (trace, "gave-trace-for-total"),
            (k * (n * (n + 1) // 2) + e * n * n, "summed-i-not-i-squared-on-diagonal"),
            (trace * n, "multiplied-trace-by-order"),
        ]
    # independent route: add the entries up one at a time
    check(sum(entry(i, i) for i in range(1, n + 1)) == trace, "diagonal sum disagrees with entry-by-entry sum")
    check(sum(entry(i, j) for i in range(1, n + 1) for j in range(1, n + 1)) == total, "total disagrees")

    rows_text = f"m + {p}"
    cols_text = f"2m - {q}"
    stem = rng.choice(_STEMS).format(rule=rule, rows=rows_text, cols=cols_text)
    parts = [
        number_part(
            "Find the value of m.",
            m,
            0,
            1,
            [f"A is square, so m + {p} = 2m − {q}, giving m = {m}."],
            [
                (value, tag)
                for value, tag in (
                    (abs(p - q), "sign-slip-in-formula"),
                    (n, "order-and-parameter-confused"),
                    (p * q, "multiplied-instead-of-added"),
                    (q + 2 * p, "sign-slip-in-formula"),
                    (q, "read-constant-as-parameter"),
                    (p, "read-constant-as-parameter"),
                )
                if value >= 1
            ],
            sensible_range=(0, None),
        ),
        number_part(
            "Find the number of rows of A.",
            n,
            0,
            1,
            [f"The number of rows is m + {p} = {m} + {p} = {n}."],
            [
                (m, "order-and-parameter-confused"),
                (n * n, "gave-entries-for-rows"),
                (2 * m, "doubled-instead-of-shifted"),
            ],
            follow_through=f"part_a + {p}",
            sensible_range=(0, None),
        ),
        number_part(
            "Find the sum of the entries on the main diagonal of A.",
            trace,
            0,
            2,
            [f"The diagonal entries are a(i,i); their sum over i = 1, ..., {n} is {trace}."],
            trace_slips,
            follow_through=ft_trace,
        ),
        number_part(
            "Find the sum of all the entries of A.",
            total,
            0,
            2,
            [f"Summing a(i,j) over i, j = 1, ..., {n} gives {total}."],
            total_slips,
            follow_through=ft_total,
        ),
    ]
    return dict(text=stem, parts=parts)
