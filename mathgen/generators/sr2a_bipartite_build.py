"""Generates `bipartite-matching-matrix` questions (easy, medium, hard) and the exam chain `sr2a_bipartite_chain`.

Representing an allocation problem as a bipartite graph and its 0/1 matrix, then reasoning with it (ACMGM110,
`financial-discrete`, answer ``matrix``). Workers and jobs are two sets of vertices, and an edge joins a worker to a
job the worker can do. The prompt never draws the graph: it says who can do what in words, and asks for the matrix
with the workers as rows (in the order listed) and the jobs as columns (in the order listed), 1 where the worker
can do the job and 0 where not.

* easy: each worker's jobs are listed; write the matrix;
* medium: the same matrix from a less direct description: either each JOB lists the workers who can do it (so the
  matrix is the transpose of what is read), or each worker lists the jobs they CANNOT do (so the 1s are the jobs left
  over);
* hard: a reduction step. Every job that only one worker can do is given to that worker, and both then leave the
  problem; write the matrix of the workers and jobs still unassigned, in their original order. Removing a worker
  removes that worker's other options with them;
* exam (``sr2a_bipartite_chain``): a problem in which some workers cannot all be placed. (a) write the matrix;
  (b) the most jobs that can be filled at once; (c) how many workers cannot be given a job in a best allocation;
  (d) in how many different ways the best number of jobs can be filled.

Matrices are built first and the text is written from them. The maximum matching and the number of maximum matchings
are found by exhaustive search over assignments (at most five workers); the forced pairs are found from column sums.
"""

from __future__ import annotations

from ..core import attempts, check, generator
from ._sr2a_kit import matrix_part, matrix_single, number_part

_PEOPLE = ("Ana", "Ben", "Cara", "Dev", "Ella", "Finn", "Grace", "Hugo")
_CONTEXTS = (
    dict(
        who="tradespeople",
        one="tradesperson",
        task="job",
        tasks="jobs",
        intro="A building firm has {n} tradespeople and {n} jobs to hand out, one job to each person.",
        jobs=("painting", "wiring", "tiling", "plumbing", "carpentry", "roofing", "glazing", "paving"),
        can="can do",
        cannot="cannot do",
        only="can be done by only",
    ),
    dict(
        who="students",
        one="student",
        task="lab station",
        tasks="lab stations",
        intro="A school has {n} students to place at {n} lab stations, one student to each station.",
        jobs=("microscopes", "titration", "circuits", "dissection", "chemistry", "robotics", "optics", "geology"),
        can="can work at",
        cannot="cannot work at",
        only="can be used by only",
    ),
    dict(
        who="drivers",
        one="driver",
        task="route",
        tasks="routes",
        intro="A bus company has {n} drivers to roster onto {n} routes, one driver to each route.",
        jobs=("coast", "airport", "hills", "city loop", "university", "harbour", "valley", "north shore"),
        can="can drive",
        cannot="cannot drive",
        only="can be driven by only",
    ),
)


def _setup(rng, rows: int, cols: int):
    ctx = rng.choice(_CONTEXTS)
    people = sorted(rng.sample(_PEOPLE, rows))
    jobs = rng.sample(ctx["jobs"], cols)
    return ctx, people, jobs


def _names(items) -> str:
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _lists_by_worker(people, jobs, m, ctx) -> str:
    return " ".join(
        f"{p} {ctx['can']} {_names(jobs[j] for j in range(len(jobs)) if m[i][j])}." for i, p in enumerate(people)
    )


def _order_note(ctx, people, jobs) -> str:
    return (
        f"Write the matrix with a row for each of {_names(people)}, in that order, and a column for each of "
        f"{_names(jobs)}, in that order, with 1 where the {ctx['one']} can be given the {ctx['task']} and 0 where "
        "not. Write it as rows, for example [[1, 0], [0, 1]]."
    )


def _transpose(m):
    return [list(r) for r in zip(*m)]


def _complement(m):
    return [[1 - v for v in r] for r in m]


def _flip(m, i, j):
    out = [list(r) for r in m]
    out[i][j] = 1 - out[i][j]
    return out


def _slips(m, rng, *, symmetric_ok=False):
    """Matrix slips common to every tier: transposed, ones and zeros swapped, one entry wrong, two rows swapped."""
    rows, cols = len(m), len(m[0])
    out = []
    if rows == cols:
        out.append((_transpose(m), "sr2a-bip-rows-columns-swapped"))
    out.append((_complement(m), "sr2a-bip-ones-zeros-swapped"))
    for _ in range(3):
        i, j = rng.randrange(rows), rng.randrange(cols)
        out.append((_flip(m, i, j), "sr2a-bip-entry-slip"))
    if rows >= 2:
        swapped = [list(r) for r in m]
        i, j = rng.sample(range(rows), 2)
        swapped[i], swapped[j] = swapped[j], swapped[i]
        out.append((swapped, "sr2a-bip-row-order-slip"))
    return out


def _random_matrix(rng, n: int, per_row=(2, 3), cols: int | None = None):
    cols = cols or n
    m = [[0] * cols for _ in range(n)]
    for i in range(n):
        for j in rng.sample(range(cols), rng.randint(*per_row)):
            m[i][j] = 1
    return m


def _col_sums(m):
    return [sum(r[j] for r in m) for j in range(len(m[0]))]


# -- easy -------------------------------------------------------------------------------------------------------


def _easy(rng):
    for _ in attempts(300, "an easy allocation matrix"):
        n = rng.randint(4, 5)
        m = _random_matrix(rng, n)
        if min(sum(r) for r in m) < 1 or min(_col_sums(m)) < 1 or m == _transpose(m):
            continue
        ctx, people, jobs = _setup(rng, n, n)
        text = (
            ctx["intro"].format(n=n)
            + " "
            + _lists_by_worker(people, jobs, m, ctx)
            + " "
            + _order_note(ctx, people, jobs)
        )
        working = [
            f"Each row is a {ctx['one']} and each column is a {ctx['task']}. Put 1 where the {ctx['one']} can do it.",
            *[f"{p}: {m[i]}" for i, p in enumerate(people)],
        ]
        q = matrix_single(text, m, working, _slips(m, rng))
        if q:
            return q
    raise AssertionError  # pragma: no cover


# -- medium -----------------------------------------------------------------------------------------------------


def _medium(rng):
    for _ in attempts(300, "a medium allocation matrix"):
        n = rng.randint(4, 5)
        ctx, people, jobs = _setup(rng, n, n)
        if rng.random() < 0.5:
            m = _random_matrix(rng, n)
            if min(sum(r) for r in m) < 1 or min(_col_sums(m)) < 1 or m == _transpose(m):
                continue
            lines = " ".join(
                f"{jobs[j].capitalize()}: {_names(people[i] for i in range(n) if m[i][j])}." for j in range(n)
            )
            text = (
                ctx["intro"].format(n=n)
                + f" For each {ctx['task']}, the {ctx['who']} who can be given it are listed. "
                + lines
                + " "
                + _order_note(ctx, people, jobs)
            )
            wrong = _slips(m, rng)
            wrong.insert(0, (_transpose(m), "sr2a-bip-rows-columns-swapped"))
            working = [
                f"The lists are by {ctx['task']}, so each list gives a column of the matrix, not a row.",
                *[f"{jobs[j]}: column {[m[i][j] for i in range(n)]}" for j in range(n)],
            ]
        else:
            # each worker can do every job except the ones listed
            m = [[1] * n for _ in range(n)]
            for i in range(n):
                for j in rng.sample(range(n), rng.randint(1, 2)):
                    m[i][j] = 0
            if min(sum(r) for r in m) < 1 or min(_col_sums(m)) < 1:
                continue
            lines = " ".join(
                f"{p} {ctx['cannot']} {_names(jobs[j] for j in range(n) if not m[i][j])}." for i, p in enumerate(people)
            )
            text = (
                ctx["intro"].format(n=n)
                + f" Each {ctx['one']} can be given every {ctx['task']} except the ones listed for them. "
                + lines
                + " "
                + _order_note(ctx, people, jobs)
            )
            wrong = [(_complement(m), "sr2a-bip-cannot-as-can"), *_slips(m, rng)]
            working = [
                f"Each {ctx['one']} can be given every {ctx['task']} except those listed, so a listed one is a 0 and every other is a 1.",
                *[f"{p}: {m[i]}" for i, p in enumerate(people)],
            ]
        q = matrix_single(text, m, working, wrong)
        if q:
            return q
    raise AssertionError  # pragma: no cover


# -- hard -------------------------------------------------------------------------------------------------------


def _forced_pairs(m):
    """(worker, job) for every job that exactly one worker can do."""
    return [(next(i for i in range(len(m)) if m[i][j]), j) for j, s in enumerate(_col_sums(m)) if s == 1]


def _reduce(m, pairs):
    gone_rows = {i for i, _ in pairs}
    gone_cols = {j for _, j in pairs}
    return [[m[i][j] for j in range(len(m[0])) if j not in gone_cols] for i in range(len(m)) if i not in gone_rows]


def _hard(rng):
    for _ in attempts(2000, "a reducible allocation matrix"):
        n = 5
        m = _random_matrix(rng, n, per_row=(2, 3))
        pairs = _forced_pairs(m)
        rows = [i for i, _ in pairs]
        if (
            not 1 <= len(pairs) <= 2
            or len(set(rows)) != len(rows)
            or min(sum(r) for r in m) < 1
            or min(_col_sums(m)) < 1
        ):
            continue
        # every forced worker has another option, so removing them matters
        if any(sum(m[i]) < 2 for i, _ in pairs):
            continue
        reduced = _reduce(m, pairs)
        if len(reduced) < 3 or not reduced[0]:
            continue
        ctx, people, jobs = _setup(rng, n, n)
        text = (
            ctx["intro"].format(n=n)
            + " "
            + _lists_by_worker(people, jobs, m, ctx)
            + f" Every {ctx['task']} that only one {ctx['one']} can be given is given to that {ctx['one']}, and both then "
            f"leave the problem. Write the matrix for the {ctx['who']} and {ctx['tasks']} that are still unassigned. "
            f"Start from a row for each of {_names(people)}, in that order, and a column for each of {_names(jobs)}, "
            f"in that order, with 1 where the {ctx['one']} can be given the {ctx['task']} and 0 where not, and keep "
            "only the rows and columns that remain. Write it as rows, for example [[1, 0], [0, 1]]."
        )
        gone_cols = {j for _, j in pairs}
        gone_rows = {i for i, _ in pairs}
        # slips: the unreduced matrix, a forced worker's row kept, a forced job's column kept, the transpose
        keep_row = [
            [m[i][j] for j in range(n) if j not in gone_cols] for i in range(n) if i not in (gone_rows - {rows[0]})
        ]
        keep_col = [
            [m[i][j] for j in range(n) if j not in (gone_cols - {pairs[0][1]})] for i in range(n) if i not in gone_rows
        ]
        wrong = [
            (keep_row, "sr2a-bip-forced-worker-kept"),
            (keep_col, "sr2a-bip-forced-job-kept"),
            (m, "sr2a-bip-no-reduction"),
            (_transpose(reduced), "sr2a-bip-rows-columns-swapped") if len(reduced) == len(reduced[0]) else (m, None),
            *_slips(reduced, rng),
        ]
        wrong = [(w, t) for w, t in wrong if t]
        forced_text = ", ".join(f"{people[i]} gets {jobs[j]}" for i, j in pairs)
        working = [
            f"The {ctx['tasks']} that only one {ctx['one']} can be given are given out: {forced_text}.",
            f"Delete those {ctx['who']}' rows and those {ctx['tasks']}' columns, keeping the order of what is left.",
            *[f"{people[i]}: {r}" for i, r in zip([i for i in range(n) if i not in gone_rows], reduced)],
        ]
        q = matrix_single(text, reduced, working, wrong)
        if q:
            return q
    raise AssertionError  # pragma: no cover


@generator(
    "sr2a_bipartite",
    level=12,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def sr2a_bipartite(rng, tier="easy"):
    """Write the 0/1 matrix of an allocation problem; the tiers are described in the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------


def _matchings(m):
    """(largest matching size, how many matchings have that size) by exhaustive search over assignments."""
    rows, cols = len(m), len(m[0])
    sizes: dict[int, int] = {}

    def go(i, used, size):
        if i == rows:
            sizes[size] = sizes.get(size, 0) + 1
            return
        go(i + 1, used, size)
        for j in range(cols):
            if m[i][j] and not used >> j & 1:
                go(i + 1, used | 1 << j, size + 1)

    go(0, 0, 0)
    best = max(sizes)
    return best, sizes[best]


def _hall_matrix(rng, n: int, deficiency_set: int):
    """A matrix where ``deficiency_set`` workers can only do ``deficiency_set - 1`` jobs between them."""
    jobs = list(range(n))
    rng.shuffle(jobs)
    stuck_jobs = jobs[: deficiency_set - 1]
    stuck_workers = rng.sample(range(n), deficiency_set)
    m = [[0] * n for _ in range(n)]
    for i in range(n):
        if i in stuck_workers:
            for j in rng.sample(stuck_jobs, rng.randint(1, len(stuck_jobs))):
                m[i][j] = 1
        else:
            for j in rng.sample(range(n), rng.randint(2, 3)):
                m[i][j] = 1
    return m


@generator("sr2a_bipartite_chain", level=12, presets={"exam": dict(tier="exam")})
def sr2a_bipartite_chain(rng, tier="exam"):
    """Exam chain: the matrix, the most jobs that can be filled, the workers left over, the number of best allocations."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(500, "an allocation problem with a shortage"):
        n = rng.randint(4, 5)
        m = _hall_matrix(rng, n, rng.randint(2, 3))
        if min(sum(r) for r in m) < 1 or m == _transpose(m):
            continue
        best, ways = _matchings(m)
        left = n - best
        if left < 1 or ways < 2 or ways > 60:
            continue
        ctx, people, jobs = _setup(rng, n, n)
        stem = (
            ctx["intro"].format(n=n)
            + " "
            + _lists_by_worker(people, jobs, m, ctx)
            + f" Not every {ctx['one']} will be given a {ctx['task']}."
        )
        ones = sum(map(sum, m))
        part_a = matrix_part(
            _order_note(ctx, people, jobs),
            m,
            2,
            [
                f"Each row is a {ctx['one']} and each column is a {ctx['task']}, with 1 where the {ctx['one']} can do it."
            ],
            _slips(m, rng),
        )
        part_b = number_part(
            f"What is the largest number of {ctx['tasks']} that can be filled at the same time, each by a different {ctx['one']}?",
            best,
            2,
            [
                f"Some {ctx['who']} can only take the same few {ctx['tasks']} between them, so at most {best} can be filled."
            ],
            [
                (n, "sr2a-bip-assumes-full"),
                (best + 1, "sr2a-bip-count-slip"),
                (best - 1, "sr2a-bip-count-slip"),
                (ones, "sr2a-bip-counts-ones"),
                (n - 1 if best != n - 1 else None, "sr2a-bip-count-slip"),
            ],
            lo=1,
        )
        part_c = number_part(
            f"In a best allocation, how many {ctx['who']} are not given a {ctx['task']}?",
            left,
            1,
            [f"{n} - {best} = {left}."],
            [
                (best, "sr2a-bip-jobs-for-left-over"),
                (left + 1, "sr2a-bip-count-slip"),
                (left - 1 if left > 1 else None, "sr2a-bip-count-slip"),
                (0, "sr2a-bip-assumes-full"),
                (n, "sr2a-bip-assumes-full"),
            ],
            follow=f"{n} - part_b",
            lo=0,
            sensible=(-1, None),
        )
        prod = 1
        for r in m:
            prod *= sum(r)
        part_d = number_part(
            f"In how many different ways can {best} {ctx['tasks']} be filled, each by a different {ctx['one']}? Two "
            f"ways are different if they pair the {ctx['who']} and {ctx['tasks']} differently.",
            ways,
            3,
            [
                f"Counting every way to give {best} {ctx['tasks']} to {best} different {ctx['who']} who can take them gives {ways}."
            ],
            [
                (ones, "sr2a-bip-counts-ones"),
                (prod if prod <= 400 else None, "sr2a-bip-multiplies-choices"),
                (ways + 1, "sr2a-bip-count-slip"),
                (ways - 1, "sr2a-bip-count-slip"),
                (1, "sr2a-bip-one-allocation"),
                (ways * 2, "sr2a-bip-count-slip"),
            ],
            lo=1,
        )
        if None in (part_a, part_b, part_c, part_d):
            continue
        return dict(text=stem, parts=[part_a, part_b, part_c, part_d])
    raise AssertionError  # pragma: no cover
