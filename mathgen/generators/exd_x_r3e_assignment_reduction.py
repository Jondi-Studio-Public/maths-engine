"""Exam-tier source for `senior-determine-the-optimum-assignment-s-by-inspection-for`: the reduction steps behind an
optimal assignment of four workers to four tasks.

A 4 x 4 table of costs is drawn and the chain follows the standard reduction:

(a) the total of the row minima                                number, 1 mark
(b) the total of the column minima left after (a)              number, 1 mark
(c) the minimum total cost of the assignment                   number, 2 marks   part_a + part_b
(d) how much more the in-order assignment costs than (c)       number, 1 mark    D - part_c

Tables are drawn until the reduced table has a complete set of zeros (one in each row and each column, which a
brute-force search over all 24 assignments then confirms is optimal), so (c) really is (a) + (b); the in-order
assignment (first worker to Task A, and so on) must cost strictly more than the optimum.
"""

from __future__ import annotations

from itertools import permutations

from ..answers import Distractor
from ..core import check, generator, phrase, pick_distractors
from ._exd_distinct_parts import distinct_part_answers

_N = 4
_TASKS = "ABCD"
_SETTINGS = (
    ("driver", "route", "A courier firm assigns four drivers to four delivery routes."),
    ("cleaner", "office", "A cleaning company assigns four cleaners to four office buildings."),
    ("tutor", "class", "A school assigns four tutors to four year-level classes."),
    ("electrician", "site", "A building firm assigns four electricians to four job sites."),
    ("mechanic", "vehicle", "A workshop assigns four mechanics to four vehicles to service."),
)


def _options(key: int, candidates, n: int = 3):
    out, seen = [], {key}
    for tag, value in candidates:
        if value in seen or value == 0 or (value < 0 and tag != "reversed-subtraction"):
            continue
        seen.add(value)
        out.append(Distractor(str(value), tag, value=value))
    return pick_distractors(str(key), out, n)


@generator("exd_x_r3e_assignment_reduction", level=15, presets={"exam": dict(tier="exam")})
@distinct_part_answers
def exd_x_r3e_assignment_reduction(rng, tier: str = "exam") -> dict:
    check(tier == "exam", f"unknown tier {tier!r}")
    worker, task, intro = rng.choice(_SETTINGS)
    for _ in range(3000):
        base = rng.choice([20, 30, 40])
        cost = [[base + 5 * rng.randint(0, 16) for _ in range(_N)] for _ in range(_N)]
        row_min = [min(r) for r in cost]
        reduced = [[c - m for c in r] for r, m in zip(cost, row_min, strict=True)]
        col_min = [min(reduced[i][j] for i in range(_N)) for j in range(_N)]
        final = [[reduced[i][j] - col_min[j] for j in range(_N)] for i in range(_N)]
        R, C = sum(row_min), sum(col_min)
        if R == 0 or C == 0:
            continue
        if not any(all(final[i][p[i]] == 0 for i in range(_N)) for p in permutations(range(_N))):
            continue
        totals = [sum(cost[i][p[i]] for i in range(_N)) for p in permutations(range(_N))]
        opt = min(totals)
        diag = sum(cost[i][i] for i in range(_N))
        if opt != R + C or diag <= opt:
            continue
        orig_col_min = sum(min(cost[i][j] for i in range(_N)) for j in range(_N))
        # greedy: each worker in turn takes the cheapest task not yet taken
        taken, greedy = set(), 0
        for i in range(_N):
            j = min((j for j in range(_N) if j not in taken), key=lambda j: cost[i][j])
            taken.add(j)
            greedy += cost[i][j]
        a_opts = _options(
            R,
            [
                ("reduced-columns-instead-of-rows", orig_col_min),
                ("left-out-one-row-or-column", R - row_min[-1]),
                ("largest-instead-of-smallest", sum(max(r) for r in cost)),
            ],
        )
        b_opts = _options(
            C,
            [
                ("column-minima-from-unreduced-matrix", orig_col_min),
                ("left-out-one-row-or-column", C - max(col_min)),
                ("largest-instead-of-smallest", sum(max(reduced[i][j] for i in range(_N)) for j in range(_N))),
                ("stopped-after-row-reduction", R),
            ],
        )
        c_opts = _options(
            opt,
            [
                ("stopped-after-row-reduction", R),
                ("column-minima-from-unreduced-matrix", R + orig_col_min),
                ("diagonal-total-as-optimum", diag),
                ("greedy-assignment-total", greedy),
            ],
        )
        d_opts = _options(
            diag - opt,
            [
                ("reversed-subtraction", opt - diag),
                ("diagonal-total-as-optimum", diag),
                ("stopped-after-row-reduction", diag - R),
                ("greedy-assignment-total", diag - greedy),
            ],
        )
        if min(len(a_opts), len(b_opts), len(c_opts), len(d_opts)) >= 3:
            break
    else:
        check(False, "no cost table with a zero assignment and enough distractors")

    check(opt == min(sum(cost[i][p[i]] for i in range(_N)) for p in permutations(range(_N))), "optimum disagrees")
    rows = "; ".join(
        f"{worker.capitalize()} {i + 1}: "
        + ", ".join(f"{task.capitalize()} {_TASKS[j]} ${cost[i][j]}" for j in range(_N))
        for i in range(_N)
    )
    stem = phrase(
        rng,
        (
            "{intro} The cost, in dollars, of each {worker} doing each {task} is: {rows}. "
            "Each {worker} is given exactly one {task} and each {task} is given to exactly one {worker}.",
            "{intro} The table of costs, in dollars, is: {rows}. "
            "Every {worker} does exactly one {task}, and every {task} is done by exactly one {worker}.",
        ),
        intro=intro,
        worker=worker,
        task=task,
        rows=rows,
    )
    return dict(
        text=stem,
        parts=[
            dict(
                prompt=(
                    f"If each {worker} could take whichever {task} costs least for them, with no limit on how many "
                    f"{worker}s choose the same {task}, what would the total cost be, in dollars?"
                ),
                answer_kind="number",
                answer=R,
                answer_text=str(R),
                marks=1,
                working=[f"The cheapest cost in each row is {', '.join(map(str, row_min))}, a total of {R}."],
                distractors=a_opts,
            ),
            dict(
                prompt=(
                    f"Take each {worker}'s cheapest cost off every cost in their row. What is the total of the "
                    f"smallest amounts then left in each {task}'s column, in dollars?"
                ),
                answer_kind="number",
                answer=C,
                answer_text=str(C),
                marks=1,
                working=[f"The column minima of the reduced table are {', '.join(map(str, col_min))}, a total of {C}."],
                distractors=b_opts,
            ),
            dict(
                prompt=(
                    f"Hence find the minimum possible total cost of giving every {worker} a different {task}, "
                    "in dollars."
                ),
                answer_kind="number",
                answer=opt,
                answer_text=str(opt),
                marks=2,
                working=[
                    f"After both reductions a zero can be chosen in every row and column, so the minimum is {R} + {C} = {opt}."
                ],
                distractors=c_opts,
                follow_through="part_a + part_b",
            ),
            dict(
                prompt=(
                    f"Instead, {worker.capitalize()} 1 is given {task.capitalize()} A, {worker.capitalize()} 2 is "
                    f"given {task.capitalize()} B, and so on in order. "
                    "How much more than the minimum does this cost, in dollars?"
                ),
                answer_kind="number",
                answer=diag - opt,
                answer_text=str(diag - opt),
                marks=1,
                working=[
                    f"The in-order total is {diag}, which is {diag} - {opt} = {diag - opt} more than the minimum."
                ],
                distractors=d_opts,
                follow_through=f"{diag} - part_c",
            ),
        ],
    )
