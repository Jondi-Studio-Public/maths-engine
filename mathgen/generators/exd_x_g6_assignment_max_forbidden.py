"""Exam-tier assignment question: maximise profit with one forbidden pairing.

Earlier tiers minimise the total cost of a square cost table.  Here the table is five by five, the objective is
to MAXIMISE total profit (so the learner has to choose to maximise, or to convert to a minimisation by
subtracting from the largest entry), and one worker is not allowed to do one task.  The restriction is drawn so
that it matters: every unrestricted optimum uses the forbidden pairing, so ignoring it gives a wrong total.

The key is found by exhaustive search over the 120 permutations and confirmed by a bitmask dynamic programme.
"""

from __future__ import annotations

import random
from itertools import permutations

from mathgen.core import check, check_equivalent, generator, latex_answer, pick_distractors

_N = 5
_TASKS = "ABCDE"


def _best_exhaustive(m, banned, maximise=True):
    best = None
    for perm in permutations(range(_N)):
        if banned is not None and perm[banned[0]] == banned[1]:
            continue
        total = sum(m[i][perm[i]] for i in range(_N))
        if best is None or (total > best if maximise else total < best):
            best = total
    return best


def _best_dp(m, banned):
    """Independent route: dynamic programme over the set of tasks already taken."""
    neg = float("-inf")
    best = {0: 0}
    for i in range(_N):
        nxt = {}
        for mask, val in best.items():
            for j in range(_N):
                if mask >> j & 1 or banned == (i, j):
                    continue
                key = mask | 1 << j
                v = val + m[i][j]
                if v > nxt.get(key, neg):
                    nxt[key] = v
        best = nxt
    return best[(1 << _N) - 1]


@generator("exd_x_g6_assignment_max_forbidden", level=12, presets={"exam": dict(tier="exam")})
def assignment_max_forbidden(rng: random.Random, tier: str = "exam") -> dict:
    """Maximum total profit for five workers and five tasks when one pairing is not allowed."""
    for _ in range(2000):
        m = [[rng.randint(8, 40) for _ in range(_N)] for _ in range(_N)]
        free = _best_exhaustive(m, None)
        banned = (rng.randrange(_N), rng.randrange(_N))
        restricted = _best_exhaustive(m, banned)
        if restricted >= free:
            continue  # the restriction must change the answer
        cost_view = _best_exhaustive(m, banned, maximise=False)
        row_best = sum(max(row) for row in m)
        diag = sum(m[i][i] for i in range(_N))
        cands = [free, cost_view, row_best, diag]
        if len({restricted, *cands}) < 4:
            continue
        break
    else:
        raise RuntimeError("exd_x_g6_assignment_max_forbidden: no valid draw")
    check_equivalent(_best_dp(m, banned), restricted, "dynamic programme disagrees with exhaustive search")
    rows = "; ".join(
        f"Worker {i + 1}: " + ", ".join(f"Task {_TASKS[j]} ${m[i][j]}" for j in range(_N)) for i in range(_N)
    )
    text = (
        f"A firm earns the following profit when each worker does each task. {rows}. "
        f"Each worker does exactly one task and each task is done by exactly one worker, but "
        f"Worker {banned[0] + 1} cannot be given Task {_TASKS[banned[1]]}. "
        f"What is the maximum possible total profit, in dollars?"
    )
    answer_text = str(restricted)
    distractors = pick_distractors(answer_text, [str(c) for c in cands])
    check(len(distractors) >= 2, "need at least two distractors")
    return dict(
        text=text,
        latex=rf"\text{{{text}}}",
        answer=restricted,
        answer_text=answer_text,
        answer_latex=latex_answer(restricted),
        working=[
            f"Best total with no restriction is {free}; every such assignment uses the banned pairing, so the best allowed total is {restricted}."
        ],
        distractors=distractors,
    )
