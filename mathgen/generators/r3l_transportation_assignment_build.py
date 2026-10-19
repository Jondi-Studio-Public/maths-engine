"""Generates `uc2-transportation-assignment` questions (easy, medium, hard) and the exam chain
`exd_x_r3l_transportation_assignment`.

``uc2-transportation-assignment`` (``optimisation``, answer ``number``): a balanced transportation problem (supplies,
demands, unit costs) and the assignment problem.

* easy: the total cost of the north-west corner starting solution of a balanced 3x3 or 3x4 transportation problem
  (allocate as much as possible to the top-left cell, move right or down when a row or column is exhausted). The
  instances are non-degenerate, so the start has exactly m + n - 1 allocations.
* medium: the least total cost of assigning four workers to four jobs, one each (the Hungarian method: row and column
  reductions, then zeros). The least-cost assignment is unique.
* hard: from the north-west corner solution, the improvement index c_ij - u_i - v_j of one named empty route (the change
  in total cost for each unit moved on to it), where the potentials u_i + v_j = c_ij on the allocated routes start
  from u_1 = 0.
* exam (``exd_x_r3l_transportation_assignment``): (a) the north-west corner allocation as a matrix, (b) its total cost,
  (c) the improvement index of an empty route with a negative index, (d) the total cost after moving the greatest
  possible number of units round that route's loop (follows through from (b) and (c)).

The assignment optimum is found by enumerating all n! assignments. The improvement index from the potentials is
re-derived by the stepping-stone loop: the signed sum of the unit costs round the unique cycle that the empty route
makes with the allocated routes (the loop also gives the number of units that can be moved).
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations

from ..answers import Distractor, exact_value
from ..core import attempts, check, generator, phrase, pick_distractors
from ._ucv1_linalg import mat_str, matrix_distractors, rows_value

_SOURCES = ("Dubbo", "Griffith", "Wagga Wagga", "Narrabri")
_DESTS = ("Newcastle", "Port Kembla", "Melbourne", "Adelaide")
_WORKERS = ("Mia", "Noah", "Priya", "Liam")
_JOBS = ("the roof", "the wiring", "the plumbing", "the painting")


def _instance(rng, m, n):
    """(supply, demand, cost, allocation) with a non-degenerate north-west corner solution."""
    for _ in attempts(1000, "a non-degenerate transportation instance"):
        supply = [5 * rng.randint(3, 9) for _ in range(m)]
        total = sum(supply)
        cuts = sorted(rng.sample(range(1, total // 5), n - 1))
        demand = [5 * (b - a) for a, b in zip([0] + cuts, cuts + [total // 5])]
        if min(demand) < 10:
            continue
        sums_s = {sum(supply[: i + 1]) for i in range(m - 1)}
        sums_d = {sum(demand[: j + 1]) for j in range(n - 1)}
        if sums_s & sums_d:
            continue
        cost = [[rng.randint(2, 12) for _ in range(n)] for _ in range(m)]
        alloc = _north_west(supply, demand)
        if sum(1 for row in alloc for v in row if v > 0) != m + n - 1:
            continue
        return supply, demand, cost, alloc
    raise AssertionError  # pragma: no cover


def _north_west(supply, demand):
    s, d = list(supply), list(demand)
    alloc = [[0] * len(d) for _ in s]
    i = j = 0
    while i < len(s) and j < len(d):
        q = min(s[i], d[j])
        alloc[i][j] = q
        s[i] -= q
        d[j] -= q
        if s[i] == 0 and i < len(s) - 1 and (d[j] != 0 or j == len(d) - 1):
            i += 1
        elif d[j] == 0 and j < len(d) - 1:
            j += 1
        elif s[i] == 0:
            i += 1
        else:
            j += 1
    return alloc


def _south_east(supply, demand):
    alloc = _north_west(list(reversed(supply)), list(reversed(demand)))
    return [list(reversed(r)) for r in reversed(alloc)]


def _least_cost(supply, demand, cost):
    s, d = list(supply), list(demand)
    alloc = [[0] * len(d) for _ in s]
    cells = sorted(((cost[i][j], i, j) for i in range(len(s)) for j in range(len(d))))
    for _c, i, j in cells:
        q = min(s[i], d[j])
        alloc[i][j] += q
        s[i] -= q
        d[j] -= q
    return alloc


def _swap_two(alloc):
    """The allocation with two allocated quantities in one row exchanged (a slip when working left to right)."""
    out = [list(r) for r in alloc]
    for r in out:
        pos = [j for j, v in enumerate(r) if v > 0]
        if len(pos) >= 2 and r[pos[0]] != r[pos[1]]:
            r[pos[0]], r[pos[1]] = r[pos[1]], r[pos[0]]
            return out
    return None


def _total(alloc, cost):
    return sum(alloc[i][j] * cost[i][j] for i in range(len(alloc)) for j in range(len(alloc[0])))


def _potentials(alloc, cost):
    """u (rows) and v (columns) with u_1 = 0 and u_i + v_j = c_ij on every allocated route."""
    m, n = len(alloc), len(alloc[0])
    u, v = [None] * m, [None] * n
    u[0] = Fraction(0)
    for _ in range(m + n):
        for i in range(m):
            for j in range(n):
                if alloc[i][j] > 0:
                    if u[i] is not None and v[j] is None:
                        v[j] = cost[i][j] - u[i]
                    elif v[j] is not None and u[i] is None:
                        u[i] = cost[i][j] - v[j]
    check(None not in u and None not in v, "potentials not determined")
    return u, v


def _loop(alloc, i0, j0):
    """The stepping-stone loop of the empty route (i0, j0): a list of ((i, j), sign) with the new route first."""
    m, n = len(alloc), len(alloc[0])
    basic = [(i, j) for i in range(m) for j in range(n) if alloc[i][j] > 0]
    # breadth-first path through the tree of allocated routes from row i0 to column j0
    start, goal = ("r", i0), ("c", j0)
    prev = {start: None}
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node == goal:
            break
        for i, j in basic:
            if node == ("r", i) and ("c", j) not in prev:
                prev[("c", j)] = (node, (i, j))
                queue.append(("c", j))
            elif node == ("c", j) and ("r", i) not in prev:
                prev[("r", i)] = (node, (i, j))
                queue.append(("r", i))
    check(goal in prev, "no loop")
    path, node = [], goal
    while prev[node] is not None:
        node, cell = prev[node]
        path.append(cell)
    path.reverse()  # from the row end to the column end
    signs = [(-1 if k % 2 == 0 else 1) for k in range(len(path))]
    return [((i0, j0), 1)] + list(zip(path, signs))


def _stepping_stone(alloc, cost, i0, j0):
    loop = _loop(alloc, i0, j0)
    index = sum(sign * cost[i][j] for (i, j), sign in loop)
    theta = min(alloc[i][j] for (i, j), sign in loop if sign < 0)
    return index, theta


def _modi_index(alloc, cost, i0, j0):
    u, v = _potentials(alloc, cost)
    return cost[i0][j0] - u[i0] - v[j0]


def _text_problem(supply, demand, cost, sources, dests, rng=None):
    opening = (
        phrase(
            rng,
            [
                "A freight company moves grain from {m} depots to {n} ports.",
                "A grain handler must deliver grain from {m} depots to {n} ports.",
                "Grain stored at {m} depots is to be sent to {n} ports.",
            ],
            m=len(supply),
            n=len(demand),
        )
        if rng is not None
        else f"A freight company moves grain from {len(supply)} depots to {len(demand)} ports."
    )
    return (
        f"{opening} "
        f"Supplies (tonnes): {', '.join(f'{s} {q}' for s, q in zip(sources, supply))}. "
        f"Demands (tonnes): {', '.join(f'{d} {q}' for d, q in zip(dests, demand))}. "
        f"The cost of moving one tonne ($), with the depots as rows in the order above and the ports as columns in the order above, is {mat_str(cost)}."
    )


def _easy(rng):
    m, n = rng.choice([(3, 3), (3, 4)])
    supply, demand, cost, alloc = _instance(rng, m, n)
    sources, dests = _SOURCES[:m], _DESTS[:n]
    total = _total(alloc, cost)
    check(sum(map(sum, alloc)) == sum(supply) and all(sum(r) == s for r, s in zip(alloc, supply)), "supply not met")
    check(all(sum(alloc[i][j] for i in range(m)) == demand[j] for j in range(n)), "demand not met")
    least = _total(_least_cost(supply, demand, cost), cost)
    se = _total(_south_east(supply, demand), cost)
    units = sum(cost[i][j] for i in range(m) for j in range(n) if alloc[i][j] > 0)
    cands = [
        Distractor(str(least), "r3l-transport-wrong-start-rule", least),
        Distractor(str(units), "r3l-transport-cost-without-quantity", units),
        Distractor(str(se), "r3l-transport-wrong-start-corner", se),
        Distractor(
            str(sum(supply) * min(min(r) for r in cost)),
            "r3l-transport-cheapest-everywhere",
            sum(supply) * min(min(r) for r in cost),
        ),
    ]
    last = max((i, j) for i in range(m) for j in range(n) if alloc[i][j] > 0)
    cands.append(
        Distractor(
            str(total - alloc[last[0]][last[1]] * cost[last[0]][last[1]]),
            "r3l-transport-cell-omitted",
            total - alloc[last[0]][last[1]] * cost[last[0]][last[1]],
        )
    )
    swapped = _swap_two(alloc)
    if swapped is not None:
        w = _total(swapped, cost)
        cands.append(Distractor(str(w), "r3l-transport-quantity-misplaced", w))
    distractors = pick_distractors(str(total), cands, 4)
    check(len(distractors) >= 3, "too few distractors")
    return dict(
        text=f"{_text_problem(supply, demand, cost, sources, dests, rng)} Find the total cost of the north-west corner starting solution, in dollars.",
        answer_kind="number",
        answer=total,
        answer_text=str(total),
        answer_latex=str(total),
        working=[
            f"Allocate from the top-left cell, moving right when a port is satisfied and down when a depot is empty: {mat_str(alloc)}.",
            "Total = "
            + " + ".join(f"{alloc[i][j]}×{cost[i][j]}" for i in range(m) for j in range(n) if alloc[i][j] > 0)
            + f" = {total}.",
        ],
        distractors=distractors,
    )


def _hungarian_reduce(cost):
    rows = [[v - min(r) for v in r] for r in cost]
    cols_min = [min(rows[i][j] for i in range(len(rows))) for j in range(len(rows))]
    return [[rows[i][j] - cols_min[j] for j in range(len(rows))] for i in range(len(rows))], sum(
        min(r) for r in cost
    ) + sum(cols_min)


def _assignment_instance(rng, n=4):
    for _ in attempts(2000, "an assignment instance with a unique best"):
        cost = [[rng.randint(3, 20) for _ in range(n)] for _ in range(n)]
        totals = sorted((sum(cost[i][p[i]] for i in range(n)), p) for p in permutations(range(n)))
        if totals[0][0] == totals[1][0]:
            continue
        greedy = _greedy(cost)
        if greedy == totals[0][0]:
            continue
        return cost, totals[0][0], totals[0][1]
    raise AssertionError  # pragma: no cover


def _greedy(cost):
    """Row by row, take the cheapest job still free."""
    n = len(cost)
    used, total = set(), 0
    for i in range(n):
        j = min((j for j in range(n) if j not in used), key=lambda j: cost[i][j])
        used.add(j)
        total += cost[i][j]
    return total


def _medium(rng):
    cost, best, perm = _assignment_instance(rng)
    n = 4
    reduced, bound = _hungarian_reduce(cost)
    # the zeros of the reduced matrix include the optimal assignment when the bound equals the optimum
    greedy = _greedy(cost)
    row_min = sum(min(r) for r in cost)
    diagonal = sum(cost[i][i] for i in range(n))
    worst = max(sum(cost[i][p[i]] for i in range(n)) for p in permutations(range(n)))
    second = sorted(sum(cost[i][p[i]] for i in range(n)) for p in permutations(range(n)))[1]
    col_greedy = _greedy([list(c) for c in zip(*cost)])
    cands = [
        Distractor(str(greedy), "r3l-assign-greedy-row-minimum", greedy),
        Distractor(str(col_greedy), "r3l-assign-greedy-column-minimum", col_greedy),
        Distractor(str(second), "r3l-assign-near-optimal", second),
        Distractor(str(bound), "r3l-assign-reduction-total", bound),
        Distractor(str(row_min), "r3l-assign-row-minima-only", row_min),
        Distractor(str(diagonal), "r3l-assign-diagonal", diagonal),
        Distractor(str(worst), "r3l-assign-maximised", worst),
    ]
    distractors = pick_distractors(str(best), cands, 4)
    check(len(distractors) >= 3, "too few distractors")
    text = phrase(
        rng,
        [
            "Four tradies, {workers} and {last}, are each given one of four jobs, {j0}, {j1}, {j2} and {j3}. The cost "
            "of each tradie doing each job (in hundreds of dollars), with the tradies as rows and the jobs as columns "
            "in the orders given, is {cost}. Each job goes to a different tradie. Find the least total cost, in "
            "hundreds of dollars.",
            "A builder has four tradies, {workers} and {last}, and four jobs to hand out, {j0}, {j1}, {j2} and {j3}. "
            "Each tradie takes exactly one job and each job is done by one tradie. The costs (in hundreds of dollars), "
            "with the tradies as rows and the jobs as columns in the orders given, are {cost}. What is the least "
            "total cost, in hundreds of dollars?",
            "Four jobs, {j0}, {j1}, {j2} and {j3}, are to be shared between four tradies, {workers} and {last}, one "
            "job each. The cost table (in hundreds of dollars) has the tradies as rows and the jobs as columns, in "
            "the orders given: {cost}. Find the smallest possible total cost, in hundreds of dollars.",
        ],
        workers=", ".join(_WORKERS[:3]),
        last=_WORKERS[3],
        j0=_JOBS[0],
        j1=_JOBS[1],
        j2=_JOBS[2],
        j3=_JOBS[3],
        cost=mat_str(cost),
    )
    return dict(
        text=text,
        answer_kind="number",
        answer=best,
        answer_text=str(best),
        answer_latex=str(best),
        working=[
            f"Subtract each row's minimum, then each column's minimum: {mat_str(reduced)}.",
            f"The zeros give the assignment {', '.join(f'{_WORKERS[i]} to {_JOBS[perm[i]]}' for i in range(n))}, costing {' + '.join(str(cost[i][perm[i]]) for i in range(n))} = {best}.",
        ],
        distractors=distractors,
    )


def _pick_cell(rng, alloc):
    empties = [(i, j) for i in range(len(alloc)) for j in range(len(alloc[0])) if alloc[i][j] == 0]
    return rng.choice(empties)


def _hard(rng):
    m, n = rng.choice([(3, 3), (3, 4), (3, 3)])
    for _ in attempts(200, "an empty route with a non-trivial index"):
        supply, demand, cost, alloc = _instance(rng, m, n)
        i0, j0 = _pick_cell(rng, alloc)
        index = _modi_index(alloc, cost, i0, j0)
        if index not in (0, cost[i0][j0]) and abs(index) >= 2:
            break
    sources, dests = _SOURCES[:m], _DESTS[:n]
    loop_index, _theta = _stepping_stone(alloc, cost, i0, j0)
    check(index == loop_index, "potentials and stepping-stone loop disagree")
    u, v = _potentials(alloc, cost)
    index = int(index)
    cands = [
        Distractor(str(-index), "r3l-modi-index-sign", -index),
        Distractor(str(int(u[i0] + v[j0])), "r3l-modi-index-potentials-only", int(u[i0] + v[j0])),
        Distractor(str(cost[i0][j0]), "r3l-modi-cost-as-index", cost[i0][j0]),
    ]
    # potentials built with the wrong sign: v_j = c_ij + u_i
    vv = [None] * n
    uu = [Fraction(0)] + [None] * (m - 1)
    for _ in range(m + n):
        for i in range(m):
            for j in range(n):
                if alloc[i][j] > 0:
                    if uu[i] is not None and vv[j] is None:
                        vv[j] = cost[i][j] + uu[i]
                    elif vv[j] is not None and uu[i] is None:
                        uu[i] = vv[j] - cost[i][j]
    if None not in uu and None not in vv:
        wrong = int(cost[i0][j0] - uu[i0] - vv[j0])
        cands.append(Distractor(str(wrong), "r3l-modi-potential-sign", wrong))
    cands.append(
        Distractor(str(int(cost[i0][j0] - u[i0] + v[j0])), "r3l-modi-potential-sign", int(cost[i0][j0] - u[i0] + v[j0]))
    )
    cands.append(Distractor(str(int(cost[i0][j0] - u[i0])), "r3l-modi-potential-omitted", int(cost[i0][j0] - u[i0])))
    cands.append(Distractor(str(int(cost[i0][j0] - v[j0])), "r3l-modi-potential-omitted", int(cost[i0][j0] - v[j0])))
    distractors = pick_distractors(str(index), cands, 4)
    check(len(distractors) >= 3, "too few distractors")
    return dict(
        text=(
            f"{_text_problem(supply, demand, cost, sources, dests, rng)} Starting from the north-west corner solution, find the "
            f"improvement index of the route from {sources[i0]} to {dests[j0]}: the change in total cost for each tonne moved "
            "onto that route. Take the potential of the first depot to be 0."
        ),
        answer_kind="number",
        answer=index,
        answer_text=str(index),
        answer_latex=str(index),
        working=[
            f"North-west corner allocation: {mat_str(alloc)}.",
            f"With u₁ = 0 and u_i + v_j = c_ij on allocated routes, u = {[int(x) for x in u]}, v = {[int(x) for x in v]}.",
            f"Index = c − u − v = {cost[i0][j0]} − ({int(u[i0])}) − ({int(v[j0])}) = {index}.",
        ],
        distractors=distractors,
    )


@generator(
    "r3l_transportation_assignment",
    level=20,
    presets={"easy": dict(tier="easy"), "medium": dict(tier="medium"), "hard": dict(tier="hard")},
)
def r3l_transportation_assignment(rng, tier="easy"):
    """Transportation start cost, the assignment optimum, and an improvement index; see the module docstring."""
    if tier == "easy":
        return _easy(rng)
    if tier == "medium":
        return _medium(rng)
    check(tier == "hard", f"unknown tier {tier!r}")
    return _hard(rng)


# -- exam -------------------------------------------------------------------------------------------------------------


@generator("exd_x_r3l_transportation_assignment", level=20, presets={"exam": dict(tier="exam")})
def exd_x_r3l_transportation_assignment(rng, tier="exam"):
    """Exam chain: NW corner allocation, its cost, an improvement index, the cost after the exchange."""
    check(tier == "exam", f"unknown tier {tier!r}")
    for _ in attempts(300, "an improving route"):
        m, n = rng.choice([(3, 3), (3, 4)])
        supply, demand, cost, alloc = _instance(rng, m, n)
        cells = [(i, j) for i in range(m) for j in range(n) if alloc[i][j] == 0]
        improving = [
            (i, j)
            for (i, j) in cells
            if _modi_index(alloc, cost, i, j) < -1 and _modi_index(alloc, cost, i, j) != -cost[i][j]
        ]
        if improving:
            break
    i0, j0 = rng.choice(improving)
    sources, dests = _SOURCES[:m], _DESTS[:n]
    total = _total(alloc, cost)
    index = int(_modi_index(alloc, cost, i0, j0))
    loop_index, theta = _stepping_stone(alloc, cost, i0, j0)
    check(loop_index == index and index < 0, "index differs from the loop")
    new_total = total + index * theta
    # independent: apply the shift and re-total
    shifted = [list(r) for r in alloc]
    for (i, j), sign in _loop(alloc, i0, j0):
        shifted[i][j] += sign * theta
    check(min(min(r) for r in shifted) >= 0 and _total(shifted, cost) == new_total, "shifted allocation total differs")
    a_cands = [
        (_south_east(supply, demand), "r3l-transport-wrong-start-corner"),
        (_least_cost(supply, demand, cost), "r3l-transport-wrong-start-rule"),
        ([list(r) for r in zip(*alloc)], "r3l-transport-allocation-transposed"),
        ([[min(supply[i], demand[j]) for j in range(n)] for i in range(m)], "r3l-transport-quantity-not-reduced"),
        (_swap_two(alloc), "r3l-transport-quantity-misplaced"),
    ]
    a_dist = matrix_distractors(
        [(([[Fraction(v) for v in r] for r in mat]) if mat else None, tag) for mat, tag in a_cands],
        rows_value([[Fraction(v) for v in r] for r in alloc]),
    )[:3]
    least = _total(_least_cost(supply, demand, cost), cost)
    se = _total(_south_east(supply, demand), cost)
    units = sum(cost[i][j] for i in range(m) for j in range(n) if alloc[i][j] > 0)
    b_dist = pick_distractors(
        str(total),
        [
            Distractor(str(least), "r3l-transport-wrong-start-rule", least),
            Distractor(str(units), "r3l-transport-cost-without-quantity", units),
            Distractor(str(se), "r3l-transport-wrong-start-corner", se),
            Distractor(
                str(total - alloc[0][0] * cost[0][0]), "r3l-transport-cell-omitted", total - alloc[0][0] * cost[0][0]
            ),
        ],
        3,
    )
    c_dist = pick_distractors(
        str(index),
        [
            Distractor(str(-index), "r3l-modi-index-sign", -index),
            Distractor(str(cost[i0][j0]), "r3l-modi-cost-as-index", cost[i0][j0]),
            Distractor(str(index * theta), "r3l-modi-index-times-quantity", index * theta),
            Distractor(
                str(int(cost[i0][j0] - _potentials(alloc, cost)[0][i0])),
                "r3l-modi-potential-omitted",
                int(cost[i0][j0] - _potentials(alloc, cost)[0][i0]),
            ),
            Distractor(
                str(int(cost[i0][j0] - _potentials(alloc, cost)[1][j0])),
                "r3l-modi-potential-omitted",
                int(cost[i0][j0] - _potentials(alloc, cost)[1][j0]),
            ),
        ],
        3,
    )
    d_dist = pick_distractors(
        str(new_total),
        [
            Distractor(str(total - index * theta), "r3l-modi-index-sign", total - index * theta),
            Distractor(str(total + index), "r3l-modi-quantity-ignored", total + index),
            Distractor(str(index * theta), "r3l-modi-index-times-quantity", index * theta),
            Distractor(str(total), "r3l-modi-quantity-ignored", total),
        ],
        3,
    )
    check(
        all(len(x) >= 2 for x in (a_dist, b_dist, c_dist, d_dist)),
        f"too few distractors {[len(x) for x in (a_dist, b_dist, c_dist, d_dist)]}",
    )
    return dict(
        text=_text_problem(supply, demand, cost, sources, dests, rng),
        parts=[
            dict(
                prompt="Use the north-west corner rule to write down a starting allocation, with the depots as rows and the ports as columns.",
                answer_kind="matrix",
                answer=rows_value([[Fraction(v) for v in r] for r in alloc]),
                answer_text=mat_str([[Fraction(v) for v in r] for r in alloc]),
                answer_latex="plain:" + mat_str([[Fraction(v) for v in r] for r in alloc]),
                marks=3,
                working=[f"Allocating from the top-left cell gives {mat_str(alloc)}."],
                distractors=a_dist,
            ),
            dict(
                prompt="Find the total cost of this allocation, in dollars.",
                answer_kind="number",
                answer=total,
                answer_text=str(total),
                marks=2,
                sensible_range=(0, None),
                working=[
                    "Total = "
                    + " + ".join(f"{alloc[i][j]}×{cost[i][j]}" for i in range(m) for j in range(n) if alloc[i][j] > 0)
                    + f" = {total}."
                ],
                distractors=b_dist,
            ),
            dict(
                prompt=f"Find the improvement index of the route from {sources[i0]} to {dests[j0]}: the change in total cost for each tonne moved onto it. Take the potential of the first depot to be 0.",
                answer_kind="number",
                answer=index,
                answer_text=str(index),
                marks=3,
                working=[
                    f"The loop through the allocated routes gives index {index}, which is negative, so the route is worth using."
                ],
                distractors=c_dist,
            ),
            dict(
                prompt=f"Move as many tonnes as possible onto the route from {sources[i0]} to {dests[j0]}, adjusting the allocated routes round its loop. Find the new total cost.",
                answer_kind="number",
                answer=new_total,
                answer_text=str(new_total),
                marks=2,
                follow_through=f"{total} + {theta}*part_c",
                sensible_range=(0, None),
                working=[
                    f"The loop lets {theta} tonnes move, so the cost changes by {index}×{theta} = {index * theta}, to {new_total}."
                ],
                distractors=d_dist,
            ),
        ],
    )
