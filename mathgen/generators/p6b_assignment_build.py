"""Optimum assignment by inspection, senior ``acara-senior-v8-4``, domain ``financial-discrete``, ``answer:
number``.

One generator:

- ``determine_the_optimum_assignment_s_by_inspection_for`` (``senior-determine-the-optimum-assignment-s-by-inspection-for``, ACMGM111) -- Given a worker-task cost matrix, find the minimum total cost to assign every worker to exactly one task. Easy tier: 3x3 matrix (exhaustive inspection feasible); medium tier: 4x4 matrix with at least one greedy shortcut coincidentally succeeding as distractor; hard tier: 4x4 matrix where both greedy shortcuts fail, forcing genuine exhaustive reasoning.
"""

from __future__ import annotations

from itertools import permutations

from ..core import check, check_distinct, check_equivalent, generator, pick_distractors


def _min_assignment_cost(matrix):
    """Exhaustive permutation search for minimum assignment cost.

    For n workers and n tasks, there are n! permutations. For n=3 (9!~360k),
    n=4 (24), this is feasible to enumerate by inspection for small-scale problems.
    """
    n = len(matrix)
    best = None
    for perm in permutations(range(n)):
        if any(matrix[i][perm[i]] is None for i in range(n)):
            continue  # a blocked (not possible) cell: infeasible assignment
        total = sum(matrix[i][perm[i]] for i in range(n))
        if best is None or total < best:
            best = total
    return best


def _min_assignment_backtracking(matrix, workers_left, tasks_used):
    """Recursive backtracking: independent check route, genuinely different
    algorithm from permutation enumeration above."""
    if not workers_left:
        return 0
    w = workers_left[0]
    rest = workers_left[1:]
    n = len(matrix)
    best = None
    for t in range(n):
        if t in tasks_used or matrix[w][t] is None:
            continue
        sub = _min_assignment_backtracking(matrix, rest, tasks_used | {t})
        if sub is None:
            continue
        cost = matrix[w][t] + sub
        if best is None or cost < best:
            best = cost
    return best


def _verify_min_assignment(matrix, answer):
    """Independent check: verify answer against two different algorithms."""
    n = len(matrix)
    permutation_result = _min_assignment_cost(matrix)
    backtracking_result = _min_assignment_backtracking(matrix, list(range(n)), frozenset())
    check_equivalent(
        backtracking_result,
        permutation_result,
        "exhaustive permutation search and recursive backtracking must agree on the minimum assignment cost",
    )
    check(answer == permutation_result, f"answer {answer} must equal the minimum assignment cost {permutation_result}")


def _format_matrix_text(matrix, task_names):
    """Format a cost matrix as the required text format.

    One line per worker, tasks in fixed order (Task A, Task B, ...), costs in dollars.
    Separate workers with semicolons to avoid newlines (lint constraint).
    """
    lines = []
    for i, row in enumerate(matrix):
        worker_num = i + 1
        costs = [
            f"Task {task_names[j]} is not possible" if row[j] is None else f"Task {task_names[j]} costs ${row[j]}"
            for j in range(len(row))
        ]
        lines.append(f"Worker {worker_num}: {', '.join(costs)}")
    return "; ".join(lines) + "."


def _row_greedy(matrix):
    """Distractor 1: each worker independently picks their cheapest task."""
    return sum(min(row) for row in matrix)


def _col_greedy(matrix):
    """Distractor 2: each task picks its cheapest worker."""
    n = len(matrix)
    return sum(min(matrix[i][j] for i in range(n)) for j in range(n))


def _diagonal(matrix):
    """Distractor 3: worker i does task i (default unchecked assignment)."""
    return sum(matrix[i][i] for i in range(len(matrix)))


def _anti_diagonal(matrix):
    """Fallback distractor: worker i does task (n-1-i) -- a second, equally
    naive positional default (reverse listing order), used only when the
    three named misconceptions above collide too often with each other or
    the answer to leave >=2 distractors (observed on small 3x3 easy-tier
    matrices). Still a genuinely different, unchecked assignment a learner
    might pick without comparing costs."""
    n = len(matrix)
    return sum(matrix[i][n - 1 - i] for i in range(n))


def _draw_hard_matrix(rng, n, lo, hi, max_attempts=200):
    """Draw a hard-tier matrix where both greedy heuristics fail.

    Both row_greedy and col_greedy must be strictly less than optimum
    (neither accidentally gives the right answer).

    Returns (matrix, attempts_used). Raises ValueError if max_attempts exceeded.
    """
    for attempt in range(max_attempts):
        matrix = [[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)]
        optimum = _min_assignment_cost(matrix)
        row_greedy_val = _row_greedy(matrix)
        col_greedy_val = _col_greedy(matrix)

        # Both greedy approaches must be strictly less than optimum
        if row_greedy_val < optimum and col_greedy_val < optimum:
            return matrix, attempt

    # If we exhaust all attempts, raise but still return the last matrix
    # for debugging purposes
    raise ValueError(
        f"Failed to draw hard matrix after {max_attempts} attempts (both row-greedy < optimum AND col-greedy < optimum)"
    )


def _allowed_row_greedy(matrix):
    """Blocked-cell variant: each worker picks their cheapest ALLOWED task."""
    return sum(min(c for c in row if c is not None) for row in matrix)


def _allowed_col_greedy(matrix):
    """Blocked-cell variant: each task picks its cheapest ALLOWED worker."""
    n = len(matrix)
    return sum(min(matrix[i][j] for i in range(n) if matrix[i][j] is not None) for j in range(n))


def _allowed_positional(matrix):
    """Blocked-cell variant of the diagonal default. Rule: the diagonal if all
    its cells are allowed; else the anti-diagonal if all of those are allowed;
    else None (the distractor is dropped)."""
    n = len(matrix)
    for cells in ([matrix[i][i] for i in range(n)], [matrix[i][n - 1 - i] for i in range(n)]):
        if all(c is not None for c in cells):
            return sum(cells)
    return None


def _draw_blocked_matrix(rng, n, lo, hi, blocked, max_attempts=2000):
    """Draw an n x n matrix with ``blocked`` cells marked None (not possible).

    Accepted only when (a) every assignment achieving the unconstrained optimum
    (blocked cells' hidden costs included) uses a blocked cell, (b) a feasible
    assignment exists and its optimum is strictly greater, and (c) the
    constrained optimum differs from every distractor value, with at least
    three distinct distractors left. Returns (matrix, unconstrained_optimum,
    distractor_values).
    """
    for _ in range(max_attempts):
        full = [[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)]
        cells = rng.sample(range(n * n), blocked)
        matrix = [[None if (i * n + j) in cells else full[i][j] for j in range(n)] for i in range(n)]
        constrained = _min_assignment_cost(matrix)
        if constrained is None:
            continue  # no feasible perfect matching (also covers an all-blocked row or column)
        unconstrained = _min_assignment_cost(full)
        if constrained <= unconstrained:
            continue  # (b): constrained optimum strictly greater
        # (a): every unconstrained-optimal assignment uses a blocked cell
        # (implied by (b), but checked directly).
        if any(
            sum(full[i][p[i]] for i in range(n)) == unconstrained and all(matrix[i][p[i]] is not None for i in range(n))
            for p in permutations(range(n))
        ):
            continue
        values = [unconstrained, _allowed_row_greedy(matrix), _allowed_col_greedy(matrix), _allowed_positional(matrix)]
        values = [v for v in values if v is not None]
        if any(v == constrained for v in values) or len(set(values)) < 3:
            continue  # (c)
        return matrix, unconstrained, values
    raise ValueError(f"Failed to draw blocked matrix after {max_attempts} attempts")


@generator(
    "determine_the_optimum_assignment_s_by_inspection_for",
    level=11,
    presets={
        "easy": dict(n=3, lo=4, hi=20),
        "medium": dict(n=4, lo=4, hi=25),
        "hard": dict(n=4, lo=4, hi=30),
    },
)
def determine_the_optimum_assignment_s_by_inspection_for(rng, n=3, lo=4, hi=20, blocked=0):
    """Given a worker-task cost matrix, determine the minimum total cost
    to assign every worker to exactly one different task."""

    if blocked:
        return _blocked_question(rng, n, lo, hi, blocked)

    # Draw the matrix
    if n == 4 and lo == 4 and hi == 30:
        # Hard tier: ensure both greedy heuristics fail
        matrix, _ = _draw_hard_matrix(rng, n, lo, hi)
    else:
        # Easy and medium tiers: just draw randomly
        matrix = [[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)]

    # Task names (A, B, C, D)
    task_names = [chr(ord("A") + i) for i in range(n)]

    # Compute the correct answer
    answer = _min_assignment_cost(matrix)
    _verify_min_assignment(matrix, answer)

    # Format text (no newlines due to lint constraint newline_in_text)
    matrix_text = _format_matrix_text(matrix, task_names)
    text = (
        f"{matrix_text} "
        f"Each worker is assigned to exactly one task, and each task is assigned to exactly one worker. "
        f"What is the minimum possible total cost, in dollars, to assign every worker to a different task?"
    )

    answer_text = str(answer)

    # Compute distractors
    row_greedy_val = _row_greedy(matrix)
    col_greedy_val = _col_greedy(matrix)
    diagonal_val = _diagonal(matrix)
    anti_diagonal_val = _anti_diagonal(matrix)

    # Collect candidates (convert to strings for pick_distractors). The
    # anti-diagonal fallback is listed last so the three named misconceptions
    # are preferred whenever they survive pick_distractors' collision-drop;
    # it only shows up when one of the first three collided with the answer
    # or with each other.
    candidates = [str(row_greedy_val), str(col_greedy_val), str(diagonal_val), str(anti_diagonal_val)]

    # Use pick_distractors to handle collisions with answer and duplication
    distractors = pick_distractors(answer_text, candidates)

    # Verify distractors are distinct from answer
    for d in distractors:
        try:
            d_val = int(d)
            check_distinct(answer, d_val, f"distractor {d} must differ from answer {answer}")
        except (ValueError, TypeError):
            pass  # Non-numeric distractor, skip check

    return dict(
        text=text,
        latex=rf"\text{{Cost matrix: }}{n}\times{n}\text{{, costs \${lo}--\${hi}; find min-cost assignment}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
    )


def _blocked_question(rng, n, lo, hi, blocked):
    """Hard-tier decision: some worker-task pairs are not possible, and the
    cheapest unconstrained assignment needs one, so the learner must exclude
    pairs and search the feasible assignments."""
    matrix, _unconstrained, values = _draw_blocked_matrix(rng, n, lo, hi, blocked)
    task_names = [chr(ord("A") + i) for i in range(n)]
    answer = _min_assignment_cost(matrix)
    _verify_min_assignment(matrix, answer)
    text = (
        f"{_format_matrix_text(matrix, task_names)} "
        f"A worker cannot be assigned a task marked not possible. "
        f"Each worker is assigned to exactly one task, and each task is assigned to exactly one worker. "
        f"What is the minimum possible total cost, in dollars, to assign every worker to a different task?"
    )
    answer_text = str(answer)
    check_distinct(answer, *values, msg="every blocked-cell distractor must differ from the answer")
    distractors = pick_distractors(answer_text, [str(v) for v in values])
    check(len(distractors) >= 3, "blocked-cell question needs three distractors")
    return dict(
        text=text,
        latex=rf"\text{{Cost matrix: }}{n}\times{n}\text{{ with {blocked} blocked cells; find min-cost feasible assignment}}",
        answer=answer,
        answer_text=answer_text,
        answer_latex=answer_text,
        distractors=distractors,
    )
