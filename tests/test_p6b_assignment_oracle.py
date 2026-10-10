"""
Independent oracle tests for the optimum-assignment skill.
Derives all answers from question text alone without reading the generator.
"""

import re
from itertools import permutations
import pytest
from mathgen import make


def parse_assignment_question(question_text):
    """
    Parse worker-to-task cost matrix from question text.
    Returns: dict of worker_id -> dict of task_letter -> cost
    Also returns: ordered list of task letters
    """
    # Pattern: "Worker N: Task X costs $C, Task Y costs $D, ..."
    worker_pattern = r"Worker (\d+):\s*(.+?)(?:;|Each worker)"
    task_pattern = r"Task ([A-Z])\s*costs\s*\$(\d+)"

    workers_data = {}
    task_letters_set = set()

    for worker_match in re.finditer(worker_pattern, question_text):
        worker_id = int(worker_match.group(1))
        worker_text = worker_match.group(2)

        tasks = {}
        for task_match in re.finditer(task_pattern, worker_text):
            task_letter = task_match.group(1)
            cost = int(task_match.group(2))
            tasks[task_letter] = cost
            task_letters_set.add(task_letter)

        workers_data[worker_id] = tasks

    # Verify all workers and tasks present
    task_letters = sorted(task_letters_set)

    # Validate: number of workers should equal number of tasks
    num_workers = len(workers_data)
    num_tasks = len(task_letters)

    if num_workers != num_tasks:
        raise ValueError(f"Mismatch: {num_workers} workers, {num_tasks} tasks")

    # Validate: all workers have all tasks
    for worker_id in sorted(workers_data.keys()):
        for task_letter in task_letters:
            if task_letter not in workers_data[worker_id]:
                raise ValueError(f"Worker {worker_id} missing cost for Task {task_letter}")

    return workers_data, task_letters


def find_min_cost_assignment(workers_data, task_letters):
    """
    Find minimum-cost one-to-one assignment using exhaustive search.
    Returns: minimum total cost
    """
    worker_ids = sorted(workers_data.keys())
    min_cost = float("inf")

    # Try all permutations of task assignments
    for perm in permutations(task_letters):
        # perm[i] is the task assigned to worker_ids[i]
        total_cost = sum(workers_data[worker_ids[i]][perm[i]] for i in range(len(worker_ids)))
        min_cost = min(min_cost, total_cost)

    return min_cost


def compute_row_greedy(workers_data, task_letters):
    """
    Compute row-greedy total: each worker's cheapest task, summed.
    Ignores conflicts (multiple workers assigned to same task).
    """
    total = 0
    for worker_id in sorted(workers_data.keys()):
        worker_costs = workers_data[worker_id]
        min_cost_for_worker = min(worker_costs[task] for task in task_letters)
        total += min_cost_for_worker
    return total


def compute_column_greedy(workers_data, task_letters):
    """
    Compute column-greedy total: each task's cheapest worker, summed.
    Ignores conflicts (multiple tasks assigned to same worker).
    """
    total = 0
    for task_letter in task_letters:
        worker_ids = sorted(workers_data.keys())
        min_cost_for_task = min(workers_data[worker_id][task_letter] for worker_id in worker_ids)
        total += min_cost_for_task
    return total


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_assignment_oracle_sweep(difficulty):
    """
    Test 60 questions across all seeds for a given difficulty.
    Verify: correct answer, distractors are incorrect, greedy heuristics
    appear in distractors.
    """
    unparseable = []
    found_greedy_misconception = False

    for seed in range(60):
        question = make("determine_the_optimum_assignment_s_by_inspection_for", seed=seed, difficulty=difficulty)

        # Parse question
        try:
            workers_data, task_letters = parse_assignment_question(question.text)
        except Exception as e:
            unparseable.append({"seed": seed, "difficulty": difficulty, "error": str(e), "text": question.text})
            continue

        # Compute minimum cost
        computed_min = find_min_cost_assignment(workers_data, task_letters)

        # Verify answer
        answer_int = int(question.answer_text)
        assert computed_min == answer_int, (
            f"Seed {seed} ({difficulty}): computed {computed_min}, but answer is {answer_int}"
        )

        # Verify distractors are not the minimum
        distractors = [int(d) for d in question.distractors]
        assert computed_min not in distractors, (
            f"Seed {seed} ({difficulty}): minimum {computed_min} appears in distractors {distractors}"
        )

        # Compute greedy heuristics
        row_greedy = compute_row_greedy(workers_data, task_letters)
        col_greedy = compute_column_greedy(workers_data, task_letters)

        # Check if at least one greedy appears in distractors
        if row_greedy in distractors or col_greedy in distractors:
            found_greedy_misconception = True

    # Report any parsing failures
    if unparseable:
        pytest.fail(
            f"Unparseable questions at {difficulty}:\n"
            + "\n".join(f"  Seed {u['seed']}: {u['error']}\n    Text: {u['text']}" for u in unparseable)
        )

    # Verify greedy misconception appears at this difficulty
    assert found_greedy_misconception, (
        f"No greedy misconception (row or column) found in any of 60 questions at {difficulty} difficulty"
    )
