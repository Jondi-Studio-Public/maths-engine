"""Exam-tier source for `f10-number-algorithms-decisions`: trace an algorithm that LOOPS until a stopping
condition is met.

The practice tiers apply a fixed one- or two-step rule once to each given input. Here the learner is told a
decision rule and a stopping test and must run the rule repeatedly, deciding at every pass which branch applies
and when to stop, then list every value the algorithm visits (start and final value included). The number of
passes is not stated; it is found by running the algorithm.

The sequence of values is produced by one simulation and then recomputed independently with a second,
differently written loop (`_replay`). Rules and starting values are drawn until the run stops within a
modest number of passes, so every question terminates.
"""

from __future__ import annotations

import random

from ..core import check, generator, pick_distractors

_MAX_PASSES = 14
_MIN_PASSES = 4

# (test text, test function, branch-1 text, branch-1 function, other-branch candidates)
_FAMILIES = {
    "even": (
        "even",
        lambda n: n % 2 == 0,
        "n ÷ 2",
        lambda n: n // 2,
        [("3n + 1", lambda n: 3 * n + 1), ("n + 5", lambda n: n + 5), ("n + 3", lambda n: n + 3)],
        [("n = 1", lambda n: n == 1)],
    ),
    "multiple_of_3": (
        "a multiple of 3",
        lambda n: n % 3 == 0,
        "n ÷ 3",
        lambda n: n // 3,
        [("n + 2", lambda n: n + 2), ("2n + 1", lambda n: 2 * n + 1), ("n - 4", lambda n: n - 4)],
        [("n = 1", lambda n: n == 1), ("n < 4", lambda n: n < 4), ("n < 6", lambda n: n < 6)],
    ),
}


def _run(start, is_branch1, branch1, other, stop, first_branch_wrong=False):
    """The visited values, or None if the stop is not reached within the pass limit."""
    values = [start]
    n = start
    for step in range(_MAX_PASSES):
        if stop(n):
            return values
        use_b1 = is_branch1(n)
        if first_branch_wrong and step == 0:
            use_b1 = not use_b1
        n = branch1(n) if use_b1 else other(n)
        values.append(n)
    return values if stop(n) else None


def _replay(start, kind, other_text, stop_text):
    """Second route for the key: re-derive the visited values from the rule's TEXT with a while loop."""
    other = {t: f for t, f in _FAMILIES[kind][4]}[other_text]
    stop = {t: f for t, f in _FAMILIES[kind][5]}[stop_text]
    n, out = start, [start]
    divisor = 2 if kind == "even" else 3
    while not stop(n):
        n = n // divisor if n % divisor == 0 else other(n)
        out.append(n)
        check(len(out) <= _MAX_PASSES + 1, "replay did not terminate")
    return out


@generator(
    "exd_x_f0_algorithm_loop",
    level=10,
    presets={
        "easy": dict(tier="exam"),
        "medium": dict(tier="exam"),
        "hard": dict(tier="exam"),
        "exam": dict(tier="exam"),
    },
)
def exd_x_f0_algorithm_loop(rng: random.Random, tier: str = "exam") -> dict:
    check(tier == "exam", "this topic only builds the exam tier")
    for _ in range(500):
        kind = rng.choice(sorted(_FAMILIES))
        test_text, is_b1, b1_text, b1, others, stops = _FAMILIES[kind]
        other_text, other = rng.choice(others)
        stop_text, stop = rng.choice(stops)
        start = rng.randint(9, 40)
        values = _run(start, is_b1, b1, other, stop)
        if values is None or not (_MIN_PASSES <= len(values) - 1 <= 9) or len(set(values)) != len(values):
            continue
        if kind == "multiple_of_3" and min(values) < 1:
            continue
        # every branch must actually be used, or the decision is decoration
        used = {is_b1(v) for v in values[:-1]}
        if used != {True, False}:
            continue
        break
    else:
        check(False, "no terminating algorithm found")
    check(values == _replay(start, kind, other_text, stop_text), "independent replay disagrees with the trace")

    text = (
        f"An algorithm starts with n = {start} and repeats this step: if n is {test_text}, replace n by "
        f"{b1_text}; otherwise replace n by {other_text}. It stops as soon as {stop_text}. "
        "List every value of n in order, including the starting value and the final value."
    )
    answer_text = ", ".join(str(v) for v in values)

    wrong_first = _run(start, is_b1, b1, other, stop, first_branch_wrong=True)
    cands = [
        values[:-1],  # stops one pass early
        values[1:],  # leaves out the starting value
        values + [b1(values[-1]) if is_b1(values[-1]) else other(values[-1])],  # runs one pass past the stop
        wrong_first,  # takes the wrong branch on the first pass
    ]
    texts = [", ".join(str(v) for v in c) for c in cands if c]
    return dict(
        text=text,
        latex=f"plain:{text}",
        answer=values,
        answer_text=answer_text,
        answer_latex=answer_text,
        working=[f"n = {a} gives {b}." for a, b in zip(values, values[1:])],
        distractors=pick_distractors(answer_text, texts),
    )
