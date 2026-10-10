"""Independent oracle for `unknown_add_sub_equation`: it solves the printed equation by trial.

The prompt contains one equation with a square blank. The oracle replaces the blank with each whole number from 0
upward, evaluates both sides left to right with its own tiny evaluator, and records every value that makes
the sides equal. There must be exactly one, it must equal `answer_text`, and no distractor may be it. The
equation is also checked to use only `+` and `-` between whole numbers and to have exactly one blank. It does not
import the generator.

Seed count is `N_SEEDS` (default 250 per tier), overridable by `UNKNOWN_EQ_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make

TOPIC = "unknown_add_sub_equation"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("UNKNOWN_EQ_ORACLE_SEEDS", "250"))
_EQUATION = re.compile(r"((?:\d+|□)(?: [+-] (?:\d+|□))*) = ((?:\d+|□)(?: [+-] (?:\d+|□))*)")


def _eval(side: str, blank: int) -> int:
    tokens = side.replace("□", str(blank)).split()
    total = int(tokens[0])
    for sign, n in zip(tokens[1::2], tokens[2::2]):
        total = total + int(n) if sign == "+" else total - int(n)
    return total


def _solutions(text: str):
    found = _EQUATION.findall(text)
    assert len(found) == 1, f"oracle: expected one equation in {text!r}"
    left, right = found[0]
    assert (left + right).count("□") == 1
    return [v for v in range(0, 1300) if _eval(left, v) == _eval(right, v)]


@pytest.mark.parametrize("tier", TIERS)
def test_the_equation_has_exactly_one_solution_and_it_is_the_key(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        solutions = _solutions(q.text)
        assert solutions == [int(q.answer_text)], (
            f"{tier} seed {seed}: {q.text!r} solves to {solutions}, key {q.answer_text}"
        )
        assert all(int(d) not in solutions for d in q.distractors)


@pytest.mark.parametrize("tier", TIERS)
def test_the_numbers_stay_in_the_tier_range(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        shown = [int(n) for n in re.findall(r"\d+", _EQUATION.search(q.text).group(0))]
        top = max(shown + [int(q.answer_text)])
        if tier == "hard":
            assert top >= 100, q.text
        else:
            assert top <= 199, q.text
