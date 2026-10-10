"""Independent oracle for `multidigit_add_sub_regroup`: it re-derives each answer from the rendered prompt.

The prompt carries exactly two whole numbers. Its wording fixes the operation and the order: "Add", "sum" and
"+" mean addition; "Subtract b from a" means a - b; "difference between" means larger minus smaller; otherwise
"a - b". The oracle then computes the answer digit by digit with its own carry or borrow loop (not Python's
`+`/`-`), compares it to `answer_text`, and checks each distractor differs. It also counts the regroups the
printed numbers force and compares that to the tier. It does not import the generator.

Seed count is `N_SEEDS` (default 250 per tier), overridable by `MULTIDIGIT_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make

TOPIC = "multidigit_add_sub_regroup"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("MULTIDIGIT_ORACLE_SEEDS", "250"))


def _parse(text: str):
    nums = [int(n) for n in re.findall(r"\d+", text)]
    assert len(nums) == 2, f"oracle: expected two numbers in {text!r}"
    if "+" in text or text.startswith("Add") or "sum of" in text:
        return "add", nums[0], nums[1]
    if text.startswith("Subtract"):
        return "sub", nums[1], nums[0]
    if "difference between" in text:
        return "sub", max(nums), min(nums)
    assert " - " in text, f"oracle: no operation found in {text!r}"
    return "sub", nums[0], nums[1]


def _column_add(a: int, b: int):
    xs, ys = str(a)[::-1], str(b)[::-1]
    out, carry, carries = [], 0, 0
    for i in range(max(len(xs), len(ys))):
        s = (int(xs[i]) if i < len(xs) else 0) + (int(ys[i]) if i < len(ys) else 0) + carry
        out.append(s % 10)
        carry = s // 10
        carries += carry
    if carry:
        out.append(carry)
    return int("".join(map(str, out[::-1]))), carries


def _column_sub(a: int, b: int):
    xs, ys = str(a)[::-1], str(b)[::-1]
    out, borrow, borrows = [], 0, 0
    for i in range(len(xs)):
        d = int(xs[i]) - (int(ys[i]) if i < len(ys) else 0) - borrow
        borrow = 1 if d < 0 else 0
        borrows += borrow
        out.append(d + 10 * borrow)
    assert borrow == 0, "oracle: subtraction went negative"
    return int("".join(map(str, out[::-1]))), borrows


@pytest.mark.parametrize("tier", TIERS)
def test_key_matches_a_column_method_and_regroup_count_matches_the_tier(tier):
    expected_regroups = {"easy": (1, 1), "medium": (2, 2), "hard": (3, 3)}
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        op, a, b = _parse(q.text)
        value, regroups = _column_add(a, b) if op == "add" else _column_sub(a, b)
        assert int(q.answer_text) == value, f"{tier} seed {seed}: {q.text!r} -> {q.answer_text} (expected {value})"
        assert all(int(d) != value for d in q.distractors)
        if op == "add":
            assert regroups == expected_regroups[tier][0], f"{tier} seed {seed}: {q.text!r} carries {regroups}"
        else:
            # a borrow through a zero is two borrows (tens and ones) in the hard tier, as in medium
            want = {"easy": 1, "medium": 2, "hard": 2}[tier]
            assert regroups == want, f"{tier} seed {seed}: {q.text!r} borrows {regroups}"


@pytest.mark.parametrize("tier", TIERS)
def test_operand_sizes_match_the_tier(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        _op, a, b = _parse(q.text)
        digits = sorted((len(str(a)), len(str(b))))
        if tier == "easy":
            assert digits == [2, 2], q.text
        else:
            assert digits[1] == 3 and digits[0] in (2, 3), q.text
