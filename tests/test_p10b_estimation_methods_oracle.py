"""Independent oracle for the `p10b_estimation_methods` questions: it re-derives each answer from the rendered
question text.

Parses ONLY the rendered question (``question.text`` and ``question.answer_text``) and re-derives the expected
answer from scratch.

Imports nothing from ``mathgen.generators.p10b_estimation_methods_build`` --
not a helper, not a constant, not a rounding rule. Every rounding/truncation
rule below is its own, separate implementation (string-slicing based, not the
generator's Fraction/``round_half_up`` route) so a bug shared between
construction and this oracle would have to be coincidental, not structural.
"""

from __future__ import annotations

import re

from mathgen import make

LD_TOPIC = "leading_digit_estimate"
AAS_TOPIC = "apply_approximation_strategy"
TIERS = ["easy", "medium", "hard"]
SEEDS = range(60)  # 60 seeds, matching the shared variety sweep


# --- Independent rounding rules (string-slicing, not Fraction/round_half_up) -


def _leading_digit_round(n: int) -> int:
    """Round ``n`` to its own leading digit (1 significant figure), ties
    away from zero -- derived here purely from the decimal string, never by
    importing the generator's own place-size or rounding helper."""
    s = str(n)
    if len(s) == 1:
        return n
    lead = int(s[0])
    rest = s[1:]
    round_up = int(rest[0]) >= 5
    if round_up:
        lead += 1
    return lead * 10 ** (len(s) - 1)


def _truncate_to_leading_digit(n: int) -> int:
    """Truncate ``n`` DOWN to its own leading digit -- never round."""
    s = str(n)
    return int(s[0]) * 10 ** (len(s) - 1)


def _round_to_place(n: int, place: int) -> int:
    """Round ``n`` to an explicitly stated place (10 or 100), independent of
    ``n``'s own leading-digit place."""
    digits_below = len(str(place)) - 1  # 1 for "ten", 2 for "hundred"
    s = str(n).zfill(digits_below + 1)
    kept, dropped = s[:-digits_below], s[-digits_below:]
    round_up = int(dropped[0]) >= 5
    base = int(kept) * place
    return base + place if round_up else base


# --- Parsing --------------------------------------------------------------


def _parse_expr(expr: str) -> tuple[list[int], list[str]]:
    """``"512 + 83 × 47"`` -> ``([512, 83, 47], ['+', '×'])``."""
    tokens = expr.rstrip(".").split()
    nums = [int(tok) for tok in tokens if tok not in ("+", "-", "×")]
    ops = [tok for tok in tokens if tok in ("+", "-", "×")]
    assert len(ops) == len(nums) - 1, f"unparseable expression: {expr!r}"
    return nums, ops


# --- Skill 1: leading_digit_estimate (ACMEM004) ---------------------------

_LD_RE = re.compile(r"^Round each number to its own leading digit, then estimate (.+)\.$")


def _oracle_leading_digit_estimate(text: str) -> int:
    m = _LD_RE.match(text)
    assert m, f"unparseable leading_digit_estimate question: {text!r}"
    nums, ops = _parse_expr(m.group(1))
    rounded = [_leading_digit_round(n) for n in nums]
    if "×" in ops:
        # This topic's only mixed-operator shape is "c + a × b": the
        # multiplication binds first, precedence-aware, independent of
        # however the generator's own construction ordered the operations.
        mult_index = ops.index("×")
        product = rounded[mult_index] * rounded[mult_index + 1]
        remaining = rounded[:mult_index] + [product] + rounded[mult_index + 2 :]
        remaining_ops = ops[:mult_index] + ops[mult_index + 1 :]
        total = remaining[0]
        for op, value in zip(remaining_ops, remaining[1:]):
            total = total + value if op == "+" else total - value
        return total
    total = rounded[0]
    for op, value in zip(ops, rounded[1:]):
        total = total + value if op == "+" else total - value
    return total


def test_leading_digit_estimate_oracle_matches_every_seed_every_tier():
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = make(LD_TOPIC, seed=seed, difficulty=tier)
            expected = _oracle_leading_digit_estimate(q.text)
            assert str(expected) == q.answer_text, (tier, seed, q.text, q.answer_text, expected)
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)


# --- Skill 2: apply_approximation_strategy (ACMEM010) ----------------------

_FRONT_END_RE = re.compile(r"^Use front-end estimation to estimate (.+)\.$")
_ROUND_TEN_RE = re.compile(r"^Round each number to the nearest ten, then estimate (.+)\.$")
_ROUND_HUNDRED_RE = re.compile(r"^Round each number to the nearest hundred, then estimate (.+)\.$")


def _oracle_apply_approximation_strategy(text: str) -> int:
    if m := _FRONT_END_RE.match(text):
        transform = _truncate_to_leading_digit
    elif m := _ROUND_TEN_RE.match(text):
        transform = lambda n: _round_to_place(n, 10)
    elif m := _ROUND_HUNDRED_RE.match(text):
        transform = lambda n: _round_to_place(n, 100)
    else:
        raise AssertionError(f"unparseable apply_approximation_strategy question: {text!r}")
    nums, ops = _parse_expr(m.group(1))
    transformed = [transform(n) for n in nums]
    total = transformed[0]
    for op, value in zip(ops, transformed[1:]):
        total = total + value if op == "+" else total - value
    return total


def test_apply_approximation_strategy_oracle_matches_every_seed_every_tier():
    checked = 0
    for tier in TIERS:
        for seed in SEEDS:
            q = make(AAS_TOPIC, seed=seed, difficulty=tier)
            expected = _oracle_apply_approximation_strategy(q.text)
            assert str(expected) == q.answer_text, (tier, seed, q.text, q.answer_text, expected)
            checked += 1
    assert checked == len(TIERS) * len(SEEDS)
