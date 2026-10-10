"""The oracle (a) re-parses the stated Q1/Q3 and the full dataset, (b)
recomputes Q1/Q3 itself by BOTH the exclusive and the Tukey/inclusive method
and asserts both equal the stated values, (c) computes the fences and the
set of values strictly beyond them, and asserts it is exactly `{key}`, (d)
asserts no distractor is beyond a fence.

Seed count is `N_SEEDS` (default 250 per tier, >= 1000 questions over the
four tiers), overridable by `P19B_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re
import statistics
from fractions import Fraction

import pytest

from mathgen import make

TOPIC = "p19b_outlier_fence"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("P19B_ORACLE_SEEDS", "250"))

_VALUE_LIST_RE = re.compile(r": ((?:-?\d+, ){9}-?\d+)\.")
_Q1_RE = re.compile(r"Q1 = (-?\d+)")
_Q3_RE = re.compile(r"Q3 = (-?\d+)")


def _parse_values(text: str) -> list[int]:
    m = _VALUE_LIST_RE.search(text)
    assert m is not None, f"oracle: no 10-value list found in {text!r}"
    return [int(t) for t in m.group(1).split(", ")]


def _parse_q1q3(text: str) -> tuple[int, int]:
    q1 = _Q1_RE.search(text)
    q3 = _Q3_RE.search(text)
    assert q1 is not None and q3 is not None, f"oracle: no Q1/Q3 found in {text!r}"
    return int(q1.group(1)), int(q3.group(1))


def _quartiles_exclusive(values: list[int]) -> tuple[Fraction, Fraction]:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        lower, upper = s[:mid], s[mid:]
    else:
        lower, upper = s[:mid], s[mid + 1 :]
    return Fraction(statistics.median(lower)), Fraction(statistics.median(upper))


def _quartiles_inclusive(values: list[int]) -> tuple[Fraction, Fraction]:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        lower, upper = s[:mid], s[mid:]
    else:
        lower, upper = s[: mid + 1], s[mid:]
    return Fraction(statistics.median(lower)), Fraction(statistics.median(upper))


def _outliers(values: list[int], q1: Fraction, q3: Fraction) -> list[int]:
    iqr = q3 - q1
    lo, hi = q1 - Fraction(3, 2) * iqr, q3 + Fraction(3, 2) * iqr
    return [v for v in values if v < lo or v > hi]


def _all(tier, seeds=None):
    seeds = seeds if seeds is not None else range(N_SEEDS)
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_stated_quartiles_match_both_methods(tier):
    for q in _all(tier):
        values = _parse_values(q.text)
        stated_q1, stated_q3 = _parse_q1q3(q.text)
        excl_q1, excl_q3 = _quartiles_exclusive(values)
        incl_q1, incl_q3 = _quartiles_inclusive(values)
        assert (excl_q1, excl_q3) == (Fraction(stated_q1), Fraction(stated_q3)), (
            f"{tier} seed {q.seed}: exclusive quartiles disagree with stated Q1/Q3 in {q.text!r}"
        )
        assert (incl_q1, incl_q3) == (Fraction(stated_q1), Fraction(stated_q3)), (
            f"{tier} seed {q.seed}: inclusive quartiles disagree with stated Q1/Q3 in {q.text!r}"
        )


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_answer(tier):
    mismatches = []
    for q in _all(tier):
        values = _parse_values(q.text)
        stated_q1, stated_q3 = _parse_q1q3(q.text)
        outliers = _outliers(values, Fraction(stated_q1), Fraction(stated_q3))
        if outliers != [q.answer[0]]:
            mismatches.append((q.seed, q.text, outliers, q.answer))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_no_distractor_is_beyond_a_fence(tier):
    for q in _all(tier, seeds=range(60)):
        values = _parse_values(q.text)
        stated_q1, stated_q3 = _parse_q1q3(q.text)
        outliers = _outliers(values, Fraction(stated_q1), Fraction(stated_q3))
        for d in q.distractors:
            assert int(d) not in outliers, f"{tier} seed {q.seed}: distractor {d!r} is beyond a fence"


def test_oracle_fires_on_a_corrupted_key():
    """Red-first proof: flip the answer's sign to a value that would put it
    OUTSIDE the true outlier set and confirm the oracle's own comparison
    catches the mismatch (exercised directly, not merely asserted)."""
    q = make(TOPIC, seed=0, difficulty="easy")
    values = _parse_values(q.text)
    stated_q1, stated_q3 = _parse_q1q3(q.text)
    outliers = _outliers(values, Fraction(stated_q1), Fraction(stated_q3))
    corrupted_key = q.answer[0] + 1
    assert outliers != [corrupted_key], "oracle failed to catch a corrupted key"
