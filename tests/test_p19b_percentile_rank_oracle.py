"""The oracle parses the data (raw list / frequency table / cumulative
frequency table) and the level from `question.text`, converts the level to
P itself, sorts, and returns the value at rank `P/100 * n`; asserts it
equals the key and that `P/100 * n` is a whole number.

Seed count is `N_SEEDS` (default 250 per tier, >= 1000 questions over the
four tiers), overridable by `P19B_ORACLE_SEEDS`.
"""

from __future__ import annotations

import os
import re

import pytest

from mathgen import make

TOPIC = "p19b_percentile_rank"
TIERS = ("easy", "medium", "hard", "exam")
N_SEEDS = int(os.environ.get("P19B_ORACLE_SEEDS", "250"))

_RAW_LIST_RE = re.compile(r": ((?:-?\d+, ){19}-?\d+)\. The Pth percentile")
_PAIR_RE = re.compile(r"(-?\d+): (\d+)")
_LEVEL_RE = re.compile(r"at (the [^.?]+)[.?]\s*$")
_PERCENTILE_RE = re.compile(r"the (\d+)(?:st|nd|rd|th) percentile")
_DECILE_RE = re.compile(r"the (\d+)(?:st|nd|rd|th) decile")
_QUARTILE_RE = re.compile(r"the (\d+)(?:st|nd|rd|th) quartile")


def _parse_level_p(text: str) -> int:
    m = _LEVEL_RE.search(text)
    assert m is not None, f"oracle: no trailing level phrase in {text!r}"
    phrase = m.group(1)
    pm = _PERCENTILE_RE.search(phrase)
    if pm:
        return int(pm.group(1))
    dm = _DECILE_RE.search(phrase)
    if dm:
        return 10 * int(dm.group(1))
    qm = _QUARTILE_RE.search(phrase)
    assert qm is not None, phrase
    return 25 * int(qm.group(1))


def _rank_value_from_list(values: list[int], k: int) -> int:
    return sorted(values)[k - 1]


def _rank_value_from_table(pairs: list[tuple[int, int]], k: int) -> int:
    total = 0
    for value, count in sorted(pairs):
        total += count
        if total >= k:
            return value
    raise AssertionError(f"oracle: rank {k} not reached in table {pairs}")


def _oracle_answer(question) -> int:
    text = question.text
    n = 20
    p = _parse_level_p(text)
    assert (p * n) % 100 == 0, f"oracle: P={p}, n={n} does not give a whole rank"
    k = (p * n) // 100

    if "cumulative" in text.lower():
        pairs = [(int(v), int(c)) for v, c in _PAIR_RE.findall(text)]
        # cumulative counts -> per-value counts by first-difference
        pairs.sort()
        per_value = []
        prev = 0
        for value, cum in pairs:
            per_value.append((value, cum - prev))
            prev = cum
        return _rank_value_from_table(per_value, k)
    if re.search(r"frequency table \(value: count\)", text):
        pairs = [(int(v), int(c)) for v, c in _PAIR_RE.findall(text)]
        return _rank_value_from_table(pairs, k)
    m = _RAW_LIST_RE.search(text)
    assert m is not None, f"oracle: no raw list found in {text!r}"
    values = [int(t) for t in m.group(1).split(", ")]
    return _rank_value_from_list(values, k)


def _all(tier, seeds=None):
    seeds = seeds if seeds is not None else range(N_SEEDS)
    return [make(TOPIC, seed=s, difficulty=tier) for s in seeds]


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_answer(tier):
    mismatches = []
    for q in _all(tier):
        oracle = _oracle_answer(q)
        if oracle != q.answer:
            mismatches.append((q.seed, q.text, oracle, q.answer))
    assert mismatches == [], mismatches


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_every_distractor_disagrees_with_the_oracle(tier):
    for q in _all(tier, seeds=range(60)):
        oracle = _oracle_answer(q)
        for d in q.distractors:
            assert int(d) != oracle, f"{tier} seed {q.seed}: distractor {d!r} matches the oracle"


def test_oracle_fires_on_a_corrupted_key():
    """Red-first proof: a corrupted key (answer + 1) must disagree with the
    oracle's own independently-recomputed rank value."""
    q = make(TOPIC, seed=0, difficulty="easy")
    oracle = _oracle_answer(q)
    corrupted_key = q.answer + 1
    assert oracle != corrupted_key, "oracle failed to catch a corrupted key"
