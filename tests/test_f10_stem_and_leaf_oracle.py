"""Independent oracle for `f10-stem-and-leaf-plots`.

Reads only `question.text` (the key, the plot and the ask) and the public grader. It parses the plot itself,
works the answer out by its own rules and checks it against `question.answer_text` and the real `number`
grader. It does not import the generator module.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from fractions import Fraction

import pytest

from mathgen import make
from mathgen.grading.typed import grade_number

TOPIC = "f10_stem_and_leaf"
TIERS = ("easy", "medium", "hard")
N_SEEDS = int(os.environ.get("F10_STEM_ORACLE_SEEDS", "300"))


def _plot_body(text):
    return re.search(r"rows written as [^:]+: (.+?)\. (?:The leaves|Leaves)", text).group(1)


def _single(text):
    """(values with None for a hidden leaf, in plot order), from a single plot's text."""
    out = []
    for row in _plot_body(text).split("; "):
        stem, leaves = row.split(" | ")
        if leaves == "(none)":
            continue
        out += [None if x == "?" else 10 * int(stem) + int(x) for x in leaves.split()]
    return out


def _both(text):
    a, b = [], []
    for row in _plot_body(text).split("; "):
        left, stem, right = row.split(" | ")
        a += [10 * int(stem) + int(x) for x in left.split()]
        b += [10 * int(stem) + int(x) for x in right.split()]
    return a, b


def _median(v):
    s = sorted(v)
    n = len(s)
    return Fraction(s[n // 2]) if n % 2 else Fraction(s[n // 2 - 1] + s[n // 2], 2)


def _solve(text):
    if "back-to-back" in text:
        a, b = _both(text)
        if "difference between the median" in text:
            return abs(_median(a) - _median(b))
        assert "difference between the range" in text
        return abs((max(a) - min(a)) - (max(b) - min(b)))
    values = _single(text)
    if "hidden" in text and "What is the hidden" in text:
        i = values.index(None)
        known = [v for v in values if v is not None]
        n = len(values)
        m = re.search(r"The median is ([\d.]+)\.", text)
        if m:
            med = Fraction(m.group(1))
            assert i in (n // 2 - 1, n // 2)
            other = values[n // 2] if i == n // 2 - 1 else values[n // 2 - 1]
            return 2 * med - other
        r = int(re.search(r"The range of the .+? is (\d+)\.", text).group(1))
        if "is the largest" in text:
            assert i == n - 1
            return min(known) + r
        assert "is the smallest" in text and i == 0
        return max(known) - r
    if "How many" in text and "are shown in the plot" in text:
        return len(values)
    if "How many" in text:
        stem = int(re.search(r"in the (\d+)0s\?", text).group(1))
        return sum(1 for v in values if v // 10 == stem)
    if re.search(r"What is the largest", text):
        return max(values)
    if re.search(r"What is the smallest", text):
        return min(values)
    if "What is the median" in text:
        return _median(values)
    if "What is the range" in text:
        return max(values) - min(values)
    if "What is the mode" in text:
        (v, c), *rest = Counter(values).most_common()
        assert not rest or rest[0][1] < c
        return v
    raise AssertionError(text)


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_matches_canonical_answer_and_grader(tier):
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        solved = Fraction(_solve(q.text))
        text = str(float(solved)) if solved.denominator == 2 else str(int(solved))
        assert text == q.answer_text, (tier, seed, text, q.answer_text, q.text)
        assert grade_number(q, text).correct
        assert grade_number(q, str(int(solved) + 1)).correct is False or solved.denominator == 2


@pytest.mark.parametrize("tier", TIERS)
def test_oracle_rejects_every_distractor_and_no_two_tiers_share_a_dominant_answer(tier):
    pool = Counter()
    for seed in range(N_SEEDS):
        q = make(TOPIC, seed=seed, difficulty=tier)
        for d in q.distractors:
            assert not grade_number(q, d).correct
        pool[q.answer_text] += 1
    assert max(pool.values()) <= 0.25 * N_SEEDS
