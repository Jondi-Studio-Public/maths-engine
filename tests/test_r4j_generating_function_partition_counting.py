"""Tests `uc2-generating-function-partition-counting` (topic `r4j_generating_function_partition_counting`) and its exam chain.

The oracle reads the numbers out of the TEXT and counts by brute force: every multiset of parts (or every subset, or
every use of limited coins) is enumerated directly, a different route from the series product in the generator. A SymPy
series of the product is a third route on the exam chain.
"""

from __future__ import annotations

import itertools
import re

import sympy

import mathgen.generators.r4j_generating_function_partition_counting_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import load_skills
from tests.r4j_support import check_carried, check_parts, check_single

TOPIC = "r4j_generating_function_partition_counting"
EXAM = "exd_x_r4j_generating_function_partition_counting"
SKILL = next(s for s in load_skills()[1] if s.id == "uc2-generating-function-partition-counting")
x = sympy.Symbol("x")


def _multisets(sizes, n, caps=None):
    caps = caps or {}
    ranges = [range(caps.get(s, n // s) + 1) for s in sizes]
    return sum(1 for t in itertools.product(*ranges) if sum(c * s for c, s in zip(t, sizes)) == n)


def _oracle(tier, text):
    nums = [int(v) for v in re.findall(r"\d+", text)]
    if tier == "easy":
        if "-point" in text:
            a, b, c, n = nums[:4]
        else:
            n, a, b, c = nums[:4]
        return _multisets([a, b, c], n)
    if tier == "medium":
        if "cards" in text:
            m, n = nums[0], nums[3]
        elif "partitions" in text:
            n, m = nums[0], nums[1]
        else:
            n, m = nums[0], nums[2]
        return sum(1 for r in range(1, m + 1) for t in itertools.combinations(range(1, m + 1), r) if sum(t) == n)
    n = int(re.search(r"(?:exactly|make) \$(\d+)", text).group(1))
    words = {"one": 1, "two": 2, "five": 5, "ten": 10}
    caps = {words[w]: int(c) for c, w in re.findall(r"only (\d+) of the (\w+)-dollar", text)}
    return _multisets([1, 2, 5, 10], n, caps)


def test_skill_metadata():
    assert list(SKILL.year_level) == ["university"] and "university-year-2" in SKILL.tags


def test_every_tier_matches_brute_force():
    for tier in ("easy", "medium", "hard"):
        for seed in range(40):
            q = make(TOPIC, seed=seed, difficulty=tier)
            assert int(q.answer_text) == _oracle(tier, q.text), (tier, seed, q.text, q.answer_text)
            check_single(q, "number")


def test_hard_limits_matter():
    differing = 0
    for seed in range(40):
        q = make(TOPIC, seed=seed, difficulty="hard")
        n = int(re.search(r"(?:exactly|make) \$(\d+)", q.text).group(1))
        differing += int(q.answer_text) != _multisets([1, 2, 5, 10], n)
    assert differing == 40


def test_exam_chain():
    for seed in range(30):
        q = make(EXAM, seed=seed, difficulty="exam")
        check_parts(q)
        n = int(re.search(r"(?:exactly|awards) (\d+)", q.text).group(1))
        a, b, c, d = (int(p.answer_text) for p in q.parts)
        assert [a, b, c] == [_multisets([3, 4, 7], n - j) for j in range(3)]
        assert d == a + b + c == _multisets([1, 3, 4, 7], n, {1: 2})
        gf = (1 + x + x**2) / ((1 - x**3) * (1 - x**4) * (1 - x**7))
        assert sympy.series(gf, x, 0, n + 1).removeO().coeff(x, n) == d
        check_carried(q, "b", str(b + 2), str(a + b + 2 + c))
