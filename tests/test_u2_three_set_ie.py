"""Tests `ucv1-three-set-inclusion-exclusion-derangements` (topics `u2_three_set_ie` and its exam chain).

Oracles read the numbers back out of the question text and rebuild the eight Venn regions from them, count integers
directly, and list permutations.
"""

from __future__ import annotations

import itertools
import re
from math import comb, factorial

import pytest

from mathgen import make
from tests import u2_support as su

TOPIC = "u2_three_set_ie"
EXAM = "exd_x_u2_three_set_ie"
SK = su.skill("ucv1-three-set-inclusion-exclusion-derangements")


def _regions(sizes, pairs, triple):
    """The seven Venn regions from the set sizes, pairwise counts and triple count; all must be non-negative."""
    a, b, c = sizes
    ab, ac, bc = pairs
    only = (a - ab - ac + triple, b - ab - bc + triple, c - ac - bc + triple)
    two = (ab - triple, ac - triple, bc - triple)
    regions = (*only, *two, triple)
    assert all(r >= 0 for r in regions), regions
    return only, two


def _numbers(text):
    return [int(v) for v in re.findall(r"\b(\d+)\b", text)]


def test_wiring():
    su.check_wiring(SK, TOPIC, EXAM, "number")


@pytest.mark.parametrize("tier", su.TIERS)
def test_key_and_distractors(tier):
    su.check_tier(SK, TOPIC, tier)


def test_easy_missing_triple_rebuilds_the_regions():
    for seed in range(100):
        q = make(TOPIC, seed=seed, difficulty="easy")
        nums = _numbers(q.text)
        union, sizes, pairs = nums[0], nums[1:4], nums[4:7]
        only, two = _regions(sizes, pairs, int(q.answer))
        assert sum(only) + sum(two) + int(q.answer) == union, (seed, q.text)


def test_medium_survey_and_derangement_keys():
    seen = set()
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="medium")
        if "envelopes" in q.text:
            n = int(q.text.split()[0])
            listed = sum(1 for p in itertools.permutations(range(n)) if all(p[i] != i for i in range(n)))
            assert int(q.answer) == listed
            seen.add("derangement")
            continue
        nums = _numbers(q.text)
        total, sizes, pairs, triple = nums[0], nums[1:4], nums[4:7], nums[7]
        only, two = _regions(sizes, pairs, triple)
        if "exactly two" in q.text:
            assert int(q.answer) == sum(two), (seed, q.text)
            seen.add("two")
        else:
            assert int(q.answer) == total - sum(only) - sum(two) - triple, (seed, q.text)
            assert int(q.answer) >= 0
            seen.add("none")
    assert seen == {"derangement", "two", "none"}


def test_hard_keys_by_direct_counting():
    seen = set()
    for seed in range(160):
        q = make(TOPIC, seed=seed, difficulty="hard")
        if "divisible by none" in q.text:
            n = int(re.search(r"to (\d+) inclusive", q.text).group(1))
            p, r, s = (int(v) for v in re.search(r"none of (\d+), (\d+) and (\d+)", q.text).groups())
            assert int(q.answer) == sum(1 for k in range(1, n + 1) if k % p and k % r and k % s)
            seen.add("divisible")
        elif "umbrella" in q.text:
            n = int(q.text.split()[0])
            k = int(re.search(r"exactly (\d+) friend", q.text).group(1))
            listed = sum(1 for p in itertools.permutations(range(n)) if sum(p[i] == i for i in range(n)) == k)
            assert int(q.answer) == listed
            seen.add("fixed")
        else:
            n = int(q.text.split()[0])
            j = len(re.findall(r"seat \d", q.text.split("(")[1].split(")")[0]))
            listed = sum(1 for p in itertools.permutations(range(n)) if all(p[i] != i for i in range(j)))
            assert int(q.answer) == listed
            seen.add("forbidden")
    assert seen == {"divisible", "fixed", "forbidden"}


def test_exam_chain_keys_and_shape():
    su.check_exam(SK, EXAM, "abcd", (3, 2, 2, 3), (False, True, True, True))
    for seed in range(100):
        q = make(EXAM, seed=seed, difficulty="exam")
        nums = _numbers(q.stem)
        total, none, sizes, pairs = nums[0], nums[1], nums[2:5], nums[5:8]
        t, one, two, d = (int(p.answer) for p in q.parts)
        only, twos = _regions(sizes, pairs, t)
        assert sum(only) + sum(twos) + t == total - none
        assert one == sum(only) and two == sum(twos) and 3 <= two <= 6
        listed = sum(1 for p in itertools.permutations(range(two)) if all(p[i] != i for i in range(two)))
        assert d == listed


def test_exam_follow_through_from_a_wrong_triple():
    def exactly_one(q, wrong):
        nums = _numbers(q.stem)
        return sum(nums[2:5]) - 2 * sum(nums[5:8]) + 3 * wrong

    su.check_follow_through(SK, EXAM, "a", "b", lambda q: int(q.parts[0].answer) + 3, exactly_one)


def test_exam_follow_through_into_the_derangement():
    def derangements_of(q, wrong):
        return sum((-1) ** i * comb(wrong, i) * factorial(wrong - i) for i in range(wrong + 1))

    su.check_follow_through(SK, EXAM, "c", "d", lambda q: int(q.parts[2].answer) + 1, derangements_of)


def test_variety():
    su.distinct_stems(TOPIC, "easy")
    su.answer_variety(TOPIC, "hard", minimum=15)
    su.answer_variety(EXAM, "exam", seeds=60, minimum=25)
