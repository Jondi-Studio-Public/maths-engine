"""Tests the counting-with-restrictions skill: counts re-derived by enumeration or an independent formula."""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction
from functools import cache
from itertools import permutations
from math import comb, factorial

import pytest

from mathgen.engine import grade
from tests import sr2b_support as sup

SKILL = "counting-with-restrictions"
TOPIC = "sr2b_counting_restrictions"
EXAM = "sr2b_exam_counting_restrictions"


def _word_count(word: str, merged: str | None = None) -> int:
    counts = Counter(word)
    n = len(word)
    if merged:
        copies = counts[merged]
        counts[merged] -= copies
        n -= copies - 1
        counts[merged + "*"] = 1
    value = factorial(n)
    for c in counts.values():
        value //= factorial(c)
    return value


@cache
def _row(n: int, kind: str, k: int = 2) -> int:
    total = 0
    for order in permutations(range(n)):
        where = [order.index(i) for i in range(k)]
        if kind == "together":
            ok = max(where) - min(where) == k - 1
        elif kind == "apart":
            ok = abs(where[0] - where[1]) != 1
        elif kind == "end":
            ok = where[0] in (0, n - 1)
        else:  # the two named people at the two ends
            ok = sorted(where[:2]) == [0, n - 1]
        total += ok
    return total


@cache
def _circle(n: int, kind: str) -> int:
    total = 0
    for rest in permutations(range(1, n)):
        seats = (0, *rest)
        gap = abs(seats.index(1) - seats.index(0)) % n
        adjacent = gap in (1, n - 1)
        total += True if kind == "none" else (adjacent if kind == "next" else not adjacent)
    return total


def _derive(stem: str) -> int:
    word = re.search(r"\b([A-Z]{5,})\b", stem)
    if word:
        return _word_count(word.group(1))
    committee = re.search(r"committee of (\d+)", stem)
    if committee:
        k = int(committee.group(1))
        m, w = (int(v) for v in re.search(r"(\d+) men and (\d+) women", stem).groups())
        if "Meera" in stem:
            return comb(m + w - 1, k - 1)
        j = int(re.search(r"(at least|at most) (\d+) wom", stem).group(2))
        low, high = (j, k) if "at least" in stem else (0, j)
        return sum(comb(w, i) * comb(m, k - i) for i in range(low, high + 1) if 0 <= k - i <= m)
    n = int(
        re.search(r"(\d+) (?:different )?(?:people|runners|flags|friends|students|books|players|[a-z]+)", stem).group(1)
    )
    if "circular" in stem or "round table" in stem:
        kind = "none"
        if "not next to each other" in stem:
            kind = "apart"
        elif "next to each other" in stem:
            kind = "next"
        return _circle(n, kind)
    if "not be next to each other" in stem:
        return _row(n, "apart")
    if "in any order among themselves" in stem:
        return _row(n, "together", 3)
    if "next to each other" in stem:
        return _row(n, "together")
    if "one at each end" in stem:
        return _row(n, "ends")
    assert "ends" in stem, stem
    return _row(n, "end")


def test_wired():
    sup.assert_wired(SKILL, TOPIC, EXAM, "number", ["permutations", "combinations", "senior-use-factorial-notation"])


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_an_independent_count(tier):
    for seed in range(120):
        q = sup.tier_question(TOPIC, tier, seed)
        assert int(q.answer_text) == _derive(q.text), (seed, q.text, q.answer_text)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=8)


def test_tiers_differ():
    easy = " ".join(sup.tier_question(TOPIC, "easy", s).text for s in range(40))
    medium = " ".join(sup.tier_question(TOPIC, "medium", s).text for s in range(80))
    hard = " ".join(sup.tier_question(TOPIC, "hard", s).text for s in range(80))
    assert "next to each other" not in easy and "committee" not in easy
    assert "next to each other" in medium and "end" in medium
    assert "committee" in hard and ("circular" in hard or "round table" in hard)
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["number", "number", "fraction", "number"], [1, 2, 2, 1], [False, False, True, True])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=30, top=0.3)


def test_exam_keys_match_an_independent_count():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        listed = re.search(r"\b\d(?:, \d)+\b", q.text)
        word = listed.group(0).replace(", ", "") if listed else re.search(r"\b([A-Z]{5,})\b", q.text).group(1)
        letter, many = re.search(r"the (two|three) (\w)s next to each other", q.text).group(2, 1)
        letter = letter.upper()
        assert word.count(letter) == {"two": 2, "three": 3}[many], (seed, q.text)
        total, together = _word_count(word), _word_count(word, letter)
        keys = [str(total), str(together), str(Fraction(together, total)), str(total - together)]
        assert [p.answer_text for p in q.parts] == keys, (seed, q.text)


def test_exam_follow_through_from_wrong_counts():
    sk = sup.skill_object(SKILL)
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        a, b = int(canon["a"]), int(canon["b"])
        wrong_b = b + 7
        replies = {**canon, "b": str(wrong_b), "c": str(Fraction(wrong_b, a)), "d": str(a - wrong_b)}
        result = grade(sk, q, replies)
        assert result.parts["c"].marks == q.parts[2].marks, seed
        assert result.parts["d"].marks == q.parts[3].marks, seed
        assert grade(sk, q, {**replies, "d": str(a - wrong_b + 1)}).parts["d"].marks == 0, seed
