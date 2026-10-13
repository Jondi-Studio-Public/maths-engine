"""Exam chain of `senior-solve-problems-involving-permutations-and-restrictions-with-or`: restricted letter counts."""

from __future__ import annotations

import re
from fractions import Fraction
from itertools import combinations
from math import factorial

import pytest
from sympy.utilities.iterables import multiset_permutations

from tests import exd_x_r3a_support as S

TOPIC = "exd_x_r3a_vowel_arrangements"
SKILL = S.load_skill("senior-solve-problems-involving-permutations-and-restrictions-with-or")
VOWELS = set("AEIOU")


def _letters(q) -> str:
    m = re.search(r"letters of the word ([A-Z]+) are arranged", q.stem)
    if m:
        return m.group(1)
    m = re.search(r"letter tiles show ([A-Z, ]+)\.", q.stem)
    return m.group(1).replace(", ", "")


def _members(q):
    m = re.search(r"copies of the letter ([A-Z])", q.text)
    return {m.group(1)} if m else VOWELS


def _rule(q) -> str:
    return "apart" if "next to each other" in q.parts[2].prompt else "together"


def _distinct(letters: str) -> int:
    out = factorial(len(letters))
    for ch in set(letters):
        out //= factorial(letters.count(ch))
    return out


def _count_by_positions(letters: str, members, rule: str) -> int:
    n = len(letters)
    v = sum(ch in members for ch in letters)
    inside = [ch for ch in letters if ch in members]
    outside = [ch for ch in letters if ch not in members]
    per = _distinct("".join(inside)) * _distinct("".join(outside))
    total = 0
    for chosen in combinations(range(n), v):
        gaps = [b - a for a, b in zip(chosen, chosen[1:])]
        ok = all(g > 1 for g in gaps) if rule == "apart" else all(g == 1 for g in gaps)
        total += per if ok else 0
    return total


def oracle(q):
    letters, members, rule = _letters(q), _members(q), _rule(q)
    v = sum(ch in members for ch in letters)
    c = len(letters) - v
    inside = "".join(ch for ch in letters if ch in members)
    outside = "".join(ch for ch in letters if ch not in members)
    total = _count_by_positions(letters, members, rule)
    if rule == "apart":
        a = _distinct(outside)
        # ways to place the vowels: choose v of the c + 1 gaps, then order them
        from math import comb

        b = comb(c + 1, v) * _distinct(inside)
    else:
        a = _distinct(inside)
        b = _distinct(outside + "#")  # the block is one extra object
    prob = Fraction(total, _distinct(letters))
    return {
        "a": str(a),
        "b": str(b),
        "c": str(total),
        "d": str(prob.numerator) if prob.denominator == 1 else f"{prob.numerator}/{prob.denominator}",
    }


@pytest.fixture(scope="module")
def questions():
    return S.build(TOPIC)


def test_wired():
    S.check_wired(SKILL, TOPIC)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.follow_through is not None for p in q.parts] == [False, False, True, True]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "fraction"]
    S.check_shapes(questions, SKILL)
    S.check_text(questions)


def test_oracle_agrees(questions):
    S.check_oracle(questions, oracle)


def test_counts_match_full_enumeration_for_short_words(questions):
    checked = 0
    for q in questions:
        letters = _letters(q)
        if _distinct(letters) > 6000:
            continue
        members, rule = _members(q), _rule(q)
        good = 0
        for perm in multiset_permutations(list(letters)):
            pos = [i for i, ch in enumerate(perm) if ch in members]
            gaps = [b - a for a, b in zip(pos, pos[1:])]
            good += all(g > 1 for g in gaps) if rule == "apart" else all(g == 1 for g in gaps)
        assert str(good) == q.parts[2].answer_text, q.text
        checked += 1
    assert checked >= 20


def test_full_marks_and_distractors(questions):
    S.check_full_marks(SKILL, questions)
    S.check_distractors_rejected(SKILL, questions)
    S.check_number_format([q for q in questions])
    S.check_misconception_tags(questions)


def test_answer_space(questions):
    S.check_answer_space(questions)


def test_follow_through_from_wrong_counts(questions):
    def chain(q):
        letters = _letters(q)
        a, b = Fraction(q.parts[0].distractors[0]), Fraction(q.parts[1].distractors[0])
        total = a * b
        prob = total / _distinct(letters)
        if prob >= 1 or len({a, b, total}) < 3:
            return None
        text = str(prob.numerator) if prob.denominator == 1 else f"{prob.numerator}/{prob.denominator}"
        return {"a": str(a), "b": str(b), "c": str(total), "d": text}, {"c", "d"}

    S.check_follow_through(SKILL, questions, chain, n=120, min_seen=10)


def test_all_wrong_zero(questions):
    S.check_all_wrong_zero(SKILL, questions, {"a": "9999999", "b": "8888888", "c": "7777777", "d": "6/5"})


def test_deterministic():
    S.check_deterministic(TOPIC)
