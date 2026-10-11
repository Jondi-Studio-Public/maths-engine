"""Exam tier of the multiplication-principle, permutations and combinations skills.

Each oracle below re-derives the answer from the rendered prompt alone, by brute-force enumeration or by a
search over the integers, and shares no code with `mathgen.generators.exd_x_h2_counting`.
"""

from __future__ import annotations

import re
from itertools import combinations, permutations, product
from math import comb, factorial, perm

import pytest

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

MULT = "exd_x_h2_multiplication_principle"
PERM = "exd_x_h2_permutations"
COMB = "exd_x_h2_combinations"

# The skill's existing (practice) source for each topic: (topic, params).
LOWER = {
    MULT: ("combinatorics", {"kind": "multiplication"}),
    PERM: ("combinatorics", {"kind": "permutation"}),
    COMB: ("combinatorics", {"kind": "combination"}),
}


def _exam(topic, seed):
    return make(topic, seed=seed, difficulty="exam")


@pytest.mark.parametrize("topic", [MULT, PERM, COMB])
def test_no_text_defects(topic):
    for seed in range(100):
        q = _exam(topic, seed)
        assert find_text_defects(q, include_working=True) == [], (seed, q.text)


@pytest.mark.parametrize("topic", [MULT, PERM, COMB])
def test_prompt_variety_and_new_shapes(topic):
    shapes = {prompt_shape(_exam(topic, s)) for s in range(60)}
    assert len(shapes) >= 3, shapes
    lower_topic, params = LOWER[topic]
    lower = {prompt_shape(make(lower_topic, seed=s, difficulty="easy", **params)) for s in range(60)}
    assert shapes - lower, "the exam source adds no prompt shape over the practice tier"


# -- oracles -------------------------------------------------------------------


def _oracle_mult(text: str) -> int:
    nums = [int(v) for v in re.findall(r"\d+", text)]
    if "boys" in text:
        m, g = nums[0], nums[1]
        patterns = sum(
            1 for p in product("BG", repeat=m + g) if p.count("B") == m and all(a != b for a, b in zip(p, p[1:]))
        )
        return patterns * factorial(m) * factorial(g)
    if "at least one" in text:
        size = 26 if "letters" in text else 10
        special = {"digit 7": 1, "even digit": 5, "3 or an 8": 2, "vowel": 5, "X or a Z": 2}
        s = next(v for key, v in special.items() if key in text)
        k = nums[0]
        # dp[has_special] = number of strings of the current length
        dp = {False: 1, True: 0}
        for _ in range(k):
            dp = {
                False: dp[False] * (size - s),
                True: dp[True] * size + dp[False] * s,
            }
        return dp[True]
    # digit code
    k = int(re.search(r"(\d+)(?:-digit| different digits)", text).group(1))
    if "multiple of 5" in text:
        last = {0, 5}
    elif "even" in text:
        last = {0, 2, 4, 6, 8}
    else:
        last = {2, 3, 5, 7}
    return sum(1 for d in permutations(range(10), k) if d[0] != 0 and d[-1] in last)


def _oracle_perm(text: str) -> int:
    nums = [int(v) for v in re.findall(r"\d+", text)]
    if "order does not matter" in text or "when it does not" in text:
        p, c = nums[0], nums[1]
        hits = [(n, r) for n in range(1, 40) for r in range(1, n + 1) if perm(n, r) == p and comb(n, r) == c]
        assert len(hits) == 1, (text, hits)
        return hits[0][0]
    if "vice-captain" in text:
        r = 2
    elif "deputy chair" in text:
        r = 4
    else:
        r = 3
    value = nums[0]
    hits = [n for n in range(r, 60) if perm(n, r) == value]
    assert len(hits) == 1, (text, hits)
    return hits[0]


def _oracle_comb(text: str) -> int:
    nums = [int(v) for v in re.findall(r"\d+", text)]
    if "points" in text:
        n, line = nums[0], nums[1]
        return sum(1 for t in combinations(range(n), 3) if not all(p < line for p in t))
    if "men" in text:
        j = 1 if "at least one" in text else 2
        # numbers in order of appearance: k, men, women  OR  men, women, k
        if text.startswith("A committee"):
            k, men, women = nums[0], nums[1], nums[2]
        else:
            men, women, k = nums[0], nums[1], nums[2]
        return sum(1 for t in combinations(range(men + women), k) if sum(1 for p in t if p < women) >= j)
    # two people who will not serve together
    if text.startswith("A team"):
        k, n = nums[0], nums[1]
    else:
        n, k = nums[0], nums[1]
    return sum(1 for t in combinations(range(n), k) if not (0 in t and 1 in t))


@pytest.mark.parametrize("topic, oracle", [(MULT, _oracle_mult), (PERM, _oracle_perm), (COMB, _oracle_comb)])
def test_key_matches_independent_oracle_over_300_seeds(topic, oracle):
    mismatches = []
    for seed in range(300):
        q = _exam(topic, seed)
        expected = oracle(q.text)
        if str(expected) != q.answer_text or q.answer != expected:
            mismatches.append((seed, q.text, expected, q.answer_text))
    assert mismatches == [], mismatches[:3]


@pytest.mark.parametrize("topic", [MULT, PERM, COMB])
def test_grader_accepts_key_and_rejects_every_distractor(topic):
    grade = GRADERS["number"]
    bad = []
    for seed in range(150):
        q = _exam(topic, seed)
        assert q.distractors, seed
        if not grade(q, q.answer_text).correct:
            bad.append(("key rejected", seed, q.answer_text))
        for d in q.distractors:
            if grade(q, d).correct:
                bad.append(("distractor accepted", seed, d, q.answer_text))
    assert bad == [], bad[:3]


@pytest.mark.parametrize("topic", [MULT, PERM, COMB])
def test_deterministic(topic):
    for seed in (0, 1, 7, 99):
        assert _exam(topic, seed).to_json() == _exam(topic, seed).to_json()


def test_every_multiplication_kind_and_combination_kind_occurs():
    mult_kinds = {("boys" in q.text, "at least one" in q.text) for q in (_exam(MULT, s) for s in range(60))}
    assert mult_kinds == {(True, False), (False, True), (False, False)}
    comb_kinds = {("points" in q.text, "men" in q.text) for q in (_exam(COMB, s) for s in range(60))}
    assert comb_kinds == {(True, False), (False, True), (False, False)}
