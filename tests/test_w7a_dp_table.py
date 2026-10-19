"""Tests `uc2-dynamic-programming-tables` (topic `w7a_dp_table`) and its exam chain `exd_x_w7a_dp_table`.

The oracles read each problem back out of the question TEXT and solve it with a different method from the generator's
tables: top-down memoised recursion, or exhaustive enumeration of routes, subsets and multisets of coins.
"""

from __future__ import annotations

import re
from functools import lru_cache
from itertools import product

import pytest

import mathgen.generators.w7a_dp_table_build  # noqa: F401  (registers the topics)
from tests import w7a_support as sup

TOPIC = "w7a_dp_table"
EXAM = "exd_x_w7a_dp_table"
SKILL = sup.skill_of("uc2-dynamic-programming-tables")


def _stairs(text: str) -> int:
    tolls = [int(c) for c in re.findall(r"step \d+: \$(\d+)", text)]
    n = len(tolls)
    moves = (1, 2, 3) if "one, two or three" in text else (1, 2)

    @lru_cache(None)
    def go(at):
        if at == n:
            return 0
        options = [tolls[at + s - 1] + go(at + s) for s in moves if at + s <= n]
        return min(options)

    return go(0)


def _amount(text: str) -> int:
    return int(re.findall(r"(\d+) cents", text)[-1])


def _fewest(coins, amount):
    @lru_cache(None)
    def go(a):
        if a == 0:
            return 0
        return min((go(a - c) + 1 for c in coins if c <= a), default=10**9)

    return go(amount)


def _route(text: str) -> int:
    arcs = {(u, v): int(c) for u, v, c in re.findall(r"([A-HST])→([A-HST]) (\d+)", text)}
    stages = [s.split(": ")[1].split(" or ") for s in re.findall(r"stage \d+: [A-H](?: or [A-H])*", text)]
    layers = [["S"], *stages, ["T"]]
    return min(sum(arcs[(r[i], r[i + 1])] for i in range(len(r) - 1)) for r in product(*layers))


def _items(text: str):
    return [(int(w), int(v)) for w, v in re.findall(r"weighs (\d+) kg and is worth \$(\d+)", text)]


def _cap(text: str) -> int:
    return int(re.search(r"at most (\d+) kg", text).group(1))


def _best_subset(items, cap):
    best = 0
    for mask in range(2 ** len(items)):
        w = sum(items[i][0] for i in range(len(items)) if mask >> i & 1)
        if w <= cap:
            best = max(best, sum(items[i][1] for i in range(len(items)) if mask >> i & 1))
    return best


def _unbounded(items, cap):
    @lru_cache(None)
    def go(room):
        return max([0] + [v + go(room - w) for w, v in items if w <= room])

    return go(cap)


def _ways(coins, amount) -> int:
    ranges = [range(amount // c + 1) for c in coins]
    return sum(1 for ks in product(*ranges) if sum(k * c for k, c in zip(ks, coins)) == amount)


def _lcs(a: str, b: str) -> int:
    @lru_cache(None)
    def go(i, j):
        if i == len(a) or j == len(b):
            return 0
        if a[i] == b[j]:
            return 1 + go(i + 1, j + 1)
        return max(go(i + 1, j), go(i, j + 1))

    return go(0, 0)


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM, "number")
    assert "dp-coin-greedy" in {m.id for m in SKILL.misconceptions}


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(tier):
    sup.check_single_tier(SKILL, TOPIC, tier, seeds=100)


def test_pools_and_shapes():
    sup.check_pools(TOPIC)


def _check_easy_medium(tier, kinds_wanted):
    kinds = set()
    for seed in range(200):
        q = sup.single(TOPIC, tier, seed)
        t = q.text
        if "toll" in t:
            assert _stairs(t) == int(q.answer), (seed, t)
            kinds.add("stairs")
        elif "fewest coins" in t:
            coins = [int(c) for c in re.findall(r"(\d+)c", t)]
            assert _fewest(coins, _amount(t)) == int(q.answer), (seed, t)
            greedy = 0
            left = _amount(t)
            for c in sorted(coins, reverse=True):
                greedy += left // c
                left %= c
            assert left == 0 and greedy > int(q.answer), "the largest-coin-first rule must fail"
            kinds.add("coins")
        elif "delivery" in t:
            assert _route(t) == int(q.answer), (seed, t)
            kinds.add("route")
        else:
            items, cap = _items(t), _cap(t)
            assert _best_subset(items, cap) == int(q.answer), (seed, t)
            kinds.add("knapsack")
    assert kinds == kinds_wanted


def test_easy_keys_by_independent_search():
    _check_easy_medium("easy", {"stairs", "coins", "route"})


def test_medium_keys_by_independent_search():
    _check_easy_medium("medium", {"knapsack", "coins", "route", "stairs"})


def test_hard_keys_by_independent_search():
    kinds = set()
    for seed in range(200):
        q = sup.single(TOPIC, "hard", seed)
        t = q.text
        if "subsequence" in t:
            a, b = re.findall(r"\b[ACGT]{8,9}\b", t)
            assert _lcs(a, b) == int(q.answer), (seed, t)
            kinds.add("lcs")
        elif "unlimited-copies" in t:
            items, cap = _items(t), _cap(t)
            assert _unbounded(items, cap) - _best_subset(items, cap) == int(q.answer), (seed, t)
            kinds.add("reuse")
        else:
            coins = [int(c) for c in re.findall(r"(\d+)c", t)]
            assert _ways(coins, _amount(t)) == int(q.answer), (seed, t)
            kinds.add("ways")
    assert kinds == {"lcs", "reuse", "ways"}


def test_exam_chain_shape():
    sup.check_exam_chain(SKILL, EXAM, [1, 2, 1, 2], ["number"] * 4)


def test_exam_keys_by_independent_search():
    for seed in range(100):
        q = sup.exam(EXAM, seed)
        items, cap = _items(q.stem), _cap(q.stem)
        a, b, c, d = q.parts
        assert int(a.answer) == _best_subset(items[:2], cap)
        assert int(b.answer) == _best_subset(items, cap)
        winners = [
            m
            for m in range(16)
            if sum(items[i][0] for i in range(4) if m >> i & 1) <= cap
            and sum(items[i][1] for i in range(4) if m >> i & 1) == int(b.answer)
        ]
        assert len(winners) == 1, "the optimal selection must be unique"
        assert int(c.answer) == sum(items[i][0] for i in range(4) if winners[0] >> i & 1)
        assert int(d.answer) == _best_subset(items, cap + 1) - int(b.answer)
        assert d.follow_through == f"{_best_subset(items, cap + 1)} - part_b"


def test_exam_follow_through_from_a_wrong_best_value():
    def wrong(q, canon):
        items, cap = _items(q.stem), _cap(q.stem)
        nxt = _best_subset(items, cap + 1)
        b = int(q.parts[1].answer)
        wb = b - 1
        if nxt - wb <= 0:
            return None
        return {"b": str(wb), "d": str(nxt - wb)}

    assert sup.check_follow_through(SKILL, EXAM, "d", wrong) >= 30


def test_answers_are_not_guessable():
    for tier in sup.TIERS:
        spread = sup.answer_spread(TOPIC, tier)
        assert len(spread) >= 10, (tier, spread)
        assert max(spread.values()) <= 14, (tier, spread.most_common(3))
