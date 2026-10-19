"""Tests `network-cut-capacity-max-flow` (topic `w7a_network_cut`) and its exam chain `exd_x_w7a_network_cut`.

The oracles read the network and the cuts back out of the question TEXT, then recompute cut capacities by summing the
links from the source side to the sink side, and the maximum flow by a depth-first Ford-Fulkerson search.
"""

from __future__ import annotations

import re
from itertools import combinations

import pytest

import mathgen.generators.w7a_network_cut_build  # noqa: F401  (registers the topics)
from tests import w7a_support as sup

TOPIC = "w7a_network_cut"
EXAM = "exd_x_w7a_network_cut"
SKILL = sup.skill_of("network-cut-capacity-max-flow")


def _arcs(text: str) -> dict:
    body = re.search(r"capacities, in [^,]+?, are (.+?), where X→Y", text).group(1)
    return {(u, v): int(c) for u, v, c in re.findall(r"([A-DST])→([A-DST]) (\d+)", body)}


def _sides(text: str):
    out = []
    for src in re.findall(r"puts ([A-DST, and]+?) on the source side", text):
        out.append(set(re.findall(r"[A-DST]", src.replace("and", ","))))
    return out


def _cut(arcs, side) -> int:
    return sum(c for (u, v), c in arcs.items() if u in side and v not in side)


def _ford_fulkerson(arcs) -> int:
    res = {}
    for (u, v), c in arcs.items():
        res[(u, v)] = res.get((u, v), 0) + c
        res.setdefault((v, u), 0)
    nodes = {x for e in res for x in e}

    def dfs(u, seen, limit):
        if u == "T":
            return limit
        seen.add(u)
        for v in sorted(nodes):
            if v not in seen and res.get((u, v), 0) > 0:
                got = dfs(v, seen, min(limit, res[(u, v)]))
                if got:
                    res[(u, v)] -= got
                    res[(v, u)] += got
                    return got
        return 0

    total = 0
    while True:
        got = dfs("S", set(), 10**6)
        if not got:
            return total
        total += got


def _smallest_cut(arcs) -> int:
    mids = sorted({x for e in arcs for x in e} - {"S", "T"})
    return min(_cut(arcs, {"S", *chosen}) for k in range(len(mids) + 1) for chosen in combinations(mids, k))


def test_wiring():
    sup.check_wiring(SKILL, TOPIC, EXAM, "number")
    assert "cut-counts-backward-pipes" in {m.id for m in SKILL.misconceptions}


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(tier):
    sup.check_single_tier(SKILL, TOPIC, tier, seeds=100)


def test_pools_and_shapes():
    sup.check_pools(TOPIC)


def test_easy_keys_by_independent_summing():
    for seed in range(120):
        q = sup.single(TOPIC, "easy", seed)
        arcs, (side,) = _arcs(q.text), _sides(q.text)
        assert _cut(arcs, side) == int(q.answer), (seed, q.text)
        assert not any(u not in side and v in side for u, v in arcs), "easy cuts have no return link"


def test_medium_keys():
    seen = set()
    for seed in range(200):
        q = sup.single(TOPIC, "medium", seed)
        arcs, sides = _arcs(q.text), _sides(q.text)
        if len(sides) == 1:
            assert _cut(arcs, sides[0]) == int(q.answer), (seed, q.text)
            assert any(u not in sides[0] and v in sides[0] for u, v in arcs), "medium cuts have a return link"
            seen.add("single")
        else:
            assert len(sides) == 3
            caps = sorted(_cut(arcs, s) for s in sides)
            assert int(q.answer) == caps[0] == _ford_fulkerson(arcs) == _smallest_cut(arcs), (seed, q.text)
            assert caps[0] < caps[1] < caps[2] or caps[0] < caps[1] <= caps[2]
            seen.add("three")
    assert seen == {"single", "three"}


def test_hard_keys():
    seen = set()
    for seed in range(150):
        q = sup.single(TOPIC, "hard", seed)
        arcs = _arcs(q.text)
        widened = re.search(r"The capacity of the \w+ ([A-DS])→([A-DT]) is increased by (\d+)", q.text)
        if widened:
            u, v, k = widened.group(1), widened.group(2), int(widened.group(3))
            wider = dict(arcs)
            wider[(u, v)] += k
            old, new = _ford_fulkerson(arcs), _ford_fulkerson(wider)
            assert new == int(q.answer) == _smallest_cut(wider), (seed, q.text)
            assert old < new < old + k, "a widening that helps, but only partly"
            seen.add("widen")
        else:
            flow = _ford_fulkerson(arcs)
            assert flow == int(q.answer) == _smallest_cut(arcs), (seed, q.text)
            out_s = sum(c for (u, _), c in arcs.items() if u == "S")
            in_t = sum(c for (_, v), c in arcs.items() if v == "T")
            assert flow < min(out_s, in_t)
            seen.add("flow")
    assert seen == {"widen", "flow"}


def test_exam_chain_shape():
    sup.check_exam_chain(SKILL, EXAM, [1, 1, 2, 1], ["number"] * 4)


def test_exam_keys_by_independent_computation():
    for seed in range(100):
        q = sup.exam(EXAM, seed)
        arcs, (s1, s2) = _arcs(q.stem), _sides(q.stem)
        a, b, c, d = q.parts
        assert int(a.answer) == _cut(arcs, s1)
        assert int(b.answer) == _cut(arcs, s2)
        assert any(u not in s2 and v in s2 for u, v in arcs), "cut 2 has a return link"
        flow = _ford_fulkerson(arcs)
        assert int(c.answer) == flow == _smallest_cut(arcs)
        assert flow < min(int(a.answer), int(b.answer)) or flow <= min(int(a.answer), int(b.answer))
        out_s = sum(cap for (u, _), cap in arcs.items() if u == "S")
        assert int(d.answer) == out_s - flow
        assert d.follow_through == f"{out_s} - part_c"


def test_exam_follow_through_from_a_wrong_flow():
    def wrong(q, canon):
        out_s = sum(cap for (u, _), cap in _arcs(q.stem).items() if u == "S")
        c = int(q.parts[2].answer)
        if c + 1 >= out_s:
            return None
        return {"c": str(c + 1), "d": str(out_s - c - 1)}

    assert sup.check_follow_through(SKILL, EXAM, "d", wrong) >= 30


def test_answers_are_not_guessable():
    for tier in sup.TIERS:
        spread = sup.answer_spread(TOPIC, tier)
        assert len(spread) >= 10, (tier, spread)
        assert max(spread.values()) <= 12, (tier, spread.most_common(3))
