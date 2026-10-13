"""Tests `graph-eulerian-trails-circuits` (topic `snet_eulerian`) and its exam chain `snet_eulerian_chain`.

The oracles read each network back out of the question TEXT (or out of the answer options, for the hard tier) and decide
it from the degrees and connectedness, and by a backtracking search of their own that tries to use every edge once.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

import pytest
from snet_support import (
    LETTERS,
    check_graded,
    check_text,
    connected,
    degrees,
    edges_from_list,
    network_from_text,
    skill,
)

from mathgen import make
from mathgen.engine import grade

SK = skill("graph-eulerian-trails-circuits")
TIERS = ("easy", "medium", "hard")
CIRCUIT = "It has an Eulerian circuit."
TRAIL = "It has an Eulerian trail but no Eulerian circuit."
NEITHER = "It has neither an Eulerian trail nor an Eulerian circuit."


def _q(tier, seed):
    return make("snet_eulerian", seed=seed, difficulty=tier)


def _exam(seed):
    return make("snet_eulerian_chain", seed=seed, difficulty="exam")


def euler_by_search(n, edges):
    """(some Eulerian trail exists, an Eulerian circuit exists, set of start vertices), trying to use every edge once."""
    edges = list(edges)
    starts, circuit = set(), False

    def walk(at, used, first):
        nonlocal circuit
        if len(used) == len(edges):
            starts.add(first)
            circuit = circuit or at == first
            return
        for i, (u, v) in enumerate(edges):
            if i not in used and at in (u, v):
                walk(v if at == u else u, used | {i}, first)

    for s in range(n):
        walk(s, frozenset(), s)
    return bool(starts), circuit, starts


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(80):
        check_graded(SK, _q(tier, seed))


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        check_text(_q(tier, seed))


def test_easy_key_matches_degrees_and_a_search():
    seen = Counter()
    for seed in range(200):
        q = _q("easy", seed)
        n, edges = network_from_text(q.text)
        assert connected(n, edges)
        odd = sum(d % 2 for d in degrees(n, edges))
        key = {0: CIRCUIT, 2: TRAIL}.get(odd, NEITHER)
        assert q.answer_text == key and sorted(q.distractors) == sorted({CIRCUIT, TRAIL, NEITHER} - {key})
        trail, circuit, _starts = euler_by_search(n, edges)
        assert (trail, circuit) == {CIRCUIT: (True, True), TRAIL: (True, False), NEITHER: (False, False)}[key]
        seen[key] += 1
    assert min(seen.values()) >= 40, seen


def test_medium_key_is_the_pair_of_odd_vertices_and_the_search_agrees():
    for seed in range(150):
        q = _q("medium", seed)
        n, edges = network_from_text(q.text)
        odd = [LETTERS[v] for v, d in enumerate(degrees(n, edges)) if d % 2]
        assert len(odd) == 2 and q.answer_text == f"{odd[0]} and {odd[1]}"
        _t, circuit, starts = euler_by_search(n, edges)
        assert not circuit and {LETTERS[s] for s in starts} == set(odd)
        options = [q.answer_text, *q.distractors]
        assert len(set(options)) == 4


def test_hard_key_is_the_only_network_with_an_eulerian_circuit():
    for seed in range(200):
        q = _q("hard", seed)
        listed = dict(re.findall(r"(Network [PQRS]): ((?:[A-H]{2}(?:, )?)+)\.", q.text))
        assert set(listed) == {"Network P", "Network Q", "Network R", "Network S"}
        assert sorted([q.answer_text, *q.distractors]) == sorted(listed)
        verdicts = {}
        for label, text in listed.items():
            edges = edges_from_list(text)
            n = max(max(e) for e in edges) + 1
            all_even = all(d % 2 == 0 for d in degrees(n, edges))
            whole = connected(n, edges)
            _t, circuit, _s = euler_by_search(n, edges)
            assert circuit == (all_even and whole)
            verdicts[label] = (circuit, all_even, whole, sum(d % 2 for d in degrees(n, edges)))
        assert [t for t, v in verdicts.items() if v[0]] == [q.answer_text], (seed, verdicts)
        shapes = sorted((v[1], v[2], v[3]) for t, v in verdicts.items() if not v[0])
        assert shapes == [(False, True, 2), (False, True, 4), (True, False, 0)], (seed, shapes)
    assert len({_q("hard", s).answer_text for s in range(100)}) == 4  # the key sits under every label


def test_tiers_differ_in_shape():
    easy = {_q("easy", s).answer_text for s in range(100)}
    medium = [_q("medium", s).answer_text for s in range(30)]
    hard = [_q("hard", s).answer_text for s in range(30)]
    assert easy == {CIRCUIT, TRAIL, NEITHER}
    assert all(" and " in a and len(a) == 7 for a in medium)
    assert all(a.startswith("Network ") for a in hard)


# -- exam chain ---------------------------------------------------------------------------------------------------


def min_added(n, edges, goal_odd):
    present = set(edges)
    absent = [p for p in combinations(range(n), 2) if p not in present]
    for k in range(5):
        for extra in combinations(absent, k):
            grown = sorted(present | set(extra))
            if sum(d % 2 for d in degrees(n, grown)) == goal_odd and connected(n, grown):
                return k
    return None


def test_exam_shape():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 1, 2]
        m = int(q.parts[2].answer) - int(q.parts[1].answer)
        assert [p.follow_through for p in q.parts] == [None, "part_a/2", f"{m} + part_b", "Abs(part_a/2 - 1)"]
        assert all(p.answer_kind == "number" and p.sensible_range for p in q.parts)
        check_text(q)


def test_exam_keys_match_an_independent_derivation():
    for seed in range(120):
        q = _exam(seed)
        n, edges = network_from_text(q.stem)
        assert connected(n, edges)
        odd = sum(d % 2 for d in degrees(n, edges))
        circuit_k, trail_k = min_added(n, edges, 0), min_added(n, edges, 2)
        assert [p.answer for p in q.parts] == [odd, circuit_k, len(edges) + circuit_k, trail_k]
        assert odd in (0, 2, 4, 6)


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(80):
        check_graded(SK, _exam(seed))


def test_exam_follow_through_from_wrong_earlier_parts():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        m = int(canon["c"]) - int(canon["b"])
        # (a) wrongly a different even count of odd vertices: (b), (c) and (d) follow through from that.
        wrong_a = 2 if int(canon["a"]) != 2 else 4
        wrong_b = wrong_a // 2
        result = grade(
            SK,
            q,
            {**canon, "a": str(wrong_a), "b": str(wrong_b), "c": str(m + wrong_b), "d": str(abs(wrong_b - 1))},
        )
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 2 and result.parts["b"].follow_through is True, (seed, result)
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True, (seed, result)
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True, (seed, result)
        checked += 1
    assert checked == 60


def test_exam_odd_counts_vary():
    assert {_exam(s).parts[0].answer for s in range(80)} == {0, 2, 4, 6}


def test_exam_answer_space():
    assert len({_exam(s).text for s in range(300)}) >= 295
    assert len({_exam(s).parts[2].answer for s in range(300)}) >= 5
