"""Tests `graph-hamiltonian-paths-cycles` (topic `snet_hamiltonian`) and its exam chain `snet_hamiltonian_chain`.

The oracles read the network, the routes or the distances back out of the question TEXT and recount: each listed route
against the definition of a Hamiltonian cycle, every ordering of the vertices for the counts, and every tour for the
shortest distance and its fuel cost.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import permutations

import pytest
from snet_support import LETTERS, check_graded, check_text, ham_cycles, network_from_text, skill

from mathgen import make
from mathgen.engine import grade

SK = skill("graph-hamiltonian-paths-cycles")
TIERS = ("easy", "medium", "hard")


def _q(tier, seed):
    return make("snet_hamiltonian", seed=seed, difficulty=tier)


def _exam(seed):
    return make("snet_hamiltonian_chain", seed=seed, difficulty="exam")


def is_ham_cycle(route, n, edges):
    r = [LETTERS.index(c) for c in route.split("-")]
    es = set(edges)
    return (
        len(r) == n + 1
        and r[0] == r[-1]
        and sorted(r[:-1]) == list(range(n))
        and all(tuple(sorted(p)) in es for p in zip(r, r[1:]))
    )


def ham_paths_from(n, edges, s):
    es = set(edges)
    return [
        (s, *p)
        for p in permutations([v for v in range(n) if v != s])
        if all(tuple(sorted(pair)) in es for pair in zip((s, *p), p))
    ]


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(80):
        check_graded(SK, _q(tier, seed))


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        check_text(_q(tier, seed))


def test_easy_key_counts_the_listed_routes_that_are_hamiltonian_cycles():
    seen = Counter()
    for seed in range(300):
        q = _q("easy", seed)
        n, edges = network_from_text(q.text)
        listed = re.search(r"in order: ([A-H\-, ]+)\.", q.text).group(1).split(", ")
        assert len(listed) == 4 and len(set(listed)) == 4
        assert q.answer == sum(is_ham_cycle(r, n, edges) for r in listed)
        # No two of the genuine ones are the same cycle written differently.
        genuine = [r for r in listed if is_ham_cycle(r, n, edges)]
        seen[q.answer] += 1
        keys = {frozenset(frozenset(p) for p in zip(r.split("-"), r.split("-")[1:])) for r in genuine}
        assert len(keys) == len(genuine), (seed, genuine)
    assert set(seen) == {0, 1, 2, 3} and min(seen.values()) >= 15, seen


def test_medium_key_counts_the_hamiltonian_cycles():
    seen = Counter()
    for seed in range(200):
        q = _q("medium", seed)
        n, edges = network_from_text(q.text)
        assert q.answer == len(ham_cycles(n, edges))
        assert 1 <= q.answer <= 6
        seen[q.answer] += 1
        # Reversed and re-started cycles are not counted again: the ordered tours from A are twice the cycles.
        ordered = sum(
            all(tuple(sorted(p)) in set(edges) for p in zip((0, *r), (*r, 0))) for r in permutations(range(1, n))
        )
        assert ordered == 2 * q.answer
    assert len(seen) >= 3, seen


def test_hard_key_counts_the_hamiltonian_paths_from_the_named_vertex():
    for seed in range(200):
        q = _q("hard", seed)
        n, edges = network_from_text(q.text)
        s = LETTERS.index(re.search(r"start at ([A-H])\?", q.text).group(1))
        assert n == 6 and q.answer == len(ham_paths_from(n, edges, s))
        assert 2 <= q.answer <= 8


def test_tiers_differ_in_shape():
    easy = [_q("easy", s).text for s in range(40)]
    medium = [_q("medium", s).text for s in range(40)]
    hard = [_q("hard", s).text for s in range(40)]
    assert all("Four routes" in t for t in easy) and not any("Four routes" in t for t in medium + hard)
    assert all("Hamiltonian cycles does" in t or "Hamiltonian cycles" in t for t in medium)
    assert all("Hamiltonian paths start at" in t for t in hard)
    assert {"A to E" in t or "A to F" in t for t in medium} == {True}  # five or six vertices
    assert {"A to F" in t for t in hard} == {True}  # six vertices


# -- exam chain ---------------------------------------------------------------------------------------------------


def _distances(text):
    listed = re.search(r"kilometres are: ([A-H]{2} \d+(?:, [A-H]{2} \d+)*)\.", text).group(1)
    weight = {}
    for item in listed.split(", "):
        pair, km = item.split()
        weight[tuple(sorted((LETTERS.index(pair[0]), LETTERS.index(pair[1]))))] = int(km)
    n = LETTERS.index(re.search(r"\bA to ([A-H])\b", text).group(1)) + 1
    return n, weight


def test_exam_shape():
    for seed in range(60):
        q = _exam(seed)
        n, _w = _distances(q.stem)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 1, 2, 1]
        assert [p.follow_through for p in q.parts] == [None, f"part_a*{2 * n}", None, "0.15*part_c"]
        assert all(p.answer_kind == "number" and p.sensible_range for p in q.parts)
        check_text(q)


def test_exam_keys_match_an_independent_derivation():
    for seed in range(200):
        q = _exam(seed)
        n, weight = _distances(q.stem)
        tours = []
        for p in permutations(range(1, n)):
            tour = (0, *p)
            if all(tuple(sorted((tour[i], tour[(i + 1) % n]))) in weight for i in range(n)):
                tours.append(sum(weight[tuple(sorted((tour[i], tour[(i + 1) % n])))] for i in range(n)))
        h = len(tours) // 2  # every cycle appears once in each direction
        shortest = min(tours)
        assert tours.count(shortest) == 2  # one cheapest cycle, seen in both directions
        assert [p.answer for p in q.parts[:3]] == [h, 2 * n * h, shortest]
        assert abs(q.parts[3].answer - shortest * 0.15) < 1e-9
        assert q.parts[3].answer_text == f"{shortest * 0.15:.2f}"


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(80):
        check_graded(SK, _exam(seed))


def test_exam_follow_through_from_wrong_earlier_parts():
    checked = 0
    for seed in range(60):
        q = _exam(seed)
        n, _w = _distances(q.stem)
        canon = {p.label: p.answer_text for p in q.parts}
        h, shortest = int(canon["a"]), int(canon["c"])
        # (a) counted in both directions: (b) follows through; (c) wrong by one: (d) follows through.
        result = grade(SK, q, {**canon, "a": str(2 * h), "b": str(2 * n * 2 * h)})
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (seed, result)
        wrong_c = shortest + 7
        result = grade(SK, q, {**canon, "c": str(wrong_c), "d": f"{wrong_c * 0.15:.2f}"})
        assert result.parts["c"].marks == 0
        assert result.parts["d"].marks == 1 and result.parts["d"].follow_through is True, (seed, result)
        checked += 1
    assert checked == 60


def test_exam_answer_space():
    assert len({_exam(s).text for s in range(300)}) >= 295
    assert Counter(_exam(s).parts[0].answer for s in range(300)).keys() == {1, 2, 3, 4, 5, 6}
