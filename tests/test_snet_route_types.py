"""Tests `graph-walks-trails-paths-cycles` (topic `snet_route_types`) and its exam chain `snet_route_types_chain`.

The oracle reads the network and the route back out of the question TEXT and names the route from the definitions:
closed or open, then a repeated edge, then a repeated vertex.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
from snet_support import LETTERS, check_graded, check_text, network_from_text, skill

from mathgen import make
from mathgen.engine import grade

SK = skill("graph-walks-trails-paths-cycles")
TIERS = ("easy", "medium", "hard")
NAMES = {
    "walk": "Walk",
    "trail": "Trail",
    "path": "Path",
    "closed walk": "Closed walk",
    "closed trail": "Closed trail (circuit)",
    "cycle": "Cycle",
}
_ROUTE = re.compile(r"\b([A-H](?:-[A-H])+)\b")


def _q(tier, seed):
    return make("snet_route_types", seed=seed, difficulty=tier)


def _exam(seed):
    return make("snet_route_types_chain", seed=seed, difficulty="exam")


def name_of(route, edges):
    """The most specific name, from the definitions, after checking the route really is a walk of the network."""
    r = [LETTERS.index(c) for c in route.split("-")]
    steps = [tuple(sorted(p)) for p in zip(r, r[1:])]
    assert all(s in edges for s in steps), (route, "is not a walk of the network")
    closed = r[0] == r[-1]
    no_repeat_edge = len(set(steps)) == len(steps)
    interior = r[:-1] if closed else r
    no_repeat_vertex = len(set(interior)) == len(interior)
    if closed:
        return "closed walk" if not no_repeat_edge else ("cycle" if no_repeat_vertex else "closed trail")
    return "walk" if not no_repeat_edge else ("path" if no_repeat_vertex else "trail")


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_and_every_distractor_rejected(tier):
    for seed in range(80):
        check_graded(SK, _q(tier, seed))


@pytest.mark.parametrize("tier", TIERS)
def test_text_quality_and_determinism(tier):
    for seed in range(60):
        check_text(_q(tier, seed))


@pytest.mark.parametrize("tier", ("easy", "medium"))
def test_named_route_matches_an_independent_classification(tier):
    for seed in range(200):
        q = _q(tier, seed)
        _n, edges = network_from_text(q.text)
        route = _ROUTE.findall(q.text.split("joins:")[1].split(".", 1)[1])[0]
        assert q.answer_text == NAMES[name_of(route, edges)], (seed, route)
        assert len(q.distractors) == 5  # every other name is on offer


def test_hard_choice_is_the_only_route_of_the_stated_kind():
    wanted = {
        "a walk that is not a trail": "walk",
        "a trail that is not a path": "trail",
        "a path": "path",
        "a closed walk that is not a closed trail": "closed walk",
        "a closed trail that is not a cycle": "closed trail",
        "a cycle": "cycle",
    }
    seen = Counter()
    for seed in range(300):
        q = _q("hard", seed)
        _n, edges = network_from_text(q.text)
        target = next(v for k, v in wanted.items() if q.text.rstrip("?").endswith(k))
        options = [q.answer_text, *q.distractors]
        assert len(options) == 4 and len(set(options)) == 4
        kinds = {o: name_of(o, edges) for o in options}
        assert [o for o, k in kinds.items() if k == target] == [q.answer_text], (seed, kinds, target)
        assert len(set(kinds.values())) == 4  # four different kinds of route
        seen[target] += 1
    assert set(seen) == set(wanted.values()) and min(seen.values()) >= 20, seen


def test_tiers_differ_in_shape():
    easy = Counter(_q("easy", s).answer_text for s in range(200))
    medium = Counter(_q("medium", s).answer_text for s in range(200))
    hard = [_q("hard", s).answer_text for s in range(60)]
    assert set(easy) == {"Walk", "Trail", "Path"}  # open routes only
    assert set(medium) == set(NAMES.values())  # all six kinds, closed ones included
    assert all("-" in a for a in hard)  # the options are routes, not names


# -- exam chain ---------------------------------------------------------------------------------------------------


def test_exam_shape():
    for seed in range(60):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [1, 2, 1, 2]
        assert [p.answer_kind for p in q.parts] == ["number", "number", "number", "mcq"]
        assert [p.follow_through for p in q.parts] == [None, None, "part_a - part_b", None]
        check_text(q)


def test_exam_keys_match_an_independent_derivation():
    for seed in range(200):
        q = _exam(seed)
        _n, edges = network_from_text(q.stem)
        route = _ROUTE.findall(q.stem.split("joins:")[1].split(".", 1)[1])[0]
        r = [LETTERS.index(c) for c in route.split("-")]
        steps = [tuple(sorted(p)) for p in zip(r, r[1:])]
        a, b = len(steps), len(set(steps))
        assert [p.answer for p in q.parts[:3]] == [a, b, a - b]
        assert a > b  # the route does re-use an edge
        assert q.parts[3].answer_text == NAMES[name_of(route, edges)]
        assert name_of(route, edges) in ("walk", "closed walk")


def test_exam_key_accepted_and_every_distractor_rejected():
    for seed in range(80):
        check_graded(SK, _exam(seed))


def test_exam_follow_through_from_wrong_earlier_parts():
    checked = 0
    for seed in range(80):
        q = _exam(seed)
        canon = {p.label: p.answer_text for p in q.parts}
        a, b = int(canon["a"]), int(canon["b"])
        for wrong_a in (a + 1, a + 2):
            if wrong_a - b == wrong_a or wrong_a - b == b:
                continue
            result = grade(SK, q, {**canon, "a": str(wrong_a), "c": str(wrong_a - b)})
            assert result.parts["a"].marks == 0
            assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True, (seed, wrong_a, result)
            checked += 1
            break
    assert checked >= 70


def test_exam_answer_space():
    texts = {_exam(s).text for s in range(300)}
    assert len(texts) >= 295
    assert len({_exam(s).answer_text for s in range(300)}) >= 10
