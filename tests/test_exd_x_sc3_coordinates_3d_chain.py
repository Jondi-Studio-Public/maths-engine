"""Exam tier of `f10-three-dimensional-coordinates`: displacement over two legs, then the straight-line distance."""

from __future__ import annotations

import math
import re

import exd_x_sc3_support as sup

SKILL = "f10-three-dimensional-coordinates"
TOPIC = "exd_x_sc3_coordinates_3d_chain"
SEEDS = range(300)

_STEM = re.compile(
    r"starts at A\((-?\d+), (-?\d+), (-?\d+)\), flies to B\((-?\d+), (-?\d+), (-?\d+)\) and then flies a further "
    r"(\d+) m (east|west), (\d+) m (north|south) and (\d+) m (up|down) to reach C\."
)
_SIGN = {"east": 1, "west": -1, "north": 1, "south": -1, "up": 1, "down": -1}


def points(q):
    m = _STEM.search(q.stem)
    assert m, q.stem
    g = m.groups()
    a, b = tuple(int(v) for v in g[0:3]), tuple(int(v) for v in g[3:6])
    move = (int(g[6]) * _SIGN[g[7]], int(g[8]) * _SIGN[g[9]], int(g[10]) * _SIGN[g[11]])
    c = tuple(bi + mi for bi, mi in zip(b, move))
    return a, b, c


def oracle(q) -> dict[str, str]:
    a, _, c = points(q)
    total = [ci - ai for ai, ci in zip(a, c)]
    return {
        "a": str(total[0]),
        "b": str(total[1]),
        "c": str(total[2]),
        "d": str(math.isqrt(sum(t * t for t in total))),
    }


def test_registered():
    sup.assert_registered(SKILL, TOPIC)


def test_text_and_determinism():
    sup.assert_clean_and_deterministic(TOPIC)


def test_shapes():
    sup.assert_shapes(SKILL, TOPIC)


def test_key_rederived_from_prompt():
    for seed in SEEDS:
        q = sup.exam(TOPIC, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        want = oracle(q)
        assert want == sup.canonical(q), (seed, q.text)
        assert int(want["d"]) ** 2 == sum(int(want[k]) ** 2 for k in "abc"), "the distance is not a whole number"


def test_every_point_is_within_drone_height_limits():
    for seed in SEEDS:
        for point in points(sup.exam(TOPIC, seed)):
            assert 5 <= point[2] <= 120, (seed, point)


def test_key_accepted_and_distractors_rejected():
    sup.assert_key_accepted_distractors_rejected(SKILL, TOPIC, range(150))


def test_answer_space():
    sup.assert_answer_space(TOPIC)


def test_follow_through_from_wrong_components():
    for seed in SEEDS:
        q = sup.exam(TOPIC, seed)
        wrong = [int(p.answer_text) + 100 for p in q.parts[:3]]  # far from the key, and never zero
        dist = math.sqrt(sum(w * w for w in wrong))
        sup.assert_follow_through(
            SKILL,
            q,
            {"a": str(wrong[0]), "b": str(wrong[1]), "c": str(wrong[2])},
            {"d": sup.typed(round(dist * 1000) / 1000, 0)},
        )


def test_one_wrong_component_still_earns_the_distance():
    for seed in range(60):
        q = sup.exam(TOPIC, seed)
        comps = [int(p.answer_text) for p in q.parts[:3]]
        comps[0] += 3
        if 0 in comps:
            continue
        dist = math.sqrt(sum(c * c for c in comps))
        if sup.typed(round(dist * 1000) / 1000, 0) == q.parts[3].answer_text:
            continue  # the slip rounds to the key, so the reply is simply right
        result = sup.grade(
            sup.skill(SKILL),
            q,
            {"a": str(comps[0]), "b": str(comps[1]), "c": str(comps[2]), "d": sup.typed(round(dist * 1000) / 1000, 0)},
        )
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == 1 and result.parts["c"].marks == 1
        assert result.parts["d"].marks == 2 and result.parts["d"].follow_through is True
