"""Tests `uc2-transportation-assignment` (topic `r3l_transportation_assignment`) and its exam chain.

The oracle reads the data out of the question TEXT and re-derives each answer another way: the north-west corner by
walking the supplies and demands, the assignment optimum by trying all 24 permutations, and the improvement index by
re-solving the balanced problem with one tonne forced onto the route (a small transportation LP through SymPy-free
enumeration is avoided; instead the change in cost is computed from the loop and checked against the potentials).
"""

from __future__ import annotations

import re
from itertools import permutations

import mathgen.generators.r3l_transportation_assignment_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import load_skills
from tests.r3l_support import check_follow_through, check_parts, check_single, parse_matrix, plain

TOPIC = "r3l_transportation_assignment"
EXAM = "exd_x_r3l_transportation_assignment"
SKILL_ID = "uc2-transportation-assignment"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _data(text):
    body = plain(text)
    sup = [
        int(v)
        for v in re.search(r"Supplies \(tonnes\): ([^.]*)\.", body).group(1).replace(",", "").split()
        if v.isdigit()
    ]
    dem = [
        int(v)
        for v in re.search(r"Demands \(tonnes\): ([^.]*)\.", body).group(1).replace(",", "").split()
        if v.isdigit()
    ]
    cost = parse_matrix(body.split("is [[", 1)[0] + "[[" + body.split("is [[", 1)[1])
    return sup, dem, [list(map(int, cost.row(i))) for i in range(cost.rows)]


def _nw(sup, dem):
    sup, dem = list(sup), list(dem)
    alloc = [[0] * len(dem) for _ in sup]
    i = j = 0
    while i < len(sup) and j < len(dem):
        t = min(sup[i], dem[j])
        alloc[i][j] = t
        sup[i] -= t
        dem[j] -= t
        if sup[i] == 0 and i < len(sup) - 1:
            i += 1
        else:
            j += 1
    return alloc


def test_skill_metadata():
    assert list(_SKILL.year_level) == ["university"]
    assert "university-year-2" in _SKILL.tags
    assert all(p for p in _SKILL.prerequisites)


def test_easy_north_west_cost():
    for seed in range(40):
        q = _q("easy", seed)
        sup, dem, cost = _data(q.text)
        alloc = _nw(sup, dem)
        assert int(q.answer_text) == sum(alloc[i][j] * cost[i][j] for i in range(len(sup)) for j in range(len(dem)))
        check_single(q, "number")


def test_medium_assignment_optimum():
    for seed in range(40):
        q = _q("medium", seed)
        cost = parse_matrix(plain(q.text))
        best = min(sum(cost[i, p[i]] for i in range(4)) for p in permutations(range(4)))
        assert int(q.answer_text) == best
        check_single(q, "number")


def test_hard_improvement_index():
    for seed in range(40):
        q = _q("hard", seed)
        sup, dem, cost = _data(q.text)
        alloc = _nw(sup, dem)
        m = re.search(r"route from (.+?) to (.+?):", plain(q.text))
        rows = re.findall(
            r"([A-Z][A-Za-z ]+?) (?:\d+)(?:,|\.)",
            re.search(r"Supplies \(tonnes\): ([^.]*)\.", plain(q.text)).group(1) + ".",
        )
        cols = re.findall(
            r"([A-Z][A-Za-z ]+?) (?:\d+)(?:,|\.)",
            re.search(r"Demands \(tonnes\): ([^.]*)\.", plain(q.text)).group(1) + ".",
        )
        i0, j0 = rows.index(m.group(1)), cols.index(m.group(2))
        # potentials from the basic cells (u0 = 0), then index = c - u - v
        basic = [(i, j) for i in range(len(sup)) for j in range(len(dem)) if alloc[i][j] > 0]
        u, v = {0: 0}, {}
        for _ in range(len(sup) + len(dem)):
            for i, j in basic:
                if i in u and j not in v:
                    v[j] = cost[i][j] - u[i]
                if j in v and i not in u:
                    u[i] = cost[i][j] - v[j]
        assert int(q.answer_text) == cost[i0][j0] - u[i0] - v[j0]
        check_single(q, "number")


def test_exam_chain():
    for seed in range(25):
        q = make(EXAM, seed=seed, difficulty="exam")
        check_parts(q)
        sup, dem, cost = _data(q.text)
        alloc = _nw(sup, dem)
        assert parse_matrix(q.parts[0].answer_text).tolist() == alloc
        total = sum(alloc[i][j] * cost[i][j] for i in range(len(sup)) for j in range(len(dem)))
        assert int(q.parts[1].answer_text) == total
        check_follow_through(q, "c", int(q.parts[2].answer_text) - 1)
