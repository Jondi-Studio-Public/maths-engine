"""Tests `uc2-lp-formulation-and-corner-points` (topic `r3l_lp_corner_points`) and its exam chain.

The oracle reads the story's coefficients back out of the question TEXT and re-solves the linear programme by scanning
every integer point of a box (all corner points are integral, so the best lattice point is the optimum), with SymPy
solving the corner points of the exam chain.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter

import sympy

import mathgen.generators.r3l_lp_corner_points_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import load_skills
from tests.r3l_support import check_follow_through, check_parts, check_single

TOPIC = "r3l_lp_corner_points"
EXAM = "exd_x_r3l_lp_corner_points"
SKILL_ID = "uc2-lp-formulation-and-corner-points"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)

_MAX_LINE = re.compile(r"takes (\d+) [a-z ]+? and each .*? takes (\d+) [a-z ]+?, with (\d+) [a-z ]+? available")
_MIN_LINE = re.compile(r"gives (\d+) [a-z]+ and each .*? gives (\d+) [a-z]+; at least (\d+) [a-z]+ are needed")
_PROFIT = re.compile(r"\$(\d+) on each .*? and \$(\d+|k) on each")
_COST = re.compile(r"costs \$(\d+) and a .*? costs \$(\d+)")


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _lattice(cons, coefs, sense, box):
    best, where = None, []
    for x, y in itertools.product(range(box + 1), repeat=2):
        ok = all((a * x + b * y <= r) if s == "<=" else (a * x + b * y >= r) for a, b, r, s in cons)
        if not ok:
            continue
        v = coefs[0] * x + coefs[1] * y
        if best is None or (v > best if sense == "max" else v < best):
            best, where = v, [(x, y)]
        elif v == best:
            where.append((x, y))
    return best, where


def _cons_max(text):
    return [(int(a), int(b), int(r), "<=") for a, b, r in _MAX_LINE.findall(text)]


def test_skill_wiring():
    assert _SKILL.answer == "number"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert _SKILL.sources[-1].stages == ["exam"]
    assert "university-year-2" in _SKILL.tags and list(_SKILL.year_level) == ["university"]


def test_easy_key_is_the_best_lattice_point():
    shapes = Counter()
    for seed in range(60):
        q = _q("easy", seed)
        cons = _cons_max(q.text)
        p, qq = map(int, _PROFIT.search(q.text).groups())
        best, where = _lattice(cons, (p, qq), "max", 45)
        assert len(cons) == 2 and best == q.answer, (seed, q.text)
        assert len(where) == 1
        shapes[where[0]] += 1
        check_single(q, "number")
    assert len(shapes) >= 10


def test_medium_key_is_the_least_cost():
    for seed in range(60):
        q = _q("medium", seed)
        cons = [(int(a), int(b), int(r), ">=") for a, b, r in _MIN_LINE.findall(q.text)]
        p, qq = map(int, _COST.search(q.text).groups())
        best, where = _lattice(cons, (p, qq), "min", 80)
        assert len(cons) == 3 and best == q.answer and len(where) == 1, (seed, q.text)
        check_single(q, "number")


def test_hard_key_makes_the_whole_edge_optimal():
    named = Counter()
    for seed in range(60):
        q = _q("hard", seed)
        cons = _cons_max(q.text)
        p = int(_PROFIT.search(q.text).group(1))
        k = int(q.answer)
        resource = re.search(r"on the ([a-z ]+?) limit", q.text).group(1)
        order = re.findall(r"([A-Z][a-z]+(?: [a-z]+)?): each", q.text)
        i = [o.lower() for o in order].index(resource)
        a, b, r, _ = cons[i]
        best, where = _lattice(cons, (p, k), "max", 45)
        edge = [
            pt
            for pt in itertools.product(range(46), repeat=2)
            if a * pt[0] + b * pt[1] == r and all(c * pt[0] + d * pt[1] <= rr for c, d, rr, _ in cons)
        ]
        assert set(where) == set(edge) and len(edge) >= 2, (seed, q.text)
        for dk in (-1, 1):
            _, w2 = _lattice(cons, (p, k + dk), "max", 45)
            assert set(w2) != set(edge), (seed, dk)  # off the critical k the named edge is not the optimal face
        named[resource] += 1
        check_single(q, "number")
    assert len(named) >= 6


def test_exam_chain():
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        cons = _cons_max(q.text)
        p, qq = map(int, _PROFIT.search(q.text).groups())
        x, y = sympy.symbols("x y")
        lines = [sympy.Eq(a * x + b * y, r) for a, b, r, _ in cons] + [sympy.Eq(x, 0), sympy.Eq(y, 0)]
        corners = set()
        for l1, l2 in itertools.combinations(lines, 2):
            sol = sympy.solve([l1, l2], [x, y], dict=True)
            if sol and len(sol[0]) == 2:
                px, py = sol[0][x], sol[0][y]
                if px >= 0 and py >= 0 and all(a * px + b * py <= r for a, b, r, _ in cons):
                    corners.add((int(px), int(py)))
        pa, pb, pc, pd = q.parts
        assert {tuple(map(int, pt)) for pt in pa.answer} == corners, (seed, q.text)
        best = max(corners, key=lambda c: p * c[0] + qq * c[1])
        assert (pb.answer, pc.answer, pd.answer) == (best[0], best[1], p * best[0] + qq * best[1])
        assert best[0] > 0 and best[1] > 0
        assert [pt.answer_kind for pt in q.parts] == ["points", "number", "number", "number"]
        assert [pt.marks for pt in q.parts] == [3, 2, 1, 2]
        check_parts(q)


def test_exam_follow_through():
    for seed in range(15):
        q = make(EXAM, seed=seed, difficulty="exam")
        bx, by = q.parts[1].answer, q.parts[2].answer
        check_follow_through(q, "b", bx + 1)
        check_follow_through(q, "c", by + 2)
