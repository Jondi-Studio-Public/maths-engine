"""Tests `uc2-simplex-final-tableau-reading` (topic `r3l_simplex_final_tableau`) and its exam chain.

The oracle reads the final tableau and the named basic variables out of the question TEXT, checks that it really is an
optimal tableau (unit basic columns, non-negative right-hand sides and objective row), and re-derives each answer from
the definition of the slack column: basic values move as RHS + t * (slack column entry) when resource k changes by t.
"""

from __future__ import annotations

import re
from collections import Counter

import sympy

import mathgen.generators.r3l_simplex_final_tableau_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import load_skills
from tests.r3l_support import check_follow_through, check_parts, check_single, parse_matrix, plain

TOPIC = "r3l_simplex_final_tableau"
EXAM = "exd_x_r3l_simplex_final_tableau"
SKILL_ID = "uc2-simplex-final-tableau-reading"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _read(text):
    body = plain(text)
    tab = parse_matrix(body)
    cols = re.search(r"with columns (.*?), RHS", body).group(1).split(", ")
    basic = re.search(r"are ((?:[xs]\d(?:, )?)+)\.", body).group(1).split(", ")
    n = sum(1 for c in cols if c.startswith("x"))
    return tab, cols, basic, n


def _check_optimal(tab, cols, basic):
    m = tab.rows - 1
    assert len(basic) == m
    for row, name in enumerate(basic):
        col = cols.index(name)
        unit = [1 if i == row else 0 for i in range(tab.rows)]
        assert list(tab[:, col]) == unit, name
    assert all(tab[i, -1] >= 0 for i in range(m)) and all(v >= 0 for v in tab[-1, :-1])


def _allowable(tab, col):
    """The largest t >= 0 keeping every basic value non-negative when the resource changes by +t."""
    limits = [-tab[i, -1] / tab[i, col] for i in range(tab.rows - 1) if tab[i, col] < 0]
    return min(limits) if limits else None


def test_skill_wiring():
    assert _SKILL.answer == "fraction"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert "university-year-2" in _SKILL.tags and list(_SKILL.year_level) == ["university"]


def test_easy_key_is_the_variable_value():
    seen = Counter()
    for seed in range(80):
        q = _q("easy", seed)
        tab, cols, basic, n = _read(q.text)
        _check_optimal(tab, cols, basic)
        name = plain(re.search(r"value of ([xs]\S+) in the optimal", q.text).group(1))
        want = tab[basic.index(name), -1] if name in basic else 0
        assert sympy.Rational(q.answer_text) == want, (seed, q.text)
        seen[name in basic] += 1
        check_single(q, "fraction")
    assert seen[True] > 20 and seen[False] > 10


def test_medium_key_is_the_shadow_price():
    zero = 0
    for seed in range(80):
        q = _q("medium", seed)
        tab, cols, basic, n = _read(q.text)
        _check_optimal(tab, cols, basic)
        k = int(re.search(r"extra unit of resource (\d)", q.text).group(1))
        price = tab[-1, n + k - 1]
        assert sympy.Rational(q.answer_text) == price, (seed, q.text)
        zero += price == 0
        check_single(q, "fraction")
    assert 0 < zero < 40


def test_hard_keys():
    kinds = Counter()
    for seed in range(80):
        q = _q("hard", seed)
        tab, cols, basic, n = _read(q.text)
        _check_optimal(tab, cols, basic)
        if "By how much can the supply of resource" in q.text:
            k = int(re.search(r"resource (\d) be increased", q.text).group(1))
            allow = _allowable(tab, n + k - 1)
            assert sympy.Rational(q.answer_text) == allow, (seed, q.text)
            # just inside the range the basis survives, just beyond it a basic value goes negative
            for t, ok in ((allow * sympy.Rational(999, 1000), True), (allow * sympy.Rational(1001, 1000), False)):
                values = [tab[i, -1] + t * tab[i, n + k - 1] for i in range(tab.rows - 1)]
                assert all(v >= 0 for v in values) == ok
            kinds["range"] += 1
        else:
            k = int(re.search(r"Resource (\d) is increased by (\d+)", q.text).group(1))
            delta = int(re.search(r"increased by (\d+) unit", q.text).group(1))
            assert delta < _allowable(tab, n + k - 1)
            assert sympy.Rational(q.answer_text) == tab[-1, -1] + delta * tab[-1, n + k - 1], (seed, q.text)
            kinds["value"] += 1
        check_single(q, "fraction")
    assert kinds["range"] > 15 and kinds["value"] > 15


def test_exam_chain():
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        tab, cols, basic, n = _read(q.text)
        _check_optimal(tab, cols, basic)
        pa, pb, pc, pd = q.parts
        k = int(re.search(r"shadow price of resource (\d)", pb.prompt).group(1))
        xs = [int(tab[basic.index(f"x{j + 1}"), -1]) if f"x{j + 1}" in basic else 0 for j in range(n)]
        assert list(pa.answer) == xs
        assert sympy.Rational(pb.answer_text) == tab[-1, n + k - 1] > 0
        assert sympy.Rational(pc.answer_text) == _allowable(tab, n + k - 1)
        delta = int(re.search(r"increased by (\d+) unit", pd.prompt).group(1))
        assert delta < sympy.Rational(pc.answer_text)
        assert sympy.Rational(pd.answer_text) == tab[-1, -1] + delta * sympy.Rational(pb.answer_text)
        assert [p.answer_kind for p in q.parts] == ["ordered_list", "fraction", "fraction", "fraction"]
        check_parts(q)


def test_exam_follow_through():
    for seed in range(15):
        q = make(EXAM, seed=seed, difficulty="exam")
        price = sympy.Rational(q.parts[1].answer_text)
        check_follow_through(q, "b", price + 1)
