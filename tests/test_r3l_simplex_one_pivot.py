"""Tests `uc2-simplex-one-pivot` (topic `r3l_simplex_one_pivot`) and its exam chain `exd_x_r3l_simplex_one_pivot`.

The oracle reads the linear programme (or the supplied tableau) back out of the question TEXT and applies the stated
pivot rule with SymPy rationals, row by row, without using the generator's kit.
"""

from __future__ import annotations

import re
from collections import Counter

import sympy

import mathgen.generators.r3l_simplex_one_pivot_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import load_skills
from tests.r3l_support import check_follow_through, check_parts, check_single, parse_matrix, plain

TOPIC = "r3l_simplex_one_pivot"
EXAM = "exd_x_r3l_simplex_one_pivot"
SKILL_ID = "uc2-simplex-one-pivot"
_SKILL = next(s for s in load_skills()[1] if s.id == SKILL_ID)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _rows(answer):
    return sympy.Matrix(
        [[sympy.Rational(e["p"], e["q"]) if isinstance(e, dict) else sympy.Integer(e) for e in row] for row in answer]
    )


def _terms(text, n):
    coefs = [0] * n
    for sign, mag, idx in re.findall(r"([+-]?)\s*(\d*)x(\d)", text.replace("−", "-")):
        coefs[int(idx) - 1] += (-1 if sign == "-" else 1) * int(mag or 1)
    return coefs


def _lp(text):
    body = plain(text)
    obj, rest = re.search(r"Maximise z = (.*?) subject to (.*?), and x1", body).groups()
    n = len(set(re.findall(r"x(\d)", obj + rest)))
    c = _terms(obj, n)
    a, b = [], []
    for part in rest.split(";"):
        lhs, rhs = part.split("≤")
        a.append(_terms(lhs, n))
        b.append(int(rhs))
    return a, b, c


def _initial(a, b, c):
    m, n = len(a), len(c)
    rows = [list(a[i]) + [int(i == j) for j in range(m)] + [b[i]] for i in range(m)]
    rows.append([-v for v in c] + [0] * m + [0])
    return sympy.Matrix(rows)


def _pivot(tab):
    """One pivot by the stated rule: most negative entry (lowest column on a tie), then the minimum ratio (lowest row)."""
    last = tab.rows - 1
    obj = list(tab.row(last))[:-1]
    col = obj.index(min(obj))
    ratios = [(tab[i, -1] / tab[i, col], i) for i in range(last) if tab[i, col] > 0]
    row = min(ratios)[1]
    out = tab.copy()
    out[row, :] = tab[row, :] / tab[row, col]
    for i in range(tab.rows):
        if i != row:
            out[i, :] = tab[i, :] - tab[i, col] * out[row, :]
    return out, row, col, min(ratios)[0]


def test_skill_wiring():
    assert _SKILL.answer == "matrix"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert "university-year-2" in _SKILL.tags and list(_SKILL.year_level) == ["university"]


def test_easy_key_is_the_initial_tableau():
    shapes = Counter()
    for seed in range(50):
        q = _q("easy", seed)
        a, b, c = _lp(q.text)
        assert _rows(q.answer) == _initial(a, b, c), (seed, q.text)
        shapes[(len(a), len(c))] += 1
        check_single(q, "matrix")
    assert len(shapes) >= 3


def test_medium_key_is_the_next_tableau():
    for seed in range(50):
        q = _q("medium", seed)
        before = parse_matrix(plain(q.text))
        a, b, c = _lp(q.text)
        assert before == _initial(a, b, c)
        after, *_ = _pivot(before)
        assert _rows(q.answer) == after, (seed, q.text)
        check_single(q, "matrix")


def test_hard_key_is_the_next_tableau_from_mid_run():
    fractional = 0
    for seed in range(50):
        q = _q("hard", seed)
        before = parse_matrix(plain(q.text))
        after, *_ = _pivot(before)
        assert _rows(q.answer) == after, (seed, q.text)
        # the supplied tableau is a genuine mid-run tableau: its right-hand sides are non-negative and a unit column is not a slack
        assert all(before[i, -1] >= 0 for i in range(before.rows - 1))
        fractional += any(v.q != 1 for v in before)
        check_single(q, "matrix")
    assert fractional >= 30


def test_tiers_differ_in_shape():
    easy = [_q("easy", s).text for s in range(10)]
    hard = [_q("hard", s).text for s in range(10)]
    assert all("Maximise" in t for t in easy) and not any("Maximise" in t for t in hard)


def test_exam_chain():
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        a, b, c = _lp(q.text)
        first = _initial(a, b, c)
        second, row, col, theta = _pivot(first)
        pa, pb, pc, pd = q.parts
        assert _rows(pa.answer) == first
        assert sympy.Rational(pb.answer_text) == theta
        assert _rows(pc.answer) == second
        assert sympy.Rational(pd.answer_text) == second[-1, -1] == -first[-1, col] * theta
        assert [p.answer_kind for p in q.parts] == ["matrix", "fraction", "matrix", "fraction"]
        check_parts(q)


def test_exam_follow_through():
    for seed in range(15):
        q = make(EXAM, seed=seed, difficulty="exam")
        theta = sympy.Rational(q.parts[1].answer_text)
        check_follow_through(q, "b", theta + 2)
