"""Tests `uc2-lp-duality-dual-formulation` (topic `r3l_lp_duality`) and its exam chain `exd_x_r3l_lp_duality`.

The oracle reads the primal back out of the question TEXT. The dual matrix is rebuilt with SymPy as [[A^T | c], [b | 0]]
(after negating any >= row of a maximisation); the hard answer must satisfy the dual feasibility, complementary
slackness and strong duality conditions against the stated optimal x*.
"""

from __future__ import annotations

import re
from collections import Counter

import sympy

import mathgen.generators.r3l_lp_duality_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import load_skills
from tests.r3l_support import check_follow_through, check_parts, check_single, plain

TOPIC = "r3l_lp_duality"
EXAM = "exd_x_r3l_lp_duality"
SKILL_ID = "uc2-lp-duality-dual-formulation"
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


def _primal(text):
    body = plain(text)
    sense, obj, rest = re.search(r"(?i)(maximise|minimise) [zw] = (.*?) subject to (.*?), and x1", body).groups()
    sense = sense.capitalize()
    n = len(set(re.findall(r"x(\d)", obj + rest)))
    c = _terms(obj, n)
    a, b, ops = [], [], []
    for part in rest.split(";"):
        op = "≤" if "≤" in part else "≥"
        lhs, rhs = part.split(op)
        a.append(_terms(lhs, n))
        b.append(int(rhs))
        ops.append(op)
    return sense, a, b, c, ops


def _dual_matrix(a, b, c):
    a, b = sympy.Matrix(a), sympy.Matrix(b)
    top = a.T.row_join(sympy.Matrix(c))
    return top.col_join(b.T.row_join(sympy.Matrix([0])))


def test_skill_wiring():
    assert _SKILL.answer == "matrix"
    assert [(s.topic, s.difficulty) for s in _SKILL.sources] == [
        (TOPIC, "easy"),
        (TOPIC, "medium"),
        (TOPIC, "hard"),
        (EXAM, "exam"),
    ]
    assert "university-year-2" in _SKILL.tags and list(_SKILL.year_level) == ["university"]


def test_easy_key_is_the_transposed_system():
    senses = Counter()
    for seed in range(60):
        q = _q("easy", seed)
        sense, a, b, c, ops = _primal(q.text)
        assert len(set(ops)) == 1 and ops[0] == ("≤" if sense == "Maximise" else "≥")
        assert _rows(q.answer) == _dual_matrix(a, b, c), (seed, q.text)
        senses[sense] += 1
        check_single(q, "matrix")
    assert senses["Maximise"] > 10 and senses["Minimise"] > 10


def test_medium_key_negates_the_at_least_rows():
    mixes = Counter()
    for seed in range(60):
        q = _q("medium", seed)
        sense, a, b, c, ops = _primal(q.text)
        assert sense == "Maximise" and "≥" in ops and "≤" in ops
        a_std = [[-v for v in row] if op == "≥" else row for row, op in zip(a, ops)]
        b_std = [-v if op == "≥" else v for v, op in zip(b, ops)]
        assert _rows(q.answer) == _dual_matrix(a_std, b_std, c), (seed, q.text)
        mixes[tuple(ops)] += 1
        check_single(q, "matrix")
    assert len(mixes) >= 4


def test_hard_key_satisfies_complementary_slackness():
    for seed in range(60):
        q = _q("hard", seed)
        sense, a, b, c, ops = _primal(q.text)
        x = [int(v) for v in re.search(r"x\* = \((\d+), (\d+)\)", plain(q.text)).groups()]
        y = _rows(q.answer)
        assert y.shape == (3, 1)
        a_m = sympy.Matrix(a)
        y_list = list(y)
        assert all(v >= 0 for v in y_list)
        assert list(a_m.T * y) == c  # dual feasibility with equality (both x* positive)
        slack = [b[i] - sum(a[i][j] * x[j] for j in range(2)) for i in range(3)]
        assert all(s >= 0 for s in slack)
        assert all(y_list[i] * slack[i] == 0 for i in range(3))  # complementary slackness
        assert sum(b[i] * y_list[i] for i in range(3)) == sum(c[j] * x[j] for j in range(2))  # strong duality
        assert sum(1 for s in slack if s > 0) == 1
        check_single(q, "matrix")


def test_exam_chain():
    for seed in range(40):
        q = make(EXAM, seed=seed, difficulty="exam")
        sense, a, b, c, ops = _primal(q.text)
        x = [int(v) for v in re.search(r"x\* = \((\d+), (\d+)\)", plain(q.text)).groups()]
        pa, pb, pc, pd = q.parts
        assert _rows(pa.answer) == _dual_matrix(a, b, c)
        slack = [b[i] - sum(a[i][j] * x[j] for j in range(2)) for i in range(3)]
        assert [int(v) for v in pb.answer] == slack
        k = int(re.search(r"dual variable y(\d)", plain(pc.prompt)).group(1)) - 1
        # solve A^T y = c over the two binding rows with SymPy
        binding = [i for i in range(3) if slack[i] == 0]
        y = sympy.symbols("y0:3")
        eqs = [sum(a[i][j] * y[i] for i in range(3) if i in binding) - c[j] for j in range(2)]
        sol = sympy.solve(eqs, [y[i] for i in binding], dict=True)[0]
        assert sympy.Rational(pc.answer_text) == sol[y[k]]
        delta = int(re.search(r"increased by (\d+)", pd.prompt).group(1))
        assert sympy.Rational(pd.answer_text) == sum(c[j] * x[j] for j in range(2)) + delta * sol[y[k]]
        assert [p.answer_kind for p in q.parts] == ["matrix", "ordered_list", "number", "number"]
        check_parts(q)


def test_exam_follow_through():
    for seed in range(15):
        q = make(EXAM, seed=seed, difficulty="exam")
        yk = sympy.Rational(q.parts[2].answer_text)
        check_follow_through(q, "c", yk + 1)
