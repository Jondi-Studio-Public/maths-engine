"""Tests for the multi-part exam source `exd_x_r2h_ridge_chain` (skill `ucv1-inverse-regularised-solve`).

The oracle reads A, b, lambda and the new input out of the rendered text and solves the system with SymPy.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy

from mathgen.engine import grade
from tests.exd_x_r2h_support import exam, follow_through_check, skill_of, standard_checks

TOPIC = "exd_x_r2h_ridge_chain"
SKILL_ID = "ucv1-inverse-regularised-solve"
HARD_TOPIC = "p14c_inverse_regularised_solve"
LABELS = list("abcd")
MARKS = [2, 2, 2, 1]


def parse(q):
    text = q.text
    rows_text = re.search(r"rows ((?:\(-?\d+, -?\d+\)(?:; )?)+)", text).group(1)
    rows = [(int(a), int(b)) for a, b in re.findall(r"\((-?\d+), (-?\d+)\)", rows_text)]
    b = [int(v) for v in re.search(r"b = \[([^\]]*)\]", text).group(1).split(",")]
    lam = int(re.search(r"lambda = (\d+)", text).group(1))
    u1, u2 = (int(v) for v in re.search(r"\(u1, u2\) = \((\d+), (\d+)\)", text).groups())
    return rows, b, lam, u1, u2


def _fr(value) -> Fraction:
    value = sympy.Rational(value)
    return Fraction(int(value.p), int(value.q))


def oracle(q):
    rows, b, lam, u1, u2 = parse(q)
    a_mat = sympy.Matrix(rows)
    m_mat = a_mat.T * a_mat + lam * sympy.eye(2)
    x = m_mat.LUsolve(a_mat.T * sympy.Matrix(b))
    return {
        "a": str(m_mat.det()),
        "b": str(_fr(x[0])),
        "c": str(_fr(x[1])),
        "d": str(_fr(u1 * x[0] + u2 * x[1])),
    }


CHECKS = standard_checks(TOPIC, SKILL_ID, oracle, LABELS, MARKS, ["b", "c", "d"], HARD_TOPIC)


def test_wired():
    CHECKS["wired"]()


def test_shape():
    CHECKS["shape"]()


def test_prompt_shapes():
    CHECKS["shapes"]()


def test_oracle_agrees_over_300_seeds():
    CHECKS["oracle_agrees"]()


def test_composed_answer_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_tagged_and_rejected():
    CHECKS["distractors_rejected"]()


def test_answer_space():
    CHECKS["answer_space"]()


def test_deterministic():
    CHECKS["deterministic"]()


def _gram(q):
    rows, b, lam, u1, u2 = parse(q)
    g11 = sum(r[0] * r[0] for r in rows)
    g12 = sum(r[0] * r[1] for r in rows)
    g22 = sum(r[1] * r[1] for r in rows)
    c1 = sum(r[0] * bi for r, bi in zip(rows, b))
    c2 = sum(r[1] * bi for r, bi in zip(rows, b))
    return g11 + lam, g12, g22 + lam, c1, c2, u1, u2


def test_follow_through_from_wrong_determinant():
    def wrong_chain(q):
        m11, m12, m22, c1, c2, u1, u2 = _gram(q)
        det = Fraction(q.parts[0].distractors[0])
        x1 = (m22 * c1 - m12 * c2) / det
        x2 = (m11 * c2 - m12 * c1) / det
        replies = {"a": q.parts[0].distractors[0], "b": str(x1), "c": str(x2), "d": str(u1 * x1 + u2 * x2)}
        return "a", replies, 5

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_follow_through_from_wrong_x1():
    def wrong_chain(q):
        *_, u1, u2 = _gram(q)
        x1 = Fraction(q.parts[1].distractors[0])
        x2 = Fraction(q.parts[2].answer_text)
        replies = {
            "a": q.parts[0].answer_text,
            "b": str(x1),
            "c": q.parts[2].answer_text,
            "d": str(u1 * x1 + u2 * x2),
        }
        return "b", replies, 5

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_all_wrong_earns_nothing():
    skill = skill_of(SKILL_ID)
    for s in range(30):
        q = exam(TOPIC, s)
        result = grade(skill, q, {"a": "1009", "b": "211/1013", "c": "-307/1019", "d": "401/1031"})
        assert result.marks == 0
