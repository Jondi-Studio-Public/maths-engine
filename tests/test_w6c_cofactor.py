"""Tests `uc2-cofactor-expansion-4x4` (topic `w6c_cofactor`) and its exam chain `exd_x_w6c_cofactor`.

The oracle reads the matrix (or the stated determinants) back out of the question TEXT and re-derives each key with
SymPy: `Matrix.det()`, `cofactor()`, `inv()`, and a direct evaluation of each stated expression on matrices with the
stated determinants.
"""

from __future__ import annotations

import ast
import re
from fractions import Fraction

import sympy

import mathgen.generators.w6c_cofactor_build  # noqa: F401  (registers the topics)
from mathgen import make
from tests import w6c_support as support

TOPIC = "w6c_cofactor"
EXAM = "exd_x_w6c_cofactor"
SKILL = support.skill_named("uc2-cofactor-expansion-4x4")
_MATRIX = re.compile(r"\[\[[^A-Za-z]*?\]\]")
_MATRIX_K = re.compile(r"\[\[[^A-Za-z]*?k[^A-Za-z]*?\]\]")


def _matrix(text):
    return sympy.Matrix(ast.literal_eval(_MATRIX.search(text).group(0)))


def _value(text):
    f = Fraction(text)
    return sympy.Rational(f.numerator, f.denominator)


def _hard_kind(text):
    if "singular" in text or "not invertible" in text or "non-zero solution" in text:
        return "parameter"
    if "row" in text:
        return "rows"
    return "properties"


def test_skill_wiring():
    support.check_wiring(SKILL, TOPIC, EXAM, ["matrix-determinant-3x3"])


def test_each_tier_key_accepted_and_every_distractor_rejected():
    for tier in support.TIERS:
        support.check_tier(SKILL, TOPIC, tier)


def test_easy_key_is_the_determinant_of_a_sparse_matrix():
    for seed in range(100):
        q = make(TOPIC, seed=seed, difficulty="easy")
        m = _matrix(q.text)
        assert m.shape == (4, 4)
        assert _value(q.answer_text) == m.det() != 0, (seed, q.text)
        sparse_lines = [r for r in range(4) if sum(1 for v in m.row(r) if v) == 1] + [
            c for c in range(4) if sum(1 for v in m.col(c) if v) == 1
        ]
        assert sparse_lines, (seed, q.text)


def test_medium_keys_are_a_determinant_or_a_cofactor():
    kinds = set()
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="medium")
        m = _matrix(q.text)
        if "cofactor" in q.text:
            i, j = (int(n) for n in re.findall(r"C(\d)(\d)", q.text)[0]) if "C" in q.text else (0, 0)
            nums = [int(n) for n in re.findall(r"i = (\d+) and j = (\d+)", q.text)[0]] if "i = " in q.text else None
            if nums:
                i, j = nums
            elif "row" in q.text:
                i, j = (int(n) for n in re.findall(r"row (\d), column (\d)", q.text)[0])
            assert _value(q.answer_text) == m.cofactor(i - 1, j - 1), (seed, q.text)
            kinds.add("cofactor")
        else:
            assert _value(q.answer_text) == m.det(), (seed, q.text)
            assert abs(m.det()) >= 2
            kinds.add("det")
    assert kinds == {"det", "cofactor"}


def test_hard_keys():
    seen = {"properties": 0, "rows": 0, "parameter": 0}
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="hard")
        kind = _hard_kind(q.text)
        seen[kind] += 1
        key = _value(q.answer_text)
        if kind == "parameter":
            literal = _MATRIX_K.search(q.text).group(0)
            rows = ast.literal_eval(literal.replace("k", "'k'"))
            k = sympy.symbols("k")
            m = sympy.Matrix([[k if v == "k" else v for v in row] for row in rows])
            assert sympy.solve(m.det(), k) == [key], (seed, q.text)
        elif kind == "rows":
            d = int(re.search(r"det A = (-?\d+)", q.text).group(1))
            factor = _net_factor(q.text)
            assert key == factor * d, (seed, q.text)
        else:
            p, r = (int(n) for n in re.findall(r"det [AB] = (-?\d+)", q.text))
            expr = next(e for e in _EXPRESSIONS if f"det({e})" in q.text or f"of {e}?" in q.text)
            assert key == _property(expr, p, r), (seed, q.text, expr)
    assert all(count >= 25 for count in seen.values()), seen


def _net_factor(text):
    """The net factor of the row operations listed in the question text, read independently of the generator."""
    ops = re.findall(r"(swapping rows \d and \d|multiplying row \d by -?\d+|adding -?\d+ times row \d to row \d)", text)
    assert len(ops) == 3, text
    factor = 1
    for op in ops:
        if op.startswith("swapping"):
            factor *= -1
        elif op.startswith("multiplying"):
            factor *= int(op.rsplit(" ", 1)[1])
    return factor


_EXPRESSIONS = ("2Aᵀ", "3A⁻¹B", "A²B⁻¹", "adj(A)", "(2A)⁻¹", "−AᵀB⁻¹", "ABAᵀ", "2AB⁻¹")


def _property(expr, p, q):
    a = sympy.diag(p, 1, 1, 1) + sympy.Matrix(4, 4, lambda r, c: 1 if (r, c) == (0, 1) else 0)
    b = sympy.diag(q, 1, 1, 1) + sympy.Matrix(4, 4, lambda r, c: 1 if (r, c) == (1, 0) else 0)
    expressions = {
        "2Aᵀ": 2 * a.T,
        "3A⁻¹B": 3 * a.inv() * b,
        "A²B⁻¹": a**2 * b.inv(),
        "adj(A)": a.adjugate(),
        "(2A)⁻¹": (2 * a).inv(),
        "−AᵀB⁻¹": -a.T * b.inv(),
        "ABAᵀ": a * b * a.T,
        "2AB⁻¹": 2 * a * b.inv(),
    }
    return expressions[expr].det()


def test_tier_shapes_and_answer_space():
    support.check_tier_shapes_differ(TOPIC)
    for tier in support.TIERS:
        support.check_answer_space(TOPIC, tier, distinct=8, share=0.25)


def test_exam_shape():
    q = make(EXAM, seed=2, difficulty="exam")
    assert [p.answer_kind for p in q.parts] == ["number", "number", "fraction", "fraction"]
    assert [p.follow_through for p in q.parts] == [None, None, "part_b/part_a", "81/part_a"]


def test_exam_keys_match_independent_derivation():
    for seed in range(120):
        q = make(EXAM, seed=seed, difficulty="exam")
        m = _matrix(q.stem)
        i, j = (int(n) for n in re.search(r"cofactor C(\d)(\d)", q.parts[1].prompt).groups())
        assert _value(q.parts[0].answer_text) == m.det()
        assert _value(q.parts[1].answer_text) == m.cofactor(i - 1, j - 1)
        assert f"row {j}, column {i}" in q.parts[2].prompt
        assert _value(q.parts[2].answer_text) == m.inv()[j - 1, i - 1]
        assert _value(q.parts[3].answer_text) == (3 * m.inv()).det()


def test_exam_grading_and_follow_through():
    support.check_exam_grading(SKILL, EXAM, [3, 2, 2, 2])
    support.check_follow_through(SKILL, EXAM)


def test_exam_answer_space():
    answers = {make(EXAM, seed=s, difficulty="exam").answer_text for s in range(200)}
    assert len(answers) >= 140
