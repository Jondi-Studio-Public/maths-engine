"""Exam chain for `expand-binomial-trinomial`: find k from a stated coefficient, expand, then evaluate."""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import exd_x_r2j_kit as kit

SKILL = "expand-binomial-trinomial"
TOPIC = "exd_x_r2j_expand_unknown_chain"
SEEDS = range(300)
X, K = sympy.symbols("x k")
_TR = standard_transformations + (implicit_multiplication_application,)


@pytest.fixture(scope="module")
def questions():
    return [kit.exam_q(SKILL, s) for s in SEEDS]


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TR, local_dict={"x": X, "k": K})


def read(q):
    m = re.search(r"\((x [+-] \d+)\)\(([^()]+)\)", q.stem)
    n = int(re.search(r"coefficient of x (?:is|in the expanded form of P\(x\) is) (-?\d+)", q.stem).group(1))
    x0 = int(re.search(r"P\((-?\d+)\)", q.parts[2].prompt).group(1))
    return _parse(m.group(1)), _parse(m.group(2)), n, x0


def oracle(q) -> dict[str, str]:
    binomial, quad, n, x0 = read(q)
    product = sympy.expand(binomial * quad)
    k = sympy.solve(sympy.Eq(product.coeff(X, 1), n), K)
    assert len(k) == 1
    expanded = sympy.expand(product.subs(K, k[0]))
    return {"a": str(k[0]), "b": sympy.sstr(expanded), "c": str(expanded.subs(X, x0))}


def test_wiring_shapes_defects_determinism():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_shape(questions):
    for q in questions:
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.answer_kind for p in q.parts] == ["number", "expression", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]
        assert q.parts[1].answer_form == "expanded"


def test_oracle_agrees(questions):
    for q in questions:
        o = oracle(q)
        c = kit.canon(q)
        assert o["a"] == c["a"] and o["c"] == c["c"], q.text
        assert sympy.expand(_parse(c["b"]) - sympy.sympify(o["b"])) == 0, q.text


def test_answer_space():
    kit.check_answer_space(SKILL)


def test_grading():
    kit.check_grading(SKILL, n=100)


def test_unexpanded_product_rejected(questions):
    for q in questions[:60]:
        binomial, quad, n, _ = read(q)
        k = int(kit.canon(q)["a"])
        unexpanded = f"({sympy.sstr(binomial)})({sympy.sstr(quad.subs(K, k))})"
        assert grade(kit.skill(SKILL), q, {**kit.canon(q), "b": unexpanded}).parts["b"].marks == 0


def test_follow_through_from_wrong_k(questions):
    checked = 0
    for q in questions[:100]:
        binomial, quad, n, x0 = read(q)
        k_true = int(kit.canon(q)["a"])
        k_wrong = k_true + 2 if k_true != -2 else k_true + 3  # zero earns no follow-through
        expanded = sympy.expand(binomial * quad.subs(K, k_wrong))
        value = expanded.subs(X, x0)
        if value == int(kit.canon(q)["c"]):
            continue
        reply = {"a": str(k_wrong), "b": sympy.sstr(expanded), "c": str(value)}
        result = grade(kit.skill(SKILL), q, reply)
        assert result.parts["a"].marks == 0
        assert result.parts["b"].marks == q.parts[1].marks, (q.text, reply)
        assert result.parts["c"].marks == q.parts[2].marks, (q.text, reply)
        checked += 1
    assert checked >= 80


def test_correct_expansion_with_wrong_evaluation_loses_only_the_last_part(questions):
    for q in questions[:60]:
        c = kit.canon(q)
        result = grade(kit.skill(SKILL), q, {**c, "c": str(int(c["c"]) + 1)})
        assert result.parts["c"].marks == 0 and result.parts["b"].marks == q.parts[1].marks
