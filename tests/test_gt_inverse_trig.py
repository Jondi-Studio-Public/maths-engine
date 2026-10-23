"""Tests the inverse-trig-values-and-graph-features skill: keys re-derived by SymPy and a scan, grading, exam chain."""

from __future__ import annotations

import re

import numpy as np
import pytest
import sympy as sp
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import sr2b_support as sup

SKILL = "inverse-trig-values-and-graph-features"
TOPIC = "gt_inverse_trig"
EXAM = "gt_inverse_trig_exam"
X = sp.Symbol("x")
_T = standard_transformations + (implicit_multiplication_application,)
_NAMES = {"arcsin": "asin", "arccos": "acos", "arctan": "atan"}


def _parse(text: str):
    text = text.replace("π", "pi").replace("^", "**")
    for k, v in _NAMES.items():
        text = text.replace(k, v)
    return parse_expr(text, local_dict={"x": X}, transformations=_T)


def _function(stem: str):
    tail = next(t for t in re.split(r"(?:y|f\(x\)) = ", stem) if "arc" in t)
    return _parse(re.match(r"(.+?)(?= has|\.(?: |$)|,)", tail).group(1))


def _same(a, b) -> bool:
    return sp.simplify(sp.nsimplify(a) - sp.nsimplify(b)) == 0 or abs(float(sp.N(a - b, 30))) < 1e-12


def _greatest(expr):
    """The greatest value: the larger value at the two ends of the domain (the function is monotonic), checked by a scan."""
    (atom,) = expr.atoms(sp.asin, sp.acos)
    ends = [r for v in (1, -1) for r in sp.solve(atom.args[0] - v, X)]
    top = max(expr.subs(X, e) for e in ends)
    xs = np.linspace(-4, 4, 200001)
    with np.errstate(invalid="ignore"):
        assert float(np.nanmax(sp.lambdify(X, expr, "numpy")(xs))) <= float(top) + 1e-9
    return top


def _derive_medium(stem: str):
    if "domain" in stem or "is defined" in stem:
        expr = _function(stem)
        (atom,) = expr.atoms(sp.asin, sp.acos)
        edges = [r for v in (1, -1) for r in sp.solve(atom.args[0] - v, X)]
        return max(edges)
    if "asymptote" in stem:
        expr = _function(stem)
        return max(sp.limit(expr, X, sp.oo), sp.limit(expr, X, -sp.oo))
    if "greatest value" in stem:
        return _greatest(_function(stem))
    m = re.search(r"f\((-?[\d/]+)\)", stem)
    return _function(stem).subs(X, sp.Rational(m.group(1)))


def _derive_hard(stem: str):
    m = re.search(r"(?:Evaluate|exact value of) (.+?)(?:\.| exactly| without| Give|$)", stem)
    return _parse(m.group(1)).doit()


def test_wired():
    sup.assert_wired(
        SKILL,
        TOPIC,
        EXAM,
        "expression",
        ["senior-understand-the-unit-circle-definition-of-cos-theta-acmmm034", "exact-trig-values-typed"],
    )


def test_easy_keys_are_the_principal_values():
    for seed in range(80):
        q = sup.tier_question(TOPIC, "easy", seed)
        m = re.search(r"(arcsin|arccos|arctan)\((.+?)\)(?: |\.|\?)", q.text + " ")
        want = {"arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan}[m.group(1)](_parse(m.group(2)))
        assert _same(_parse(q.answer_text), want), (seed, q.text, q.answer_text)
        lo, hi = (0, sp.pi) if m.group(1) == "arccos" else (-sp.pi / 2, sp.pi / 2)
        assert lo <= _parse(q.answer_text) <= hi


def test_medium_keys_match_an_independent_derivation():
    kinds = set()
    for seed in range(120):
        q = sup.tier_question(TOPIC, "medium", seed)
        want = _derive_medium(q.text)
        kinds.add(
            next(
                (k for k in ("domain", "asymptote", "greatest value") if k in q.text),
                "defined" if "is defined" in q.text else "value",
            )
        )
        got = _parse(q.answer_text)
        assert abs(float(sp.N(got - want, 30))) < 1e-6, (seed, q.text, q.answer_text, want)
    assert {"asymptote", "greatest value", "value"} <= kinds and kinds & {"domain", "defined"}, kinds


def test_hard_keys_match_an_independent_derivation():
    for seed in range(100):
        q = sup.tier_question(TOPIC, "hard", seed)
        want = _derive_hard(q.text)
        got = _parse(q.answer_text)
        assert abs(float(sp.N(got - want, 30))) < 1e-9, (seed, q.text, q.answer_text, want)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=8, top=0.25)


def test_equivalent_spellings_are_accepted():
    sk = sup.skill_object(SKILL)
    q = next(q for q in (sup.tier_question(TOPIC, "easy", s) for s in range(40)) if q.answer_text == "pi/6")
    for reply in ("pi/6", "1/6*pi", "(1/6)pi", "30/180*pi"):
        assert grade(sk, q, reply).correct, reply


def test_degrees_are_not_accepted_for_a_radian_answer():
    sk = sup.skill_object(SKILL)
    q = next(q for q in (sup.tier_question(TOPIC, "easy", s) for s in range(40)) if q.answer_text == "pi/6")
    assert not grade(sk, q, "30").correct


def test_tiers_differ():
    sup.assert_tier_shapes(TOPIC, EXAM, minimum_hard=2)
    easy = " ".join(sup.tier_question(TOPIC, "easy", s).text for s in range(40))
    hard = " ".join(sup.tier_question(TOPIC, "hard", s).text for s in range(40))
    assert "(sin(" not in easy and "(sin(" in hard


def test_exam_keys_match_an_independent_derivation():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        f = _function(q.text)
        a, b, c, d = q.parts
        (atom,) = f.atoms(sp.asin)
        edges = [r for v in (1, -1) for r in sp.solve(atom.args[0] - v, X)]
        assert sp.Rational(a.answer_text) == max(edges), (seed, q.text)
        assert _same(_parse(b.answer_text), _greatest(f)), (seed, q.text)
        x0 = sp.Rational(re.search(r"f\((-?[\d/]+)\)", c.prompt).group(1))
        assert _same(_parse(c.answer_text), f.subs(X, x0)), (seed, q.text)
        zeros = [r for r in sp.solve(f, X)]
        assert [sp.Rational(d.answer_text)] == zeros, (seed, q.text)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(
        EXAM, ["fraction", "expression", "expression", "fraction"], [1, 2, 2, 2], [False, False, False, True]
    )
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=30, top=0.1)


def test_exam_part_d_follows_through_from_a():
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        s = sp.Rational(q.parts[3].follow_through.split(")")[0].strip("("))
        replies = {**canon, "a": "3/7", "d": str(s * sp.Rational(3, 7))}
        assert sup.part_marks(SKILL, q, replies, "d") == q.parts[3].marks, (seed, replies)
        assert sup.part_marks(SKILL, q, {**replies, "d": "5"}, "d") == 0
