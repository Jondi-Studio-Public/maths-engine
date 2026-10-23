"""Tests the graph-f-squared-features skill: keys re-derived from the drawn graph or the rule, grading, exam chain."""

from __future__ import annotations

import re

import numpy as np
import pytest
import sympy as sp
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from tests import sr2b_support as sup

SKILL = "graph-f-squared-features"
TOPIC = "gt_f_squared"
EXAM = "gt_f_squared_exam"
X = sp.Symbol("x")
_T = standard_transformations + (implicit_multiplication_application,)


def _rule(stem: str):
    text = re.search(r"f\(x\) = ([^.?]*?)(?:\. |\.$|, )", stem + " ") or re.search(r"rule f\(x\) = (.*?)\.", stem)
    return parse_expr(text.group(1).replace("^", "**"), local_dict={"x": X}, transformations=_T)


def _pair(text: str):
    return [sp.Rational(v) for v in re.findall(r"= (-?\d+(?:/\d+)?)", text)]


def _drawn_turning_points(fig):
    """(x, y, kind) of the turning points of f(x)^2 read from the drawn samples of f, one piece at a time."""
    found = []
    curve = next(c for c in fig["curves"] if c["label"] == "f")
    for piece in curve["pieces"]:
        xs = np.array([p[0] for p in piece])
        g = np.array([p[1] for p in piece]) ** 2
        for i in range(1, len(g) - 1):
            if (
                g[i] < g[i - 1] - 1e-12
                and g[i] <= g[i + 1] + 1e-12
                and g[i] < g[i + 1] + 1e-12
                or (g[i] < g[i - 1] - 1e-12 and g[i] < g[i + 1] - 1e-12)
            ):
                found.append((float(xs[i]), float(g[i]), "min"))
            if g[i] > g[i - 1] + 1e-12 and g[i] > g[i + 1] + 1e-12:
                found.append((float(xs[i]), float(g[i]), "max"))
    return found


def test_wired():
    sup.assert_wired(SKILL, TOPIC, EXAM, "pair", ["function-notation", "senior-graph-reciprocal-abs-compare"])


def test_easy_key_is_the_zero_of_the_drawn_line():
    for seed in range(60):
        q = sup.tier_question(TOPIC, "easy", seed)
        assert q.figure is not None
        pts = [p for piece in q.figure["curves"][0]["pieces"] for p in piece]
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        root = x0 - y0 * (x1 - x0) / (y1 - y0)
        assert [float(v) for v in _pair(q.answer_text)] == pytest.approx([root, 0.0], abs=1e-6), (seed, q.text)
        tps = _drawn_turning_points(q.figure)
        assert len(tps) == 1 and tps[0][2] == "min" and abs(tps[0][0] - root) < 1e-6


def test_medium_key_is_the_requested_turning_point_of_the_drawn_square():
    for seed in range(80):
        q = sup.tier_question(TOPIC, "medium", seed)
        tps = _drawn_turning_points(q.figure)
        want = "max" if "maximum" in q.text or "local max" in q.text else None
        picked = [t for t in tps if t[2] == want] if want else tps
        assert len(picked) == 1, (seed, q.text, tps)
        x, y, _ = picked[0]
        assert [float(v) for v in _pair(q.answer_text)] == pytest.approx([x, y], abs=1e-6), (seed, q.text)
        if "three turning points" in q.text:
            assert len(tps) == 3
        if "never meets" in q.text:
            assert len(tps) == 1


def test_hard_keys_match_an_independent_derivation():
    seen = set()
    for seed in range(100):
        q = sup.tier_question(TOPIC, "hard", seed)
        f = _rule(q.text)
        if "local maximum" in q.text:
            seen.add("max")
            assert sp.degree(f, X) == 2 and sp.Poly(f, X).LC() == 1
            roots = sp.real_roots(sp.Poly(f**2, X).diff(X))
            tops = [r for r in roots if sp.diff(f**2, X, 2).subs(X, r) < 0]
            assert len(tops) == 1
            assert _pair(q.answer_text) == [tops[0], (f**2).subs(X, tops[0])], (seed, q.text)
        else:
            seen.add("meet")
            assert "meet" in q.text
            on_axis = sp.solve(f, X)
            others = [r for r in sp.solve(sp.Eq(f**2, f), X) if r not in on_axis]
            assert len(others) == 1, (seed, q.text)
            assert _pair(q.answer_text) == [others[0], f.subs(X, others[0])], (seed, q.text)
    assert seen == {"max", "meet"}


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=8, top=0.3)


def test_order_of_a_pair_does_not_matter():
    from mathgen.engine import grade

    sk = sup.skill_object(SKILL)
    q = sup.tier_question(TOPIC, "medium", 0)
    first, second = q.answer_text.split(", ")
    assert grade(sk, q, f"{second}, {first}").correct


def test_tiers_differ():
    assert all(q.figure is not None for q in (sup.tier_question(TOPIC, "easy", s) for s in range(20)))
    assert all(q.figure is not None for q in (sup.tier_question(TOPIC, "medium", s) for s in range(20)))
    assert all(q.figure is None for q in (sup.tier_question(TOPIC, "hard", s) for s in range(20)))
    sup.assert_tier_shapes(TOPIC, EXAM, minimum_hard=2)


def test_exam_keys_match_an_independent_derivation():
    for seed in range(100):
        q = sup.exam_question(EXAM, seed)
        f = _rule(q.text)
        a, b, c, d = q.parts
        least = min(sp.real_roots(sp.diff(f, X)) and [f.subs(X, r) for r in sp.real_roots(sp.diff(f, X))])
        assert sp.Rational(a.answer_text) == least, (seed, q.text)
        assert sp.Rational(b.answer_text) == least**2
        slope = sp.Poly(sp.diff(f**2, X), X)
        turning = {r for r in sp.real_roots(slope)}
        assert int(c.answer_text) == len(turning), (seed, q.text)
        k = sp.Rational(re.search(r"\^2 = (-?\d+)", d.prompt).group(1))
        assert int(d.answer_text) == len(set(sp.real_roots(sp.Poly(f**2 - k, X)))), (seed, q.text)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["number"] * 4, [2, 2, 1, 2], [False, True, False, False])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=25, top=0.1)


def test_exam_part_b_follows_through_from_a():
    for seed in range(20):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        replies = {**canon, "a": "-3", "b": "9"}
        assert sup.part_marks(SKILL, q, replies, "b") == q.parts[1].marks
        assert sup.part_marks(SKILL, q, {**replies, "b": "-9"}, "b") == 0
