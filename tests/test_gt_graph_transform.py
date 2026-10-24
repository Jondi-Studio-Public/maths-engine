"""Tests the graph-transformation-equation skill: keys re-derived from the stem or the drawn graph, grading, exam chain."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest
import sympy as sp
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen.engine import grade
from tests import sr2b_support as sup

SKILL = "graph-transformation-equation"
TOPIC = "gt_graph_transform"
EXAM = "gt_graph_transform_exam"
X = sp.Symbol("x")
_T = standard_transformations + (implicit_multiplication_application,)
FAMILIES = {
    "x^2": X**2,
    "x^3": X**3,
    "2^x": 2**X,
    "3^x": 3**X,
    "e^x": sp.exp(X),
    "sin(x)": sp.sin(X),
    "cos(x)": sp.cos(X),
    "tan(x)": sp.tan(X),
    "1/x": 1 / X,
}
_BASE = re.compile(r"(?:y|f\(x\)) = (x\^2|x\^3|2\^x|3\^x|e\^x|sin\(x\)|cos\(x\)|tan\(x\)|1/x)")
_OP = re.compile(
    r"translated (?P<n>\S+) units? (?P<d>to the right|to the left|up|down)"
    r"|reflected in the (?P<r>[xy])-axis"
    r"|dilated by a factor of (?P<k>\S+) from the (?P<a>[xy])-axis"
)


def _num(text: str):
    return parse_expr(text.replace("π", "pi"), transformations=_T)


def _typed(text: str):
    return parse_expr(text.replace("^", "**"), local_dict={"x": X, "e": sp.E, "pi": sp.pi}, transformations=_T)


def _ops(stem: str):
    """The transformations in the stem as ("tx", h), ("ty", k), ("rx",), ("ry",), ("dx", k), ("dy", c)."""
    out = []
    for m in _OP.finditer(stem):
        if m["n"]:
            n = _num(m["n"])
            out.append(
                {"to the right": ("tx", n), "to the left": ("tx", -n), "up": ("ty", n), "down": ("ty", -n)}[m["d"]]
            )
        elif m["r"]:
            out.append(("rx",) if m["r"] == "x" else ("ry",))
        else:
            out.append(("dy" if m["a"] == "x" else "dx", _num(m["k"])))
    return out


def _image(stem: str):
    base = FAMILIES[_BASE.search(stem).group(1)]
    for op in _ops(stem):
        if op[0] == "tx":
            base = base.subs(X, X - op[1])
        elif op[0] == "ty":
            base = base + op[1]
        elif op[0] == "rx":
            base = -base
        elif op[0] == "ry":
            base = base.subs(X, -X)
        elif op[0] == "dy":
            base = op[1] * base
        else:
            base = base.subs(X, X / op[1])
    return base


def _same(a, b) -> bool:
    fa, fb = sp.lambdify(X, a, "numpy"), sp.lambdify(X, b, "numpy")
    xs = np.linspace(-3.1, 3.1, 61) + 0.0137
    with np.errstate(all="ignore"):
        va, vb = np.asarray(fa(xs), dtype=float), np.asarray(fb(xs), dtype=float)
    ok = np.isfinite(va) & np.isfinite(vb) & (np.abs(va) < 1e4)
    return ok.sum() > 20 and bool(np.allclose(va[ok], vb[ok], rtol=1e-9, atol=1e-9))


def test_wired():
    sup.assert_wired(
        SKILL, TOPIC, EXAM, "expression", ["senior-graph-translation-read-shift", "senior-graph-dilation-read-factor"]
    )


@pytest.mark.parametrize("tier", ["easy", "medium"])
def test_worded_keys_match_the_stated_transformations(tier):
    for seed in range(120):
        q = sup.tier_question(TOPIC, tier, seed)
        ops = _ops(q.text)
        assert len(ops) >= (2 if tier == "easy" else 3), (seed, q.text)
        assert _same(_typed(q.answer_text), _image(q.text)), (seed, q.text, q.answer_text)


def test_drawn_keys_pass_through_the_graph_of_g():
    for seed in range(50):
        q = sup.tier_question(TOPIC, "hard", seed)
        fig = q.figure
        assert fig is not None
        scale = math.pi if fig.get("x_tick_format") == "pi" else 1.0
        key = sp.lambdify(X, _typed(q.answer_text), "numpy")
        g = next(c for c in fig["curves"] if c["label"] == "g")
        pts = [p for piece in g["pieces"] for p in piece]
        assert len(pts) > 50
        for x, y in pts[:: max(1, len(pts) // 60)]:
            assert abs(float(key(x * scale)) - y) < 1e-4 * (1 + abs(y)), (seed, x, y, q.answer_text)
        for m in fig["points"]:
            x, y = m["at"]
            assert abs(float(key(x * scale)) - y) < 1e-9, (seed, m, q.answer_text)
        base = FAMILIES[_BASE.search(q.text).group(1)]
        f = next(c for c in fig["curves"] if c["label"] == "f")
        fn = sp.lambdify(X, base, "numpy")
        for x, y in [p for piece in f["pieces"] for p in piece][::40]:
            assert abs(float(fn(x * scale)) - y) < 1e-4 * (1 + abs(y)), (seed, x, y)


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier, seeds=range(25))
    sup.assert_answer_space(TOPIC, tier, distinct=20, top=0.1)


def test_equivalent_spellings_are_accepted():
    sk = sup.skill_object(SKILL)
    q = next(
        q for q in (sup.tier_question(TOPIC, "easy", s) for s in range(60)) if "x^2" in q.text and "x-axis" in q.text
    )
    assert grade(sk, q, sp.sstr(sp.expand(_typed(q.answer_text))).replace("**", "^")).correct


def test_the_order_of_the_transformations_matters():
    """Every distractor of a medium question is a different function from the key, so no op chain is interchangeable."""
    for seed in range(60):
        q = sup.tier_question(TOPIC, "medium", seed)
        key = _typed(q.answer_text)
        for d in q.distractors:
            assert not _same(_typed(d), key), (seed, d)


def test_tiers_differ():
    sup.assert_tier_shapes(TOPIC, EXAM)
    assert all(q.figure is None for q in (sup.tier_question(TOPIC, "easy", s) for s in range(20)))
    assert all(q.figure is not None for q in (sup.tier_question(TOPIC, "hard", s) for s in range(20)))


def _pair(text: str):
    return [sp.Rational(v) for v in re.findall(r"= (-?\d+(?:/\d+)?)", text)]


def _move(point, ops):
    x, y = point
    for op in ops:
        if op[0] == "tx":
            x += op[1]
        elif op[0] == "ty":
            y += op[1]
        elif op[0] == "rx":
            y = -y
        elif op[0] == "ry":
            x = -x
        elif op[0] == "dy":
            y *= op[1]
        else:
            x *= op[1]
    return [sp.nsimplify(x), sp.nsimplify(y)]


def test_exam_keys_match_an_independent_derivation():
    seen = set()
    for seed in range(70):
        q = sup.exam_question(EXAM, seed)
        g = _image(q.text)
        a, b, c, d = q.parts
        assert _same(_typed(a.answer_text), g), (seed, q.text, a.answer_text)
        gs = sp.simplify(g)
        if "turning point" in b.prompt:
            (xt,) = [r for r in sp.solve(sp.diff(g, X), X) if r.is_real]
            assert _pair(b.answer_text) == [xt, sp.simplify(g.subs(X, xt))], (seed, q.text)
        elif "inflection" in b.prompt:
            (xt,) = [r for r in sp.solve(sp.diff(g, X, 2), X) if r.is_real]
            assert _pair(b.answer_text) == [xt, sp.simplify(g.subs(X, xt))], (seed, q.text)
        elif "horizontal" in b.prompt:
            # SymPy 1.14 answers oo for the limit of 3*3**(2*x - 4) as x -> -oo (it is 0); a positive variable avoids the slip
            t = sp.Symbol("t", positive=True)
            limits = {sp.limit(g.subs(X, t), t, sp.oo), sp.limit(g.subs(X, -t), t, sp.oo)} - {sp.oo, -sp.oo}
            assert sp.Rational(b.answer_text) in limits, (seed, q.text)
        elif "vertical" in b.prompt:
            (xv,) = sp.solve(sp.denom(sp.together(gs)), X)
            assert sp.Rational(b.answer_text) == xv, (seed, q.text)
        else:
            assert "greatest value" in b.prompt, b.prompt
            xs = np.linspace(-20, 20, 400001)
            top = float(np.max(sp.lambdify(X, g, "numpy")(xs)))
            assert abs(float(sp.Rational(b.answer_text)) - top) < 1e-3, (seed, q.text)
        seen.add(b.prompt.split(".")[0][:30])
        assert sp.Rational(c.answer_text) == g.subs(X, 0), (seed, q.text)
        if "period" in d.prompt:
            assert sp.simplify(_typed(d.answer_text) - sp.periodicity(g, X)) == 0, (seed, q.text)
        else:
            px, py = [_num(v) for v in re.search(r"P\((-?[\d/]+), (-?[\d/]+)\)", d.prompt).groups()]
            moved = _move((px, py), _ops(q.text))
            assert _pair(d.answer_text) == moved, (seed, q.text)
            assert sp.simplify(g.subs(X, moved[0]) - moved[1]) == 0
    assert len(seen) >= 4, seen


def test_exam_shape_and_grading():
    for seed in range(40):
        q = sup.exam_question(EXAM, seed)
        assert [p.label for p in q.parts] == list("abcd")
        assert [p.marks for p in q.parts][::2] == [2, 1] and q.parts[3].marks in (1, 2)
        assert [bool(p.follow_through) for p in q.parts] == [False, False, True, False]
        assert q.parts[0].answer_kind == "expression"
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM, seeds=range(20))
    sup.assert_exam_answer_space(EXAM, distinct=60, top=0.05)


def test_exam_part_c_follows_through_from_a():
    for seed in range(10):
        q = sup.exam_question(EXAM, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        wrong = {**canon, "a": "2*x + 7", "c": "7"}
        assert sup.part_marks(SKILL, q, wrong, "c") == q.parts[2].marks
        assert sup.part_marks(SKILL, q, {**wrong, "c": "8"}, "c") == 0
