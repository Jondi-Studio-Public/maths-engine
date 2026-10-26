"""Tests for `slope-field-readings`: the practice tiers (`sc_slope_field`) and the exam chain (`sc_slope_field_chain`).

The oracle reads the equation, the point, the readings or the window back out of the rendered text and recomputes
the answer with SymPy (and, for the exam chain, with mpmath's ODE solver) rather than with the generator's helpers.
"""

from __future__ import annotations

import re
from fractions import Fraction

import mpmath
import pytest
import sympy

import r4e_support as t
from mathgen import make
from mathgen.text_quality import prompt_shape
from r4e_support import sup

SKILL = "slope-field-readings"
CHAIN = "sc_slope_field_chain"
x, y, k = sympy.symbols("x y k")
SYMS = dict(x=x, y=y, k=k)
INT = r"(-?\d+)"


def rhs(text):
    return sup.parse_poly(text, **SYMS)


def at(expr, px, py):
    return expr.subs({x: sympy.nsimplify(px), y: sympy.nsimplify(py)})


def fmt(value):
    value = sympy.nsimplify(value)
    if value.q == 1:
        return str(int(value))
    return sup.dp(value, len(str(value.q)) if False else _places(value))


def _places(value):
    n = 0
    while (value * 10**n).q != 1:
        n += 1
    return n


def same(want, got):
    return Fraction(str(want)) == Fraction(got)


def oracle_easy(q):
    f = re.search(r"dy/dx = ([^,.]+)", q.text).group(1)
    m = re.search(rf"x = {INT} and y = {INT}", q.text)
    if m:
        px, py = m.groups()
    else:
        px, py = re.findall(rf"\({INT}, {INT}\)", q.text)[-1]
    return fmt(at(rhs(f), px, py))


def oracle_medium(q):
    text = q.text
    if "Readings from a slope field" in text or "Its segments have a slope of" in text:
        with_c = "ax + by + c" in text
        readings = [(int(s), int(a), int(b)) for s, a, b in re.findall(rf"{INT} at \({INT}, {INT}\)", text)]
        qx, qy = [int(v) for v in re.findall(rf"\({INT}, {INT}\)", text)[-1]]
        a, b, c = sympy.symbols("a b c")
        eqs = [a * px + b * py + (c if with_c else 0) - s for s, px, py in readings]
        sol = sympy.solve(eqs, [a, b, c] if with_c else [a, b], dict=True)[0]
        return fmt(sol[a] * qx + sol[b] * qy + (sol[c] if with_c else 0))
    if "value of k" in text or "What is k" in text:
        f = re.search(r"dy/dx = ([^,.]+)", text).group(1)
        px, py = re.search(rf"\({INT}, {INT}\)", text).groups()
        if "horizontal" in text:
            s = 0
        elif "parallel" in text:
            coef = re.search(r"y = (-?\d*)x \+ 4", text).group(1)
            s = {"": 1, "-": -1}.get(coef) or int(coef)
        else:
            s = int(re.search(rf"slope {INT}\.", text).group(1))
        sol = sympy.solve(sympy.Eq(at(rhs(f), px, py), s), k)
        return fmt(sol[0])
    f = re.search(r"dy/dx = ([^,.]+)", text).group(1)
    p = int(re.search(rf"x = {INT}", text.split("dy/dx = ")[1].split(f)[-1]).group(1))
    return fmt(sympy.solve(rhs(f).subs(x, p), y)[0])


def oracle_hard(q):
    text = q.text
    if "path" in text:
        f = re.search(r"dy/dx = ([^,.]+)", text).group(1)
        x0, y0 = re.search(rf"(?:starts at the point|Starting at) \({INT}, {INT}\)", text).groups()
        h = sympy.nsimplify(re.search(r"(?:moves|goes) ([\d.]+)", text).group(1), rational=True)
        n = 1 if "one step" in text else 2
        px, py = sympy.nsimplify(x0), sympy.nsimplify(y0)
        expr = rhs(f)
        for _ in range(n):
            py = py + h * at(expr, px, py)
            px = px + h
        return fmt(py)
    f = re.search(r"dy/dx = ([^,.]+)", text).group(1)
    xlo, xhi = [int(v) for v in re.search(rf"{INT} ≤ x ≤ {INT}", text).groups()]
    ylo, yhi = [int(v) for v in re.search(rf"{INT} ≤ y ≤ {INT}", text).groups()]
    m = 0 if "horizontal" in text else int(re.search(rf"slope {INT}", text).group(1))
    expr = rhs(f)
    return str(sum(1 for px in range(xlo, xhi + 1) for py in range(ylo, yhi + 1) if at(expr, px, py) == m))


@pytest.mark.parametrize("tier, oracle", [("easy", oracle_easy), ("medium", oracle_medium), ("hard", oracle_hard)])
def test_tier(tier, oracle):
    t.assert_tier(SKILL, tier, seeds=range(50), oracle=oracle)


def test_tiers_differ_in_shape():
    t.assert_tiers_differ(SKILL)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_answer_space(tier):
    t.assert_answer_space(SKILL, tier, distinct=20, top=0.12)


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_prompt_variety(tier):
    shapes = {prompt_shape(t.question(SKILL, tier, s)) for s in range(120)}
    assert len(shapes) >= 3, shapes


@pytest.mark.parametrize("tier", ["easy", "medium", "hard"])
def test_wording_lint(tier):
    for seed in range(120):
        text = t.question(SKILL, tier, seed).text
        assert "*" not in text and "**" not in text, text
        assert not re.search(r"\b1 parts\b", text), text
        assert "  " not in text and "?." not in text, text


def test_every_hard_form_is_reached():
    kinds = set()
    for seed in range(80):
        text = t.question(SKILL, "hard", seed).text
        kinds.add("count" if "How many" in text or "At how many" in text else "walk")
    assert kinds == {"count", "walk"}


def test_medium_forms_all_reached():
    kinds = set()
    for seed in range(120):
        text = t.question(SKILL, "medium", seed).text
        kinds.add(
            "table"
            if ("Readings" in text or "Its segments" in text)
            else ("k" if "value of k" in text or "What is k" in text else "horizontal")
        )
    assert kinds == {"table", "k", "horizontal"}


def test_unknown_tier_raises():
    with pytest.raises(ValueError):
        make("sc_slope_field", seed=1, difficulty="easy", tier="bogus")
    with pytest.raises(ValueError):
        make(CHAIN, seed=1, difficulty="exam", tier="bogus")


def test_registered():
    sources = {s["difficulty"]: s for s in sup.skill_yaml(SKILL)["sources"]}
    assert set(sources) == {"easy", "medium", "hard", "exam"}
    sup.assert_registered(SKILL, CHAIN)


def test_chain_shape():
    sup.assert_chain_shape(CHAIN, ["number"] * 6, [1, 1, 1, 2, 3, 2], [0, 1, 1, 1, 0, 1])


def test_chain_text_tags_and_grading():
    sup.assert_clean_and_deterministic(CHAIN)
    sup.assert_tagged(CHAIN)
    sup.assert_grading(SKILL, CHAIN, seeds=range(25))
    sup.assert_shapes(SKILL, CHAIN)
    sup.assert_answer_space(CHAIN, distinct=100, top=0.05)


def test_chain_key_matches_independent_derivation():
    mpmath.mp.dps = 30
    for seed in range(30):
        q = sup.exam_question(CHAIN, seed)
        f = rhs(re.search(r"dy/dx = ([^,.]+)", q.stem).group(1))
        x0, y0 = [sympy.Integer(v) for v in re.search(rf"(?:The point|through) \({INT}, {INT}\)", q.stem).groups()]
        h = sympy.nsimplify(re.search(r"([\d.]+) (?:across|wide)", q.stem).group(1), rational=True)
        slope = sympy.lambdify((x, y), f, "mpmath")
        s0 = at(f, x0, y0)
        y1 = y0 + h * s0
        s1 = at(f, x0 + h, y1)
        y2 = y1 + h * s1
        solve = mpmath.odefun(slope, float(x0), float(y0))
        exact = solve(float(x0 + 2 * h))
        got = {p.label: p.answer_text for p in q.parts}
        assert same(sympy.nsimplify(got["a"]), s0) if False else sympy.nsimplify(got["a"]) == s0, (seed, got)
        assert sympy.nsimplify(got["b"]) == y1 and sympy.nsimplify(got["c"]) == s1 and sympy.nsimplify(got["d"]) == y2
        assert abs(mpmath.mpf(got["e"]) - exact) <= mpmath.mpf("0.0005001"), (seed, got["e"], exact)
        assert abs(mpmath.mpf(got["f"]) - abs(exact - float(y2))) <= mpmath.mpf("0.0011"), (seed, got)
