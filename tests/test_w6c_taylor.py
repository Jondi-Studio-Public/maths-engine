"""Tests `uc2-multivariable-taylor-hessian` (topic `w6c_taylor`) and its exam chain `exd_x_w6c_taylor`.

The oracle reads the function and the point back out of the question TEXT, parses the function with SymPy and
differentiates it symbolically (`diff`, `hessian`), so the generator's closed-form jets are checked against computer
algebra, not against themselves.
"""

from __future__ import annotations

import re
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    split_symbols,
    standard_transformations,
)

import mathgen.generators.w6c_taylor_build  # noqa: F401  (registers the topics)
from mathgen import make
from tests import w6c_support as support

TOPIC = "w6c_taylor"
EXAM = "exd_x_w6c_taylor"
SKILL = support.skill_named("uc2-multivariable-taylor-hessian")
X, Y, K = sympy.symbols("x y k")
_TRANSFORMS = standard_transformations + (split_symbols, implicit_multiplication_application, convert_xor)
_FUNCTION = re.compile(r"f\(x, y\) = (.+?)(?: about| using| to second|\.? (?:The point|At the)| has|, which|, with|\. )")
_ABOUT = re.compile(r"about (?:the point )?\((-?\d+), (-?\d+)\)")
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def _function(text):
    raw = _FUNCTION.search(text.replace("−", "-")).group(1)
    raw = re.sub(r"([a-z)])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: m.group(1) + "**" + m.group(2).translate(_SUPERSCRIPTS), raw)
    raw = re.sub(r"\be\^\(", "exp(", raw).replace("√", "sqrt").replace("ln(", "log(")
    return parse_expr(raw, local_dict={"x": X, "y": Y, "k": K}, transformations=_TRANSFORMS)


def _point(text):
    a, b = _ABOUT.search(text.replace("−", "-")).groups()
    return int(a), int(b)


def _rational(text):
    f = Fraction(text)
    return sympy.Rational(f.numerator, f.denominator)


def _derivatives(f, a, b):
    at = {X: a, Y: b}
    return {
        "v": f.subs(at),
        "fx": sympy.diff(f, X).subs(at),
        "fy": sympy.diff(f, Y).subs(at),
        "fxx": sympy.diff(f, X, 2).subs(at),
        "fxy": sympy.diff(f, X, Y).subs(at),
        "fyy": sympy.diff(f, Y, 2).subs(at),
    }


def _t2(d, h, k):
    h, k = sympy.nsimplify(h), sympy.nsimplify(k)
    return d["v"] + d["fx"] * h + d["fy"] * k + (d["fxx"] * h * h + 2 * d["fxy"] * h * k + d["fyy"] * k * k) / 2


def test_skill_wiring():
    support.check_wiring(SKILL, TOPIC, EXAM, ["ucv1-hessian-2x2", "ucv1-taylor-polynomial"])


def test_each_tier_key_accepted_and_every_distractor_rejected():
    for tier in support.TIERS:
        support.check_tier(SKILL, TOPIC, tier)


def test_easy_key_is_a_coefficient_of_the_degree_two_polynomial():
    seen = set()
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="easy")
        f, (a, b) = _function(q.text), _point(q.text)
        d = _derivatives(f, a, b)
        shown = re.search(r"coefficient of (.+?)(?: in the|\?|\.$)", q.text).group(1)
        if "x" in shown and "y" in shown:
            expected, kind = d["fxy"], "xy"
        elif "x" in shown:
            expected, kind = d["fxx"] / 2, "xx"
        else:
            expected, kind = d["fyy"] / 2, "yy"
        seen.add(kind)
        assert _rational(q.answer_text) == expected, (seed, q.text, kind)
    assert seen == {"xx", "yy", "xy"}


def test_medium_key_is_the_second_order_polynomial_at_a_nearby_point():
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="medium")
        f, (a, b) = _function(q.text), _point(q.text)
        x1, y1 = re.search(r"(?:f|T₂)\((-?\d[\d.]*), (-?\d[\d.]*)\)", q.text.replace("−", "-")).groups()
        d = _derivatives(f, a, b)
        assert _rational(q.answer_text) == _t2(d, sympy.nsimplify(x1) - a, sympy.nsimplify(y1) - b), (seed, q.text)
        # an honest estimate: the polynomial is close to the function itself
        true = float(f.subs({X: sympy.nsimplify(x1), Y: sympy.nsimplify(y1)}))
        assert abs(true - float(_rational(q.answer_text))) <= 0.2 * max(1.0, abs(true)), (seed, q.text)


def _classify(hess, k):
    fxx, fxy, fyy = hess[0, 0].subs(K, k), hess[0, 1].subs(K, k), hess[1, 1].subs(K, k)
    det = fxx * fyy - fxy * fxy
    if det < 0:
        return "saddle"
    if det == 0:
        return "inconclusive"
    return "minimum" if fxx > 0 else "maximum"


def test_hard_key_comes_from_the_hessian_at_the_origin():
    kinds = set()
    for seed in range(120):
        q = make(TOPIC, seed=seed, difficulty="hard")
        f = _function(q.text)
        hess = sympy.hessian(f, (X, Y)).subs({X: 0, Y: 0})
        assert sympy.diff(f, X).subs({X: 0, Y: 0}) == 0 and sympy.diff(f, Y).subs({X: 0, Y: 0}) == 0
        if "saddle" in q.text:
            expected = next(k for k in range(1, 100) if _classify(hess, k) == "saddle")
            kinds.add("saddle")
        elif "inconclusive" in q.text:
            expected = next(k for k in range(1, 100) if _classify(hess, k) == "inconclusive")
            assert len([k for k in range(1, 100) if _classify(hess, k) == "inconclusive"]) == 1
            kinds.add("inconclusive")
        else:
            wanted = "minimum" if "minimum" in q.text else "maximum"
            expected = sum(1 for k in range(-100, 101) if _classify(hess, k) == wanted)
            kinds.add("count")
        assert int(q.answer_text) == expected, (seed, q.text)
    assert kinds == {"saddle", "inconclusive", "count"}


def test_tier_shapes_and_answer_space():
    support.check_tier_shapes_differ(TOPIC)
    for tier in support.TIERS:
        support.check_answer_space(TOPIC, tier, distinct=5)


def test_exam_shape():
    q = make(EXAM, seed=3, difficulty="exam")
    assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
    assert q.parts[2].answer_kind == "mcq" and len(q.parts[2].distractors) == 3


def test_exam_keys_match_independent_derivation():
    seen = set()
    for seed in range(60):
        q = make(EXAM, seed=seed, difficulty="exam")
        f = _function(q.stem)
        a, b = (int(v) for v in re.search(r"P = \((-?\d+), (-?\d+)\)", q.stem.replace("−", "-")).groups())
        d = _derivatives(f, a, b)
        assert (d["fx"], d["fy"]) == (0, 0)
        det = d["fxx"] * d["fyy"] - d["fxy"] ** 2
        assert _rational(q.parts[0].answer_text) == d["fxx"]
        assert _rational(q.parts[1].answer_text) == det
        verdict = "a saddle point" if det < 0 else ("a local minimum" if d["fxx"] > 0 else "a local maximum")
        seen.add(verdict)
        assert q.parts[2].answer_text == verdict
        x1, y1 = re.search(r"estimate f\((-?\d[\d.]*), (-?\d[\d.]*)\)", q.parts[3].prompt.replace("−", "-")).groups()
        assert _rational(q.parts[3].answer_text) == _t2(d, sympy.nsimplify(x1) - a, sympy.nsimplify(y1) - b)
    assert seen == {"a saddle point", "a local minimum", "a local maximum"}


def test_exam_grading_and_follow_through():
    support.check_exam_grading(SKILL, EXAM, (1, 2, 2, 3))
    support.check_follow_through(SKILL, EXAM)
