"""Tests `uc2-transformation-of-random-variable` (topic `r3k_transformation`) and `exd_x_r3k_transformation`.

The oracle reads the density of X and the map back out of the question TEXT and checks the key by an independent
route: the probability of a sub-interval of Y, computed by integrating the density of X over the pre-image of that
interval numerically, must equal the integral of the stated f_Y over the same interval.
"""

from __future__ import annotations

import re
from collections import Counter

import mpmath
import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.r3k_transformation_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade
from tests.r3k_support import check_no_follow_through_collision, second_route_on  # noqa: F401  (autouse fixture)
from tests.r3k_support import check_exam_grading, check_single_tier, check_wiring, skill, stem

TOPIC = "r3k_transformation"
EXAM = "exd_x_r3k_transformation"
SK = skill("uc2-transformation-of-random-variable")
TIERS = ("easy", "medium", "hard")
x = sympy.Symbol("x", real=True)
y = sympy.Symbol("y", positive=True)
s = sympy.Symbol("s", positive=True)
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
mpmath.mp.dps = 25


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _parse(text, local):
    text = text.replace("²", "**2").replace("³", "**3").replace("^", "**")
    return parse_expr(text, local_dict=local, transformations=_TRANSFORMS)


def _fx_and_support(text):
    body, lo_hi = re.search(r"f_X\(x\) = (.*?) for (.+?), and 0", text).groups()
    body = body.replace("e**", "exp").replace("e^(", "exp(")
    if body.endswith("e^(-4x)") or "e^" in body:
        pass
    body = re.sub(r"(\d+)e\^\((-\d+)x\)", r"\1*exp(\2*x)", body)
    fx = _parse(body, {"x": x})
    if lo_hi.startswith("x >"):
        return fx, (0, mpmath.inf)
    m = re.match(r"(-?\d+) < x < (\d+)", lo_hi)
    return fx, (int(m.group(1)), int(m.group(2)))


def _map(text):
    g = re.search(r"(?:Let |Y = )Y = ([^.]+?)\. Find|Let Y = ([^,]+?)(?:,| \.|\. Find)", text)
    raw = (g.group(1) or g.group(2)).strip()
    if raw == "|X|":
        return abs(x)
    raw = raw.replace("√X", "sqrt(x)").replace("e^(-X)", "exp(-x)").replace("e^X", "exp(x)").replace("X", "x")
    return _parse(raw, {"x": x})


def _y_interval(text):
    m = re.search(r"for ([-\d/]+) < y < ([-\d/]+)\.", text)
    if m:
        return sympy.Rational(m.group(1)), sympy.Rational(m.group(2))
    m = re.search(r"for y > (\d+)\.", text)
    return sympy.Integer(int(m.group(1))), sympy.Integer(int(m.group(1))) + 3


def _answer(obj):
    return sympy.sympify(str(obj.answer).replace("^", "**"), locals={"y": y, "s": s})


def _probability_by_preimage(fx, support, gexpr, y1, y2):
    """P(y1 < g(X) < y2) by integrating f_X over the pre-image found by solving g(x) = y1, y2."""
    lo, hi = support
    points = {mpmath.mpf(lo)}
    if hi != mpmath.inf:
        points.add(mpmath.mpf(hi))
    for yy in (y1, y2):
        for root in sympy.solve(sympy.Eq(gexpr, yy), x):
            if root.is_real:
                points.add(mpmath.mpf(float(root)))
    cuts = sorted(p for p in points if lo <= p <= (hi if hi != mpmath.inf else mpmath.inf))
    if hi == mpmath.inf:
        cuts.append(mpmath.inf)
    f = sympy.lambdify(x, fx, "mpmath")
    g = sympy.lambdify(x, gexpr, "mpmath")
    total = mpmath.mpf(0)
    for a_, b_ in zip(cuts, cuts[1:]):
        mid = (a_ + 1) if b_ == mpmath.inf else (a_ + b_) / 2
        if y1 < g(mid) < y2:
            total += mpmath.quad(f, [a_, b_])
    return total


def _integral_of_answer(ans, y1, y2):
    return mpmath.quad(sympy.lambdify(y, ans, "mpmath"), [mpmath.mpf(float(y1)), mpmath.mpf(float(y2))])


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["uc2-joint-continuous-density"], "expression")


def _check_density_by_preimage(q, seed):
    fx, support = _fx_and_support(q.text)
    gexpr = _map(q.text)
    lo, hi = _y_interval(q.text)
    width = hi - lo
    y1, y2 = lo + width * sympy.Rational(3, 11), lo + width * sympy.Rational(8, 13)
    expected = _probability_by_preimage(fx, support, gexpr, y1, y2)
    got = _integral_of_answer(_answer(q), y1, y2)
    assert abs(expected - got) < 1e-8, (seed, q.text, expected, got)


def test_easy_key_matches_the_preimage_probability():
    signs = Counter()
    for seed in range(100):
        q = _q("easy", seed)
        _check_density_by_preimage(q, seed)
        signs[("-" in re.search(r"Y = (\S+)", q.text).group(1))] += 1
        assert _answer(q).is_polynomial(y)
    assert signs[True] > 20 and signs[False] > 20, signs


def test_medium_key_matches_the_preimage_probability():
    maps = Counter()
    for seed in range(100):
        q = _q("medium", seed)
        _check_density_by_preimage(q, seed)
        maps[re.search(r"Let Y = ([^.]+)\.", q.text).group(1)] += 1
    assert len(maps) >= 4, maps


def test_hard_key_matches_the_preimage_probability_or_the_convolution():
    seen = Counter()
    for seed in range(120):
        q = _q("hard", seed)
        if "S = X + Y" in q.text:
            a, b = (
                int(v) for v in re.search(r"uniform on \(0, (\d+)\) and Y uniform on \(0, (\d+)\)", q.text).groups()
            )
            lo, hi = (int(v) for v in re.search(r"for (\d+) < s < (\d+)\.", q.text).groups())
            s1, s2 = lo + (hi - lo) * 0.21, lo + (hi - lo) * 0.77

            def cdf_y(v):
                return min(max(v, 0), b) / b

            def inner(u, s1=s1, s2=s2, b=b):
                return cdf_y(s2 - u) - cdf_y(s1 - u)

            breaks = sorted({0, a} | {p for p in (s1 - b, s2 - b, s1, s2) if 0 < p < a})
            expected = mpmath.quad(lambda u: inner(u) / a, [mpmath.mpf(v) for v in breaks])
            got = mpmath.quad(sympy.lambdify(s, _answer(q), "mpmath"), [s1, s2])
            assert abs(expected - got) < 1e-9, (seed, q.text)
            seen["conv"] += 1
        else:
            _check_density_by_preimage(q, seed)
            seen["branches"] += 1
    assert seen["conv"] > 15 and seen["branches"] > 30, seen


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SK, lambda seed: _q(tier, seed), seeds=50)


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, seed).answer_text for seed in range(100)]
        assert len(set(answers)) >= 15, (tier, Counter(answers).most_common(3))


def test_tiers_differ_in_shape():
    assert all(re.search(r"Y = -?\d+X", _q("easy", seed).text) for seed in range(30))
    medium = " ".join(_q("medium", seed).text for seed in range(60))
    assert "X²" in medium and "√X" in medium and "e^" in medium
    hard = " ".join(_q("hard", seed).text for seed in range(60))
    assert "|X|" in hard and "S = X + Y" in hard and "-1 < x <" in hard.replace("−", "-") or "-2 < x <" in hard


# -- exam chain -----------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [3, 1, 2, 2]
        assert [p.answer_kind for p in q.parts] == ["expression", "fraction", "expression", "fraction"]
        assert q.parts[1].follow_through.startswith("Subs(part_a")
        assert q.parts[3].follow_through.endswith("- part_b")


def test_exam_keys_by_preimage_probability():
    for seed in range(80):
        q = _exam(seed)
        text = stem(q)
        fx, support = _fx_and_support(text)
        gexpr = _map(text)
        # F_Y(y0) by the pre-image, and the interval probability
        y0 = sympy.Rational(re.search(r"P\(Y ≤ ([\d/]+)\)", q.parts[1].prompt).group(1))
        pre = _probability_by_preimage(fx, support, gexpr, mpmath.mpf(0), mpmath.mpf(float(y0)))
        got = (
            mpmath.mpf(q.parts[1].answer["p"]) / q.parts[1].answer["q"]
            if isinstance(q.parts[1].answer, dict)
            else q.parts[1].answer
        )
        assert abs(pre - got) < 1e-8, (seed, text)
        y_a, y_b = (sympy.Rational(v) for v in re.search(r"P\(([\d/]+) < Y < ([\d/]+)\)", q.parts[3].prompt).groups())
        pre2 = _probability_by_preimage(fx, support, gexpr, mpmath.mpf(float(y_a)), mpmath.mpf(float(y_b)))
        got2 = mpmath.mpf(q.parts[3].answer["p"]) / q.parts[3].answer["q"]
        assert abs(pre2 - got2) < 1e-8, (seed, text)
        # F_Y is the integral of f_Y, and f_Y is its derivative
        cdf, dens = _answer(q.parts[0]), _answer(q.parts[2])
        assert sympy.simplify(sympy.diff(cdf, y) - dens) == 0
        assert cdf.subs(y, y0) == y0 * 0 + sympy.nsimplify(got) or abs(float(cdf.subs(y, y0)) - float(got)) < 1e-12


def test_exam_grading():
    check_exam_grading(SK, _exam, seeds=30)


def test_exam_follow_through_never_lands_on_a_distractor():
    check_no_follow_through_collision(_exam)


def test_exam_follow_through_from_a_wrong_cdf():
    q = _exam(3)
    canon = {p.label: p.answer_text for p in q.parts}
    wrong = q.parts[0].distractors[0]
    wrong_expr = sympy.sympify(wrong.replace("^", "**"), locals={"y": y})
    reply = dict(canon)
    reply["a"] = wrong
    y0 = re.search(r"P\(Y ≤ ([\d/]+)\)", q.parts[1].prompt).group(1)
    reply["b"] = str(sympy.nsimplify(wrong_expr.subs(y, sympy.Rational(y0))))
    y_a, y_b = re.search(r"P\(([\d/]+) < Y < ([\d/]+)\)", q.parts[3].prompt).groups()
    reply["d"] = str(sympy.nsimplify(wrong_expr.subs(y, sympy.Rational(y_b)) - sympy.sympify(reply["b"])))
    result = grade(SK, q, reply)
    assert result.parts["a"].marks == 0
    assert result.parts["b"].marks == q.parts[1].marks, reply["b"]
    assert result.parts["d"].marks == q.parts[3].marks, reply["d"]
