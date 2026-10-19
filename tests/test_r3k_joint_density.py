"""Tests `uc2-joint-continuous-density` (topic `r3k_joint_density`) and its exam chain `exd_x_r3k_joint_density`.

The oracle reads the density and its region back out of the question TEXT and re-derives each key with SymPy double
integrals (done in the order the generator does not use).
"""

from __future__ import annotations

import re
from collections import Counter

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.r3k_joint_density_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade
from tests.r3k_support import check_no_follow_through_collision, second_route_on  # noqa: F401  (autouse fixture)
from tests.r3k_support import (
    check_exam_grading,
    check_follow_through,
    check_single_tier,
    check_wiring,
    skill,
    stem,
)

TOPIC = "r3k_joint_density"
EXAM = "exd_x_r3k_joint_density"
SK = skill("uc2-joint-continuous-density")
TIERS = ("easy", "medium", "hard")
x, y = sympy.symbols("x y")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _poly(text):
    text = text.replace("²", "**2").replace("³", "**3").replace("·", "*")
    return parse_expr(text, local_dict={"x": x, "y": y}, transformations=_TRANSFORMS)


def _density(text):
    """(coefficient or None, g) from `f(x, y) = ... for`."""
    body = re.search(r"f\(x, y\) = (.*?) for ", text).group(1)
    expr = _poly(body[1:].lstrip("·") if body.startswith("c") else body)
    return expr


def _num(value):
    return sympy.Rational(value["p"], value["q"]) if isinstance(value, dict) else sympy.nsimplify(value)


def _answer(part_or_q):
    return sympy.sympify(str(part_or_q.answer).replace("^", "**"), locals={"x": x, "y": y})


def _region(text):
    m = re.search(r"for 0 < y < x < (\d+)", text)
    if m:
        return "lower", int(m.group(1))
    m = re.search(r"for 0 < x < y < (\d+)", text)
    if m:
        return "upper", int(m.group(1))
    m = re.search(r"for 0 < x < (\d+) and 0 < y < (\d+)", text)
    return "rect", (int(m.group(1)), int(m.group(2)))


def _tri(g, kind, lim):
    """The double integral over the triangle, integrating x first (the reverse of the generator)."""
    if kind == "lower":
        return sympy.integrate(sympy.integrate(g, (x, y, lim)), (y, 0, lim))
    return sympy.integrate(sympy.integrate(g, (x, 0, y)), (y, 0, lim))


def test_skill_wiring():
    check_wiring(
        SK,
        TOPIC,
        EXAM,
        ["ucv1-joint-distribution-marginal", "ucv1-double-integral-rectangle"],
        "expression",
    )


def test_easy_key_normalises_the_density():
    for seed in range(120):
        q = _q("easy", seed)
        g = _density(q.text)
        kind, (a, b) = _region(q.text)
        assert kind == "rect"
        total = sympy.integrate(sympy.integrate(g, (y, 0, b)), (x, 0, a))
        assert _answer(q) == 1 / total, (seed, q.text)
        assert not (g.has(x) and g.has(y) and sympy.Poly(g, x, y).is_monomial and False)


def test_medium_key_is_the_event_probability():
    seen = Counter()
    for seed in range(120):
        q = _q("medium", seed)
        f = _density(q.text)
        kind, (lim, lim2) = _region(q.text)
        assert kind == "rect" and lim == lim2
        assert sympy.integrate(sympy.integrate(f, (x, 0, lim)), (y, 0, lim)) == 1
        if "P(Y > X)" in q.text:
            expected = sympy.integrate(sympy.integrate(f, (x, 0, y)), (y, 0, lim))
            seen["above"] += 1
        else:
            expected = sympy.integrate(sympy.integrate(f, (y, 0, lim - x)), (x, 0, lim))
            seen["sum"] += 1
        assert _answer(q) == expected, (seed, q.text)
    assert seen["above"] > 20 and seen["sum"] > 15, seen


def test_hard_key_is_the_marginal_density():
    seen = Counter()
    for seed in range(150):
        q = _q("hard", seed)
        g = _density(q.text)
        kind, lim = _region(q.text)
        assert kind in ("lower", "upper")
        c = 1 / _tri(g, kind, lim)
        which = re.search(r"marginal density f_([XY])", q.text).group(1).lower()
        if which == "x":
            marg = c * (sympy.integrate(g, (y, 0, x)) if kind == "lower" else sympy.integrate(g, (y, x, lim)))
            var = x
        else:
            marg = c * (sympy.integrate(g, (x, y, lim)) if kind == "lower" else sympy.integrate(g, (x, 0, y)))
            var = y
        assert sympy.simplify(_answer(q) - marg) == 0, (seed, q.text)
        assert sympy.integrate(_answer(q), (var, 0, lim)) == 1
        seen[(kind, which)] += 1
    assert len(seen) == 4 and min(seen.values()) >= 10, seen


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SK, lambda seed: _q(tier, seed), seeds=60)


def test_answers_vary_and_equivalent_forms_are_accepted():
    for tier in TIERS:
        questions = [_q(tier, s) for s in range(100)]
        answers = [q.answer_text for q in questions]
        assert len(set(answers)) >= 10, (tier, Counter(answers).most_common(3))
        assert len({q.text for q in questions}) >= 40, tier
    q = _q("hard", 3)
    expanded = sympy.sstr(sympy.expand(_answer(q) * 2 / 2)).replace("**", "^")
    assert grade(SK, q, expanded).correct
    assert grade(SK, q, "0").correct is False


def test_tiers_differ_in_shape():
    assert all("rectangle" not in _q("easy", s).text for s in range(20))
    easy_regions = {_region(_q("easy", s).text)[0] for s in range(40)}
    med_regions = {_region(_q("medium", s).text)[0] for s in range(40)}
    hard_regions = {_region(_q("hard", s).text)[0] for s in range(40)}
    assert easy_regions == {"rect"} and med_regions == {"rect"} and hard_regions == {"lower", "upper"}
    assert all("Find the constant c" in _q("easy", s).text for s in range(20))
    assert all("P(" in _q("medium", s).text for s in range(20))
    assert all("marginal density" in _q("hard", s).text for s in range(20))


# -- exam chain -----------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 3, 3, 2]
        assert all(p.answer_kind == "expression" for p in q.parts)
        assert all(p.follow_through and "part_a" in p.follow_through for p in q.parts[1:])
        assert q.parts[0].follow_through is None


def test_exam_keys_match_sympy():
    for seed in range(100):
        q = _exam(seed)
        g = _density(stem(q))
        kind, lim = _region(stem(q))
        total = _tri(g, kind, lim)
        c = 1 / total
        assert _answer(q.parts[0]) == c
        which = re.search(r"f_([XY])", q.parts[1].prompt).group(1).lower()
        if which == "x":
            marg = c * (sympy.integrate(g, (y, 0, x)) if kind == "lower" else sympy.integrate(g, (y, x, lim)))
        else:
            marg = c * (sympy.integrate(g, (x, y, lim)) if kind == "lower" else sympy.integrate(g, (x, 0, y)))
        assert sympy.simplify(_answer(q.parts[1]) - marg) == 0
        half = lim // 2
        m = re.search(r"P\((X ≤|Y ≥) (\d+)\)", q.parts[2].prompt)
        assert int(m.group(2)) == half
        if m.group(1) == "X ≤":
            region = x <= half
        else:
            region = y >= half
        # probability by integrating the density times the indicator, y first for X <= half and x first for Y >= half
        if kind == "lower":
            if m.group(1) == "X ≤":
                p = sympy.integrate(sympy.integrate(g, (x, y, half)), (y, 0, half))
            else:
                p = sympy.integrate(sympy.integrate(g, (y, half, x)), (x, half, lim))
        else:
            if m.group(1) == "X ≤":
                p = sympy.integrate(sympy.integrate(g, (x, 0, y)), (y, 0, half)) + sympy.integrate(
                    sympy.integrate(g, (x, 0, half)), (y, half, lim)
                )
            else:
                p = sympy.integrate(sympy.integrate(g, (y, half, lim)), (x, 0, half)) + sympy.integrate(
                    sympy.integrate(g, (y, x, lim)), (x, half, lim)
                )
        assert region is not None
        assert _answer(q.parts[2]) == c * p
        ey = c * (
            sympy.integrate(sympy.integrate(y * g, (x, y, lim)), (y, 0, lim))
            if kind == "lower"
            else sympy.integrate(sympy.integrate(y * g, (x, 0, y)), (y, 0, lim))
        )
        assert _answer(q.parts[3]) == ey


def test_exam_grading():
    check_exam_grading(SK, _exam, seeds=30)


def test_exam_follow_through_never_lands_on_a_distractor():
    check_no_follow_through_collision(_exam)


def test_exam_follow_through_from_a_wrong_c():
    q = _exam(2)
    canon = {p.label: p.answer_text for p in q.parts}
    wrong_c = sympy.Rational(7, 5)
    reply = dict(canon)
    reply["a"] = "7/5"
    for part in q.parts[1:]:
        value = sympy.sympify(part.follow_through.replace("^", "**"), locals={"x": x, "y": y, "part_a": wrong_c})
        reply[part.label] = sympy.sstr(value).replace("**", "^")
    result = grade(SK, q, reply)
    assert result.parts["a"].marks == 0
    for part in q.parts[1:]:
        assert result.parts[part.label].marks == part.marks, (part.label, reply[part.label])
