"""Tests for the multi-part exam source `exd_x_r3g_quadratic_sensitivity` (minimiser, minimum, perturbation)."""

from __future__ import annotations

import re
from fractions import Fraction

import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from tests import exd_x_r3g_support as sup

TOPIC = "exd_x_r3g_quadratic_sensitivity"
SKILL = "ucv1-optimality-sensitivity"
X = sympy.Symbol("x")
_T = standard_transformations + (implicit_multiplication_application,)


def _text(value) -> str:
    value = sympy.Rational(value)
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


def _facts(q):
    body = re.search(r"f\(x\) = (.+?)(?:, defined| is to be|\. (?:A|The))", q.text).group(1)
    f = parse_expr(body.replace("²", "**2"), transformations=_T, local_dict={"x": X})
    new_b = int(re.search(r"to (-?\d+)", q.text).group(1))
    return f, new_b


def _minimum(f):
    (xstar,) = sympy.solve(sympy.diff(f, X), X)
    return xstar, f.subs(X, xstar)


def _derive(q):
    f, new_b = _facts(q)
    a, b, c = sympy.Poly(f, X).all_coeffs()
    xstar, vmin = _minimum(f)
    g = a * X**2 + new_b * X + c
    _, vnew = _minimum(g)
    estimate = vmin + (new_b - b) * xstar
    return {"a": _text(xstar), "b": _text(vmin), "c": _text(vnew), "d": _text(vnew - estimate)}


def test_wired():
    sup.check_wired(SKILL, TOPIC)


def test_variety_and_new_shapes():
    sup.check_variety(TOPIC, "p17a_optimality_sensitivity", {})


def test_clean_and_deterministic():
    sup.check_clean_and_deterministic(TOPIC)


def test_shape():
    for q in sup.questions(TOPIC, 40):
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["fraction"] * 4
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, True]


def test_keys_match_independent_derivation_over_300_seeds():
    bad = []
    for q in sup.questions(TOPIC):
        if sup.canonical(q) != _derive(q):
            bad.append((q.text, sup.canonical(q), _derive(q)))
    assert bad == [], bad[:2]


def test_the_estimate_is_never_exact():
    for q in sup.questions(TOPIC, 100):
        assert Fraction(sup.canonical(q)["d"]) < 0


def test_answer_space():
    sup.check_answer_space(TOPIC)


def test_grading_and_format():
    sup.check_grading(SKILL, TOPIC)


def test_tags_are_catalogue_ids():
    sup.check_tagged(TOPIC)


def test_follow_through_from_a_wrong_minimiser():
    def replies(q):
        f, new_b = _facts(q)
        a, b, c = sympy.Poly(f, X).all_coeffs()
        wrong = sympy.Rational(q.parts[0].distractors[0])
        if wrong == 0:
            return None
        v = f.subs(X, wrong)
        vnew = v - sympy.Rational(2 * b * (new_b - b) + (new_b - b) ** 2, 4 * a)
        err = vnew - v - (new_b - b) * wrong
        reply = {"a": _text(wrong), "b": _text(v), "c": _text(vnew), "d": _text(err)}
        if len(set(reply.values())) < 4 or 0 in (v, vnew, err):
            return None
        return reply, {"b", "c", "d"}, {"a"}

    sup.check_follow_through(SKILL, TOPIC, replies)


def test_follow_through_from_a_wrong_new_minimum():
    def replies(q):
        f, new_b = _facts(q)
        b = sympy.Poly(f, X).all_coeffs()[1]
        xstar, vmin = _minimum(f)
        for option in q.parts[2].distractors:
            wrong = sympy.Rational(option)
            err = wrong - vmin - (new_b - b) * xstar
            reply = {**sup.canonical(q), "c": _text(wrong), "d": _text(err)}
            if len(set(reply.values())) == 4 and err != 0:
                return reply, {"d"}, {"c"}
        return None

    sup.check_follow_through(SKILL, TOPIC, replies)
