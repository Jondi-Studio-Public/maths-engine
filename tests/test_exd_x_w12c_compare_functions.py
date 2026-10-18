"""Tests the exam chain for comparing functions: a drawn parabola f against a vertex-form rule g.

The oracle reads f from the drawn curve (the extreme sample is the turning point; the drop one column across gives
the leading coefficient) and g from the rule in the stem, then re-derives every part from those two functions with
SymPy, never from the key.
"""

from __future__ import annotations

import re

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import figures, make
from tests.exd_x_r2c_support import (
    canonical,
    check_answer_space,
    check_clean_and_deterministic,
    check_follow_through,
    check_grading,
    check_shapes,
    check_wired,
    exam,
)

TOPIC = "exd_x_w12c_compare_functions"
SKILL_ID = "compare-functions-representations"
HARD_TOPIC = "fg1_compare_functions"
HARD_KWARGS = {"tier": "hard"}
x = sympy.Symbol("x")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _parse(text: str):
    return parse_expr(text.replace("^", "**"), transformations=_TRANSFORMS, local_dict={"x": x})


def read_f(fig):
    """f(x) as an expression, from the sampled curve."""
    samples = {x_: y for piece in fig["curves"][0]["pieces"] for x_, y in map(tuple, piece)}
    ys = list(samples.values())
    opens_up = samples[min(samples)] > min(ys)
    k = min(ys) if opens_up else max(ys)
    h = next(xx for xx, yy in samples.items() if yy == k and xx == int(xx))
    a = samples[h + 1] - k
    return a * (x - int(h)) ** 2 + int(k)


def oracle(q):
    f = read_f(q.figure)
    g = _parse(re.search(r"g\(x\) = (.+?)\.(?: |$)", q.stem).group(1))
    p, qq = (int(v) for v in re.search(r"Between x = (-?\d+) and x = (-?\d+)", q.parts[2].prompt).groups())
    a_f, a_g = sympy.Poly(f, x).coeff_monomial(x**2), sympy.Poly(g, x).coeff_monomial(x**2)
    vertex = lambda e: sympy.solve(sympy.diff(e, x), x)[0]  # noqa: E731
    k_f, k_g = f.subs(x, vertex(f)), g.subs(x, vertex(g))
    rate = lambda e: (e.subs(x, qq) - e.subs(x, p)) / (qq - p)  # noqa: E731
    roots = sorted(sympy.solve(sympy.Eq(f, g), x))
    return f, g, abs(k_f - k_g), abs(rate(f) - rate(g)), roots, (a_f * a_g > 0)


@pytest.fixture(scope="module")
def questions():
    return [exam(TOPIC, s) for s in range(320)]


def test_clean_and_deterministic():
    check_clean_and_deterministic(TOPIC)


def test_shape(questions):
    for q in questions[:60]:
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.answer_kind for p in q.parts] == ["expression", "number", "number", "set"]
        assert [p.marks for p in q.parts] == [2, 1, 2, 2]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True, False]
        assert all(p.sensible_range is not None for p in q.parts if p.answer_kind == "number")


def test_figure_is_a_parabola_and_hides_the_rule(questions):
    for q in questions[:100]:
        assert q.figure is not None and figures.validate(q.figure) == []
        assert len(q.figure["curves"]) == 1 and not q.figure["lines"]
        figures.assert_no_leak(q.figure, [q.parts[0].answer_text])


def test_maximum_and_minimum_versions_both_occur(questions):
    words = {("maximum" if "maximum" in q.parts[1].prompt else "minimum") for q in questions[:100]}
    assert words == {"maximum", "minimum"}


def test_oracle_agrees_over_300_seeds(questions):
    bad = []
    for q in questions[:300]:
        f, g, k_gap, rate_gap, roots, same_way = oracle(q)
        a_t, b_t, c_t, d_t = (p.answer_text for p in q.parts)
        shown = sorted(int(v) for v in re.findall(r"-?\d+", d_t))
        ok = (
            sympy.expand(_parse(a_t) - f) == 0
            and sympy.Rational(b_t) == k_gap
            and k_gap > 0
            and sympy.Rational(c_t) == rate_gap
            and rate_gap > 0
            and shown == roots
            and len(roots) == 2
            and same_way
        )
        if not ok:
            bad.append((q.stem, canonical(q)))
    assert bad == [], bad[:2]


def test_grading_and_distractors():
    check_grading(SKILL_ID, TOPIC, 120)


def test_follow_through_from_a_wrong_turning_point_height():
    """A turning point one unit further from g's gives a larger gap in (b); the rate in (c) is unchanged."""

    def carried(q):
        f, g, k_gap, rate_gap, _roots, _ = oracle(q)
        a_f = sympy.Poly(f, x).coeff_monomial(x**2)
        h = sympy.solve(sympy.diff(f, x), x)[0]
        k = f.subs(x, h)
        k_g = g.subs(x, sympy.solve(sympy.diff(g, x), x)[0])
        step = 1 if k > k_g else -1
        wrong = f"{a_f}*(x - ({h}))^2 + ({k + step})"
        return "a", wrong, {"b": str(k_gap + 1), "c": str(rate_gap)}

    check_follow_through(SKILL_ID, TOPIC, carried, 60)


def test_follow_through_from_a_wrong_leading_coefficient():
    """A rule with the wrong leading coefficient (and, where the interval is symmetric about the turning point, a
    turning point one column out) is carried through the heights in (b) and the rates of change in (c)."""

    def carried(q):
        f, g, k_gap, rate_gap, _roots, _ = oracle(q)
        p, qq = (int(v) for v in re.search(r"Between x = (-?\d+) and x = (-?\d+)", q.parts[2].prompt).groups())
        h = sympy.solve(sympy.diff(f, x), x)[0]
        k = f.subs(x, h)
        a_f = sympy.Poly(f, x).coeff_monomial(x**2)
        k_g = g.subs(x, sympy.solve(sympy.diff(g, x), x)[0])
        rate_g = (g.subs(x, qq) - g.subs(x, p)) / (qq - p)
        sign_k = 1 if k > k_g else -1
        sign_r = 1 if ((f.subs(x, qq) - f.subs(x, p)) / (qq - p)) > rate_g else -1
        for a2, h2 in [(a2, h2) for h2 in (h, h + 1, h - 1) for a2 in sorted(range(-8, 9), key=lambda v: abs(v - a_f))]:
            if a2 in (0, a_f):
                continue
            f2 = a2 * (x - h2) ** 2 + k
            b_val = sign_k * (f2.subs(x, h) - k_g)
            c_val = sign_r * ((f2.subs(x, qq) - f2.subs(x, p)) / (qq - p) - rate_g)
            if (b_val == k_gap or b_val > 1) and c_val > 1 and c_val not in (b_val, k_gap, rate_gap):
                return "a", f"{a2}*(x - ({h2}))^2 + ({k})", {"b": str(b_val), "c": str(c_val)}
        raise AssertionError("no usable wrong rule")

    check_follow_through(SKILL_ID, TOPIC, carried, 60)
