"""``uc2-cauchy-riemann-harmonic-conjugate``: answers grade right, distractors wrong, each answer meets the Cauchy-Riemann equations."""

from __future__ import annotations

import re

import pytest
import sympy

from mathgen import make

from w5a_support import (
    ORACLE_SEEDS,
    X,
    Y,
    Z,
    answer_value,
    balanced_after,
    check_follow_through,
    check_grading,
    check_text,
    close,
    coverage,
    numeric,
    parse,
    questions,
    tier_topic,
)

NAME = "cauchy_riemann"
ALL = ("easy", "medium", "hard", "exam")
POINTS = (
    (sympy.Rational(3, 7), sympy.Rational(5, 9)),
    (-sympy.Rational(8, 5), sympy.Rational(2, 3)),
    (sympy.Rational(11, 6), -sympy.Rational(7, 4)),
)
K = sympy.Symbol("k")


@pytest.mark.parametrize("tier", ALL)
def test_answers_and_distractors_grade(tier):
    for q in questions(NAME, tier):
        check_grading(q)
        check_text(q)


@pytest.mark.parametrize("tier", ALL)
def test_variety_and_determinism(tier):
    qs = questions(NAME, tier, range(1, 25))
    assert len({q.text for q in qs}) >= 12
    again = make(tier_topic(NAME, tier), seed=7, difficulty=tier)
    assert again.to_dict() == make(tier_topic(NAME, tier), seed=7, difficulty=tier).to_dict()


def test_exam_chain_shape():
    for q in questions(NAME, "exam"):
        assert 3 <= len(q.parts) <= 4
        assert [p.label for p in q.parts] == list("abcd")[: len(q.parts)]
        assert any(p.follow_through for p in q.parts)
        check_follow_through(q)


def test_tiers_are_distinct():
    shapes = {t: {q.text for q in questions(NAME, t, range(1, 13))} for t in ("easy", "medium", "hard")}
    assert shapes["easy"] != shapes["medium"] != shapes["hard"]
    assert not (shapes["easy"] & shapes["hard"])


def test_single_part_tiers_have_no_parts():
    for t in ("easy", "medium", "hard"):
        assert all(not q.parts for q in questions(NAME, t, range(1, 6)))


# -- the oracle ----------------------------------------------------------------------------------------------------------


def _at(expr, x0, y0) -> complex:
    return complex(sympy.N(expr.subs({X: x0, Y: y0}), 20))


def _satisfies_cr(u, v) -> bool:
    ux, uy, vx, vy = (sympy.diff(e, s) for e, s in ((u, X), (u, Y), (v, X), (v, Y)))
    return all(abs(_at(ux - vy, *p)) < 1e-9 and abs(_at(uy + vx, *p)) < 1e-9 for p in POINTS)


def _u_of(text: str):
    for marker in ("u(x, y) = ", "Re f = ", "real part is ", "u = "):
        if marker in text:
            return parse(balanced_after(text, marker))
    raise AssertionError(text)


def _check_answer(q) -> bool:
    """Recompute the answer from the stem alone; False when the stem is of a shape this oracle does not read."""
    text = q.text
    if "f(0) = " in text and "f(z)" in text:
        u = _u_of(text)
        f = parse(q.answer_text)
        for x0, y0 in POINTS:
            value = complex(sympy.N(f.subs(Z, x0 + sympy.I * y0), 20))
            assert abs(value.real - _at(u, x0, y0).real) < 1e-8, (text, q.answer_text)
        f0 = parse(re.search(r"f\(0\) = (.+?)(?: whose|\. |\.$)", text).group(1))
        assert close(numeric(f.subs(Z, 0)), numeric(f0), 1e-9), (text, q.answer_text)
        return True
    if "differentiable" in text:
        m = re.search(r"u = (.+?) and v = (.+?)\.", text) or re.search(r"\(([^()]+)\) \+ i\(([^()]+)\)", text)
        u, v = parse(m.group(1)), parse(m.group(2))
        ux, uy, vx, vy = (sympy.diff(e, s) for e, s in ((u, X), (u, Y), (v, X), (v, Y)))
        sol = sympy.solve([ux - vy, uy + vx], [X, Y], dict=True)
        assert len(sol) == 1
        x0, y0 = sol[0][X], sol[0][Y]
        if "Find f'" in text or "value of the derivative" in text or "value of f'" in text:
            want = ux.subs({X: x0, Y: y0}) + sympy.I * vx.subs({X: x0, Y: y0})
        else:
            want = x0 + sympy.I * y0
        assert close(answer_value(q), numeric(want), 1e-9), (text, q.answer_text, want)
        return True
    if re.search(r"[^A-Za-z]k[^A-Za-z]", text) and (
        "value of" in text or "Find k" in text or "k >" in text or "positive value" in text
    ):
        u = _u_of(text)
        k = answer_value(q).real
        lap = sympy.diff(u, X, 2) + sympy.diff(u, Y, 2)
        assert all(abs(_at(lap.subs(K, sympy.nsimplify(k)), *p)) < 1e-8 for p in POINTS), (text, q.answer_text)
        other = lap.subs(K, sympy.nsimplify(k) + 1)
        assert any(abs(_at(other, *p)) > 1e-6 for p in POINTS), text  # k is determined
        return True
    if "conjugate" in text or "analytic" in text:
        u = _u_of(text)
        v = parse(q.answer_text)
        assert _satisfies_cr(u, v), (text, q.answer_text)
        if "v(0, 0) = 0" in text:
            assert abs(_at(v, 0, 0)) < 1e-9, (text, q.answer_text)
        else:
            assert abs(_at(v, 5, 0)) < 1e-9, (text, q.answer_text)  # v = 0 on the real axis
        return True
    return False


@pytest.mark.parametrize("tier", ("easy", "medium", "hard"))
def test_answer_meets_the_stem_conditions(tier):
    qs = questions(NAME, tier, ORACLE_SEEDS)
    read = sum(1 for q in qs if _check_answer(q))
    coverage(read, len(qs), 0.9)


def test_exam_parts_are_recomputed_from_the_stem():
    a, b = sympy.symbols("a b")
    for q in questions(NAME, "exam", ORACLE_SEEDS):
        u = parse(balanced_after(q.text, "u(x, y) = "))
        pa, pb = (answer_value(q.parts[i]) if False else numeric(parse(q.parts[i].answer_text)) for i in (0, 1))
        fixed = u.subs({a: sympy.nsimplify(pa.real), b: sympy.nsimplify(pb.real)})
        assert sympy.simplify(sympy.diff(fixed, X, 2) + sympy.diff(fixed, Y, 2)) == 0, q.text
        # the stem fixes a and b uniquely
        lap = sympy.expand(sympy.diff(u, X, 2) + sympy.diff(u, Y, 2))
        assert len(sympy.solve(sympy.Poly(lap, X).all_coeffs(), [a, b], dict=True)) == 1
        v = parse(q.parts[2].answer_text)
        assert _satisfies_cr(fixed, v) and abs(_at(v, 0, 0)) < 1e-12, q.text
        deriv = _at(sympy.diff(fixed, X), 1, 1) - 1j * _at(sympy.diff(fixed, Y), 1, 1)
        assert close(numeric(parse(q.parts[3].answer_text)), deriv, 1e-9), q.text
