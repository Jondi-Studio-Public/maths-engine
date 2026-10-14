"""Tests `ucv1-conic-sections-features` (topics `u2_conic_features` and its exam chain).

The oracle reads the conic's equation out of the question TEXT, completes the square itself with exact fractions, and
derives the requested feature from the standard form; parabolas are checked against the focus-directrix definition.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

from mathgen import make
from tests import u2_support as su

TOPIC = "u2_conic_features"
EXAM = "exd_x_u2_conic_features"
SK = su.skill("ucv1-conic-sections-features")
X, Y = sympy.symbols("x y")
_T = standard_transformations + (implicit_multiplication_application,)


def _equation(text):
    """(A, B, D, E, F) of 'A x² + B y² + D x + E y + F = 0' found in `text`, 0 for a missing term."""
    body = re.search(r"((?:-?\d*[xy]²|-?\d*[xy])[^=]*?) = 0", text).group(1)
    poly = sympy.Poly(
        sympy.expand(parse_expr(body.replace("²", "**2"), local_dict={"x": X, "y": Y}, transformations=_T)), X, Y
    )
    return tuple(Fraction(int(poly.coeff_monomial(m))) for m in (X**2, Y**2, X, Y, 1))


def _standard(coeffs):
    """(h, k, a2, b2): centre and the signed denominators A' = R/A, B' = R/B of (x-h)²/A' + (y-k)²/B' = 1."""
    A, B, D, E, F = coeffs
    h, k = -D / (2 * A), -E / (2 * B)
    R = -F + D * D / (4 * A) + E * E / (4 * B)
    return h, k, R / A, R / B


def _features(coeffs):
    h, k, pa, pb = _standard(coeffs)
    if pa > 0 and pb > 0:  # ellipse
        a2, b2 = max(pa, pb), min(pa, pb)
        c2 = a2 - b2
        vertical = pb > pa
    else:  # hyperbola: the positive denominator belongs to the transverse axis
        a2, b2 = (pa, -pb) if pa > 0 else (pb, -pa)
        c2 = a2 + b2
        vertical = pa < 0
    return dict(h=h, k=k, a2=a2, b2=b2, c2=c2, vertical=vertical, ellipse=pa > 0 and pb > 0)


def _sqrt(frac):
    root = sympy.sqrt(sympy.Rational(frac.numerator, frac.denominator))
    assert root.is_Rational, frac
    return Fraction(int(root.p), int(root.q))


def test_wiring():
    su.check_wiring(SK, TOPIC, EXAM, "number")


@pytest.mark.parametrize("tier", su.TIERS)
def test_key_and_distractors(tier):
    su.check_tier(SK, TOPIC, tier)


def test_easy_keys_by_definition():
    seen = set()
    for seed in range(150):
        q = make(TOPIC, seed=seed, difficulty="easy")
        key = Fraction(q.answer["p"], q.answer["q"]) if isinstance(q.answer, dict) else Fraction(q.answer)
        if "eccentricity" in q.text:
            d1, d2 = (int(v) for v in re.findall(r"/(\d+)", q.text))
            a2, b2 = (
                (max(d1, d2), min(d1, d2))
                if "ellipse" in q.text
                else ((d1, d2) if q.text.index("x²") < q.text.index("y²") else (d2, d1))
            )
            hyper = "hyperbola" in q.text
            if hyper:
                first_is_x = q.text.index("x²") < q.text.index("y²")
                transverse = d1 if True else d2
                a2, b2 = d1, d2
            e2 = (a2 + b2 if hyper else a2 - b2) / Fraction(a2)
            assert key * key == e2, (seed, q.text)
            seen.add("hyperbola" if hyper else "ellipse")
        else:
            var = re.search(r"parabola (\w)² = (-?\d+)(\w)", q.text)
            four_p = int(var.group(2))
            p = Fraction(four_p, 4)
            focus_asked = "focus" in q.text
            assert key == (p if focus_asked else -p), (seed, q.text)
            seen.add("parabola")
    assert seen == {"ellipse", "hyperbola", "parabola"}


def test_medium_keys_from_completed_square():
    seen = set()
    for seed in range(150):
        q = make(TOPIC, seed=seed, difficulty="medium")
        f = _features(_equation(q.text))
        assert f["ellipse"] == ("ellipse" in q.text)
        a, b, c = _sqrt(f["a2"]), _sqrt(f["b2"]), _sqrt(f["c2"])
        key = Fraction(q.answer["p"], q.answer["q"]) if isinstance(q.answer, dict) else Fraction(q.answer)
        if "latus rectum" in q.text:
            assert key == 2 * b * b / a, (seed, q.text)
            seen.add("latus")
        else:
            assert key == 2 * c, (seed, q.text)
            seen.add("foci")
    assert seen == {"latus", "foci"}


def test_hard_keys():
    seen = set()
    for seed in range(200):
        q = make(TOPIC, seed=seed, difficulty="hard")
        key = Fraction(q.answer["p"], q.answer["q"]) if isinstance(q.answer, dict) else Fraction(q.answer)
        if q.text.startswith("The parabola"):
            body = re.search(r"parabola (.*?) = 0", q.text).group(1)
            poly = sympy.Poly(
                sympy.expand(parse_expr(body.replace("²", "**2"), local_dict={"x": X, "y": Y}, transformations=_T)),
                X,
                Y,
            )
            # A point P=(x,y) on the curve is as far from the focus as from the directrix: check with SymPy.
            vertical_axis = poly.coeff_monomial(X**2) != 0
            if vertical_axis:
                coeff = poly.coeff_monomial(X**2)
                h = -sympy.Rational(int(poly.coeff_monomial(X)), 2 * int(coeff))
                four_p = -sympy.Rational(int(poly.coeff_monomial(Y)), int(coeff))
                k = (h * h * coeff + poly.coeff_monomial(X) * h + poly.coeff_monomial(1)) / -poly.coeff_monomial(Y)
                p = four_p / 4
                focus, directrix_at = (h, k + p), k - p
                px = h + 3
                py = sympy.solve(poly.as_expr().subs(X, px), Y)[0]
                assert (
                    sympy.simplify(sympy.sqrt((px - focus[0]) ** 2 + (py - focus[1]) ** 2) - abs(py - directrix_at))
                    == 0
                )
            else:
                coeff = poly.coeff_monomial(Y**2)
                k = -sympy.Rational(int(poly.coeff_monomial(Y)), 2 * int(coeff))
                four_p = -sympy.Rational(int(poly.coeff_monomial(X)), int(coeff))
                h = (k * k * coeff + poly.coeff_monomial(Y) * k + poly.coeff_monomial(1)) / -poly.coeff_monomial(X)
                p = four_p / 4
                focus, directrix_at = (h + p, k), h - p
                py = k + 3
                px = sympy.solve(poly.as_expr().subs(Y, py), X)[0]
                assert (
                    sympy.simplify(sympy.sqrt((px - focus[0]) ** 2 + (py - focus[1]) ** 2) - abs(px - directrix_at))
                    == 0
                )
            assert key == Fraction(int(sympy.Rational(directrix_at).p), int(sympy.Rational(directrix_at).q)), (
                seed,
                q.text,
            )
            seen.add("parabola")
        else:
            f = _features(_equation(q.text))
            a, c = _sqrt(f["a2"]), _sqrt(f["c2"])
            centre = f["k"] if f["vertical"] else f["h"]
            assert ("upper" in q.text) == f["vertical"]
            assert key == centre + a * a / c, (seed, q.text)
            seen.add("central")
    assert seen == {"parabola", "central"}


def test_exam_chain():
    su.check_exam(SK, EXAM, "abcd", (3, 2, 2, 3), (False, False, True, True))
    for seed in range(100):
        q = make(EXAM, seed=seed, difficulty="exam")
        f = _features(_equation(q.stem))
        a, c = _sqrt(f["a2"]), _sqrt(f["c2"])
        assert (q.parts[0].answer["x"], q.parts[0].answer["y"]) == (f["h"], f["k"])
        assert Fraction(q.parts[1].answer) == c
        e = q.parts[2].answer
        assert Fraction(e["p"], e["q"]) == c / a
        d = q.parts[3].answer
        assert Fraction(d["p"], d["q"]) == a * a / c


def test_exam_follow_through_from_a_wrong_focal_distance():
    def a_of(q):
        return _sqrt(_features(_equation(q.stem))["a2"])

    def eccentricity(q, c):
        return str(Fraction(c) / a_of(q))

    def directrix(q, c):
        return str(a_of(q) ** 2 / Fraction(c))

    def wrong_c(q):
        # A wrong c whose follow-through value is neither zero nor the same number as the wrong c itself.
        a = a_of(q)
        for c in (int(q.parts[1].answer) + 1, int(q.parts[1].answer) + 2, int(q.parts[1].answer) - 1):
            if c > 0 and a * a / c != c:
                return c
        raise AssertionError

    su.check_follow_through(SK, EXAM, "b", "c", wrong_c, eccentricity)
    su.check_follow_through(SK, EXAM, "b", "d", wrong_c, directrix)


def test_variety():
    for tier in su.TIERS:
        su.distinct_stems(TOPIC, tier, minimum=20)
    su.answer_variety(TOPIC, "hard", seeds=80, minimum=15)
    su.answer_variety(EXAM, "exam", seeds=40, minimum=6)
