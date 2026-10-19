"""`ucv1-taylor-polynomial`: Taylor and Maclaurin coefficients of order 2 to 4, and approximating with the polynomial.

The oracle reads the function (and the centre, the power and the point) out of the question's own text and recomputes the
coefficients with SymPy's series; it never imports the generator module.
"""

from __future__ import annotations

import re
from fractions import Fraction

import pytest
import sympy

from mathgen import make
from tests.fy1so_support import (
    SEEDS,
    check_chain,
    check_follow_through,
    check_single,
    check_tiers_differ,
    parse_text,
)

TOPIC = "fy1so_taylor_polynomial"
EXAM = "fy1so_exd_taylor_polynomial"
x = sympy.Symbol("x")
POWERS = {"²": 2, "³": 3, "⁴": 4}


def _frac(value) -> Fraction:
    value = sympy.Rational(value)
    return Fraction(int(value.p), int(value.q))


def _oracle(text: str) -> Fraction:
    """The coefficient a coefficient question asks for, recomputed from its wording."""
    shifted = re.search(r"coefficient of \(x - (\d+)\)([²³⁴]) in the Taylor .* of f\(x\) = (.+) about x = (\d+)", text)
    if shifted:
        a, power, body = int(shifted.group(1)), POWERS[shifted.group(2)], shifted.group(3)
        assert int(shifted.group(4)) == a
        f = parse_text(body)
        return _frac(sympy.diff(f, x, power).subs(x, a) / sympy.factorial(power))
    plain = re.search(r"coefficient of x([²³⁴]) in the Maclaurin .* of f\(x\) = (.+)\.$", text)
    assert plain, text
    power, f = POWERS[plain.group(1)], parse_text(plain.group(2))
    series = sympy.series(f, x, 0, power + 1).removeO()
    return _frac(series.coeff(x, power))


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_tier_answers_and_distractors(difficulty):
    check_single(TOPIC, difficulty, "fraction")


def test_tiers_differ_in_shape():
    pools = check_tiers_differ(TOPIC)
    assert all("about x" not in t for t in pools["easy"] | pools["hard"])
    assert all("about x" in t and "Taylor" in t for t in pools["medium"])
    assert all("Maclaurin" in t for t in pools["easy"] | pools["hard"])
    # hard asks for a product, a quotient or a function of x squared; easy never does
    assert all(re.search(r"f\(x\) = [^ ]+( |/)", t) is None or "x²" in t or "/" in t or ")" in t for t in pools["hard"])
    assert not any("x²)" in t or "x²" in t.split("f(x) =")[1] for t in pools["easy"])


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_oracle_agrees_with_every_answer(difficulty):
    for seed in SEEDS:
        q = make(TOPIC, seed=seed, difficulty=difficulty)
        assert Fraction(q.answer_text) == _oracle(q.text), (seed, q.text, q.answer_text)


def test_exam_chain_shape_and_marks():
    check_chain(EXAM, ["fraction", "fraction", "fraction", "number"], [1, 1, 2, 2])
    check_follow_through(EXAM)


def test_exam_oracle():
    for seed in SEEDS:
        q = make(EXAM, seed=seed, difficulty="exam")
        match = re.search(r"f\(x\) = (.+?), and let P\(x\)", q.stem)
        f = parse_text(match.group(1))
        x0 = sympy.Rational(re.search(r"find P\(([\d.]+)\)", q.text).group(1))
        poly = sympy.series(f, x, 0, 4).removeO()
        a, b, c, d = q.parts
        assert Fraction(a.answer_text) == _frac(poly.coeff(x, 2))
        assert Fraction(b.answer_text) == _frac(poly.coeff(x, 3))
        assert Fraction(c.answer_text) == _frac(poly.subs(x, x0))
        error = abs(sympy.N(f.subs(x, x0), 40) - sympy.N(poly.subs(x, x0), 40))
        places = len(d.answer_text.split(".")[1])
        assert abs(sympy.N(d.answer_text, 40) - error) <= sympy.Rational(1, 2 * 10**places), (
            seed,
            d.answer_text,
            error,
        )
        assert f"correct to {places} decimal places" in d.prompt
        assert 0 < error < sympy.Rational(1, 10)


def test_exam_follow_through_from_wrong_coefficients():
    """A wrong x² coefficient carried into the polynomial value, and that value carried into the error."""
    from mathgen.grading import grade

    done = 0
    for seed in range(30):
        q = make(EXAM, seed=seed, difficulty="exam")
        match = re.search(r"f\(x\) = (.+?), and let P\(x\)", q.stem)
        f = parse_text(match.group(1))
        x0 = sympy.Rational(re.search(r"find P\(([\d.]+)\)", q.text).group(1))
        poly = sympy.series(f, x, 0, 4).removeO()
        wrong_a = sympy.nsimplify(Fraction(q.parts[0].distractors[0]))
        wrong_c = poly.subs(x, x0) + (wrong_a - poly.coeff(x, 2)) * x0**2
        wrong_d = abs(sympy.N(f.subs(x, x0), 40) - sympy.N(wrong_c, 40))
        places = len(q.parts[3].answer_text.split(".")[1])
        reply_d = f"{float(wrong_d):.{places}f}"
        result = grade(
            "fraction",
            q,
            {"a": q.parts[0].distractors[0], "b": q.parts[1].answer_text, "c": str(wrong_c), "d": reply_d},
        )
        assert result.parts["a"].marks == 0
        assert result.parts["c"].marks == 2 and result.parts["c"].follow_through is True, (seed, result.parts["c"])
        if wrong_d > 0 and abs(wrong_d - sympy.N(poly.subs(x, x0), 40)) > 0:
            assert result.parts["d"].marks == 2, (seed, reply_d)
        done += 1
    assert done == 30


def test_skill_loads_and_has_an_exam_tier():
    from mathgen.skills import load_skills

    _, skills = load_skills()
    skill = next(s for s in skills if s.id == "ucv1-taylor-polynomial")
    assert skill.status == "active" and skill.answer == "fraction"
    assert [s.difficulty for s in skill.sources] == ["easy", "medium", "hard", "exam"]
    assert skill.sources[-1].stages == ["exam"]
