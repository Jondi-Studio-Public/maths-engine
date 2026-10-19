"""Tests `uc2-mgf-moments-identify` (topic `r3k_mgf`) and its exam chain `exd_x_r3k_mgf`.

The oracle reads the moment generating function(s) back out of the question TEXT and re-derives the key with SymPy:
derivatives at 0 for the moments, the coefficient of u^k in the generating function (u = e^t) for the probabilities.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy
from sympy.parsing.sympy_parser import implicit_multiplication_application, parse_expr, standard_transformations

import mathgen.generators.r3k_mgf_build  # noqa: F401  (registers the topics)
from mathgen import make
from mathgen.engine import grade
from tests.r3k_support import check_no_follow_through_collision, second_route_on  # noqa: F401  (autouse fixture)
from tests.r3k_support import as_fraction, check_exam_grading, check_single_tier, check_wiring, skill, stem

TOPIC = "r3k_mgf"
EXAM = "exd_x_r3k_mgf"
SK = skill("uc2-mgf-moments-identify")
TIERS = ("easy", "medium", "hard")
t, u = sympy.symbols("t u")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)


def _q(tier, seed):
    return make(TOPIC, seed=seed, difficulty=tier)


def _exam(seed):
    return make(EXAM, seed=seed, difficulty="exam")


def _parse(text):
    text = text.replace("e^(", "exp(").replace("e^t", "exp(t)").replace("^", "**")
    return parse_expr(text, local_dict={"t": t, "exp": sympy.exp}, transformations=_TRANSFORMS)


def _mgfs(text):
    found = re.findall(r"M(?:_[XY])?\(t\) = (.+?)(?=, valid| and M_Y|\. |\.$)", text)
    return [_parse(f) for f in found]


def _moments(mgf, k=2):
    return [sympy.nsimplify(sympy.diff(mgf, t, i).subs(t, 0)) for i in range(1, k + 1)]


def _answer(obj):
    return sympy.sympify(str(obj.answer).replace("^", "**"), locals={"t": t})


def _as_exact(value):
    return sympy.Rational(value["p"], value["q"]) if isinstance(value, dict) else sympy.Integer(value)


def _coefficient(mgf, k):
    """P(X = k) as the coefficient of u^k in the generating function (u = e^t)."""
    g = mgf.subs(sympy.exp(t), u)
    return sympy.series(g, u, 0, k + 1).removeO().coeff(u, k)


def test_skill_wiring():
    check_wiring(SK, TOPIC, EXAM, ["discrete-expected-value", "ucv1-poisson-count-probability"], "expression")


def test_easy_key_is_the_first_derivative_at_zero():
    for seed in range(120):
        q = _q("easy", seed)
        (m,) = _mgfs(q.text)
        assert sympy.nsimplify(_answer(q) - _moments(m)[0]) == 0, (seed, q.text)


def test_medium_key_is_m2_minus_m1_squared():
    for seed in range(120):
        q = _q("medium", seed)
        (m,) = _mgfs(q.text)
        d1, d2 = _moments(m)
        assert sympy.nsimplify(_answer(q) - (d2 - d1**2)) == 0, (seed, q.text)


def test_hard_keys():
    seen = Counter()
    for seed in range(160):
        q = _q("hard", seed)
        text = q.text
        if "P(X + Y" in text:
            m1, m2 = _mgfs(text)
            k = int(re.search(r"P\(X \+ Y = (\d+)\)", text).group(1))
            expected = _coefficient(
                sympy.simplify(m1 * m2)
                if "exp(" not in str(m1)
                else sympy.exp(sympy.expand_log(sympy.log(m1 * m2), force=True)),
                k,
            )
            got = _answer(q)
            assert sympy.simplify(got - expected) == 0, (seed, text, got, expected)
            seen["sum"] += 1
        elif "moment generating function M_Y(t)" in text:
            (mx,) = _mgfs(text)
            m = re.search(r"Y = (-?\d*)X ([+-]) (\d+)", text)
            a, b = (
                int({"": "1", "-": "-1"}.get(m.group(1), m.group(1))),
                int(m.group(3)) * (1 if m.group(2) == "+" else -1),
            )
            got = _answer(q)
            d1x, d2x = _moments(mx)
            d1, d2 = _moments(got)
            assert d1 == a * d1x + b and d2 == a * a * d2x + 2 * a * b * d1x + b * b, (seed, text)
            assert sympy.simplify(got - sympy.exp(b * t) * mx.subs(t, a * t)) == 0
            seen["linear"] += 1
        else:
            (m,) = _mgfs(text)
            assert _answer(q) == sympy.nsimplify(sympy.diff(m, t, 3).subs(t, 0)), (seed, text)
            seen["third"] += 1
    assert min(seen.values()) >= 15 and len(seen) == 3, seen


@pytest.mark.parametrize("tier", TIERS)
def test_key_accepted_distractors_rejected_text_and_determinism(tier):
    check_single_tier(SK, lambda seed: _q(tier, seed), seeds=60)


def test_answers_vary():
    for tier in TIERS:
        answers = [_q(tier, seed).answer_text for seed in range(100)]
        assert len(set(answers)) >= 10, (tier, Counter(answers).most_common(3))
    families = Counter()
    for seed in range(80):
        text = _q("easy", seed).text
        families[
            "poisson" if "(e^t - 1)" in text else "gamma" if "(1 - " in text and "e^t" not in text else "other"
        ] += 1
    assert families["poisson"] > 5 and families["gamma"] > 5 and families["other"] > 10, families


def test_tiers_differ_in_shape():
    assert all("Find E(X)." in _q("easy", s).text for s in range(30))
    assert all("Find Var(X)." in _q("medium", s).text for s in range(30))
    hard = " ".join(_q("hard", s).text for s in range(40))
    assert "P(X + Y" in hard and "M_Y(t) of Y" in hard and "E(X³)" in hard


# -- exam chain -----------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    for seed in range(40):
        q = _exam(seed)
        assert [p.label for p in q.parts] == ["a", "b", "c", "d"]
        assert [p.marks for p in q.parts] == [2, 3, 2, 2]
        assert q.parts[1].follow_through.endswith("- part_a**2")
        assert q.parts[3].follow_through.endswith("*part_b")


def test_exam_keys_match_sympy():
    for seed in range(100):
        q = _exam(seed)
        (m,) = _mgfs(stem(q))
        d1, d2 = _moments(m)
        assert as_fraction(q.parts[0].answer) == Fraction(int(d1.p), int(d1.q))
        var = d2 - d1**2
        assert as_fraction(q.parts[1].answer) == Fraction(int(var.p), int(var.q))
        k = int(re.search(r"P\(X = (\d+)\)", q.parts[2].prompt).group(1))
        pmf = _coefficient(m, k)
        if q.parts[2].answer_kind == "fraction":
            assert sympy.nsimplify(_as_exact(q.parts[2].answer) - pmf) == 0, (seed, stem(q))
        else:
            assert sympy.simplify(_answer(q.parts[2]) - pmf) == 0
        a = int(re.search(r"Y = (-?\d+)X", q.parts[3].prompt).group(1))
        assert as_fraction(q.parts[3].answer) == Fraction(int(var.p), int(var.q)) * a * a


def test_exam_grading():
    check_exam_grading(SK, _exam, seeds=30)


def test_exam_follow_through_never_lands_on_a_distractor():
    check_no_follow_through_collision(_exam)


def test_exam_follow_through_carries_a_wrong_mean_and_a_wrong_variance():
    q = _exam(2)
    canon = {p.label: p.answer_text for p in q.parts}
    wrong_a = Fraction(7, 3)
    m2 = sympy.sympify(q.parts[1].follow_through.split(" - ")[0])
    b_value = sympy.nsimplify(m2 - sympy.Rational(7, 3) ** 2)
    reply = {**canon, "a": "7/3", "b": str(b_value)}
    reply["d"] = str(b_value * int(q.parts[3].follow_through.split("*")[0]))
    result = grade(SK, q, reply)
    assert result.parts["a"].marks == 0
    assert result.parts["b"].marks == q.parts[1].marks, reply
    assert result.parts["d"].marks == q.parts[3].marks, reply
    assert wrong_a != as_fraction(q.parts[0].answer)
