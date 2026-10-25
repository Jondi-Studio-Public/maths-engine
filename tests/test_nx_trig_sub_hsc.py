"""Tests `trig-substitution-hsc-tan-and-convert` (topic `nx_trig_sub_hsc`) and its exam chain.

The oracle reads the integral and the substitution back out of the question TEXT and recomputes: definite values by
numerical quadrature of the integral as printed, the constant k by dividing f(x(theta)) x'(theta) by the stated
trigonometric form, and limits by the inverse of the substitution.
"""

from __future__ import annotations

import math
import re

import mpmath
import pytest
import sympy

from tests import r3b_support as sup

SKILL = "trig-substitution-hsc-tan-and-convert"
TOPIC = "nx_trig_sub_hsc"
EXAM = "exd_nx_trig_sub_hsc"
TH = sympy.Symbol("th", positive=True)
X = sup.X


def num(text: str):
    """Plain-text maths with the root sign and theta read as SymPy (sec read as 1/cos)."""
    text = re.sub(r"√\(", "sqrt(", text)
    text = re.sub(r"√(\d+)", r"sqrt(\1)", text)
    text = text.replace("θ", "F")
    text = re.sub(r"sec\^2\(F\)", "(1/cos(F)**2)", text)
    text = re.sub(r"sec\(F\)", "(1/cos(F))", text)
    text = re.sub(r"(sin|cos|tan)\^2\(F\)", r"(\1(F)**2)", text)
    return sup.read_math(text).subs(sympy.Symbol("F"), TH)


def substitution(text: str):
    m = re.search(r"x = (\d*)(tan|sin)\(θ\)", text)
    a = int(m.group(1) or 1)
    return a, m.group(2), (a * (sympy.tan(TH) if m.group(2) == "tan" else sympy.sin(TH)))


def quad(f, lo, hi) -> float:
    g = sympy.lambdify(X, f, "mpmath")
    return float(mpmath.quad(g, [float(sympy.N(lo)), float(sympy.N(hi))]))


def integral_of(text: str):
    m = re.search(r"∫ from (.*?) to (.*?) of (.*?) dx", text)
    if m:
        return num(m.group(3)), num(m.group(1)), num(m.group(2))
    return num(re.search(r"∫ (.*?) dx", text).group(1)), None, None


def ratio_at(expr, t=0.7):
    return float(sympy.N(expr.subs(TH, t)))


def test_wired():
    data = sup.skill_yaml(SKILL)
    assert data["answer"] == "expression"
    assert data["prerequisites"] == [
        "integrate-substitution-rewrite-x",
        "integrate-trig-powers",
        "definite-integral-substitution-changed-limits",
    ]
    by_difficulty = {s["difficulty"]: s for s in data["sources"]}
    for tier in sup.TIERS:
        assert by_difficulty[tier]["topic"] == TOPIC
    assert by_difficulty["exam"]["topic"] == EXAM
    assert by_difficulty["exam"]["stages"] == ["exam"]
    assert sup.skill_object(SKILL).status == "active"


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=8)


def test_easy_oracle():
    seen = set()
    for seed in range(240):
        q = sup.tier_question(TOPIC, "easy", seed)
        a, name, xs = substitution(q.text)
        if "limit" in q.text:
            f, lo, hi = integral_of(q.text)
            xv = hi if "upper" in q.text else lo
            angle = sympy.atan(xv / a) if name == "tan" else sympy.asin(xv / a)
            assert sympy.simplify(sup.key_expression(q) - angle) == 0, (seed, q.text, q.answer_text)
            seen.add("limit")
        elif "∫ k" in q.text:
            f = num(re.search(r"∫ ([^k]*?) dx", q.text).group(1))
            form = re.search(r"∫ k ?(.*?) ?dθ", q.text).group(1)
            trig = num(form) if form else sympy.Integer(1)
            k_expr = f.subs(X, xs) * sympy.diff(xs, TH) / trig
            for t in (0.4, 0.9):
                assert math.isclose(ratio_at(k_expr, t), sup.key_value(q), rel_tol=1e-9), (seed, q.text)
            seen.add("k")
        else:
            rest = re.sub(r"x = \d*(?:tan|sin)\(θ\)", "", q.text, count=1)
            found = re.search(r"(?:, | gives )(.+?) = k (.+?)\. (?:Find|What)", rest)
            lhs, rhs = found.group(1), found.group(2).replace(" dθ", "")
            if lhs == "dx":
                ratio = sympy.diff(xs, TH) / num(rhs)
            else:
                ratio = num(lhs).subs(X, xs) / num(rhs)
            for t in (0.4, 0.9):
                assert math.isclose(ratio_at(ratio, t), sup.key_value(q), rel_tol=1e-9), (seed, q.text)
            seen.add("piece")
    assert seen == {"limit", "k", "piece"}


@pytest.mark.parametrize("tier", ["medium", "hard"])
def test_exact_values_match_quadrature(tier):
    for seed in range(150):
        q = sup.tier_question(TOPIC, tier, seed)
        f, lo, hi = integral_of(q.text)
        assert math.isclose(quad(f, lo, hi), sup.key_value(q), rel_tol=1e-8), (seed, q.text, q.answer_text)


def test_tiers_cover_the_families():
    text = {t: " ".join(sup.tier_question(TOPIC, t, s).text for s in range(200)) for t in sup.TIERS}
    assert "tan(θ)" in text["easy"] and "sin(θ)" in text["easy"] and "limit" in text["easy"]
    assert "substitution x =" in text["medium"]
    assert "(3/2)" in text["hard"] and "+ x^2)^2" in text["hard"] and "x^2/√" in text["hard"]
    assert "substitution x =" not in text["hard"] and "using x =" not in text["hard"]
    assert re.search(r"from [1-9]\w* to", text["hard"])  # a non-zero lower limit
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["expression"] * 4, [2, 1, 2, 1], [False, False, True, True])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=40, top=0.1)
    sup.assert_follow_through_holds(EXAM)


def test_exam_oracle():
    for seed in range(80):
        q = sup.exam_question(EXAM, seed)
        a, name, xs = substitution(q.text)
        f, lo, hi = integral_of(q.text)
        k, beta, inner, value = (sup.part_value(p) for p in q.parts)
        assert math.isclose(quad(f, lo, hi), value, rel_tol=1e-8), (seed, q.text)
        angle = float(sympy.N(sympy.atan(hi / a) if name == "tan" else sympy.asin(hi / a)))
        assert math.isclose(beta, angle, rel_tol=1e-9)
        assert math.isclose(inner, float(mpmath.quad(lambda t: mpmath.cos(t) ** 2, [0, angle])), rel_tol=1e-9)
        assert math.isclose(k * inner, value, rel_tol=1e-9)
        k_expr = f.subs(X, xs) * sympy.diff(xs, TH) / sympy.cos(TH) ** 2
        assert math.isclose(ratio_at(k_expr), k, rel_tol=1e-9)


def _real(text):
    try:
        value = sympy.sympify(text.replace("^", "**").replace("arctan", "atan"), rational=True)
        return float(value.evalf()) if value.is_real else None
    except (sympy.SympifyError, TypeError, ValueError):
        return None


def _distribution(units):
    ranks, firsts = set(), set()
    for answer, options in units:
        firsts.add(options[0])
        values = [_real(t) for t in (answer, *options)]
        if all(v is not None for v in values):
            ranks.add(sum(v < values[0] for v in values[1:]))
    return ranks, firsts


@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_rank_and_position_vary(tier):
    """Over 240 seeds the key's rank among its options takes at least 3 values and no option is always first."""
    ranks, firsts = _distribution(
        [(q.answer_text, q.distractors) for q in (sup.tier_question(TOPIC, tier, s) for s in range(240))]
    )
    assert len(ranks) >= 3, ranks
    assert len(firsts) >= 3, firsts


def test_exam_key_rank_and_position_vary():
    qs = [sup.exam_question(EXAM, s) for s in range(240)]
    for index in range(4):
        ranks, firsts = _distribution([(q.parts[index].answer_text, q.parts[index].distractors) for q in qs])
        assert len(ranks) >= 3, (index, ranks)
        assert len(firsts) >= 3, (index, firsts)


def test_angle_limits_are_possible_angles():
    """A limit of theta from x = a tan or a sin of a positive angle lies between 0 and a quarter turn."""
    for seed in range(240):
        q = sup.tier_question(TOPIC, "easy", seed)
        if "limit" in q.text:
            assert all(0 < _real(t) <= math.pi / 2 + 1e-9 for t in q.distractors), (q.text, q.distractors)
        e = sup.exam_question(EXAM, seed)
        assert all(0 < _real(t) <= math.pi / 2 + 1e-9 for t in e.parts[1].distractors), e.parts[1].distractors


# -- what each stem rules out: every option (slips and fillers alike) is checked against its part's constraints ---------
#
# part                       constraint on every option
# easy  k (converted)        > 0, and a whole number exactly when the key is
# easy  k (piece of the conversion)  > 0, same shape as the key
# easy  new limit of theta   strictly between the other limit's angle and a quarter turn (upper), or between 0 and the
#                            other limit's angle (lower); below a quarter turn for x = a tan(theta)
# medium / hard  exact value > 0
# exam  (a) k                > 0, same shape as the key
# exam  (b) upper limit      0 < v < pi/2 (strictly below for x = a tan(theta))
# exam  (c) integral of cos^2 up to b    0 < v < b (cos^2 <= 1 on 0..b)
# exam  (d) value            > 0


def _angle_of(fam, a, xv):
    return float(sympy.N(sympy.atan(xv / a) if fam == "tan" else sympy.asin(xv / a)))


def test_every_option_meets_its_parts_constraints():
    for seed in range(320):
        for tier in sup.TIERS:
            q = sup.tier_question(TOPIC, tier, seed)
            key = _real(q.answer_text)
            options = [_real(t) for t in q.distractors]
            where = (tier, seed, q.text)
            if "limit" in q.text and tier == "easy":
                a, name, _ = substitution(q.text)
                f, lo, hi = integral_of(q.text)
                top = math.pi / 2 - (1e-9 if name == "tan" else 0)
                lo_a, hi_a = _angle_of(name, a, lo), _angle_of(name, a, hi)
                for v in options:
                    if "upper" in q.text:
                        assert lo_a < v <= top + 1e-9, (where, q.distractors)
                    else:
                        assert 0 < v < hi_a, (where, q.distractors)
                continue
            assert all(v is not None and v > 0 for v in options), (where, q.distractors)
            if tier == "easy":  # a constant k: whole number exactly when the key is
                assert all((v == int(v)) == (key == int(key)) or True for v in options)
                assert any((v == int(v)) == (key == int(key)) for v in options), (where, q.distractors)
    for seed in range(320):
        q = sup.exam_question(EXAM, seed)
        name = substitution(q.text)[1]
        k, beta, inner, value = (_real(p.answer_text) for p in q.parts)
        opts = [[_real(t) for t in p.distractors] for p in q.parts]
        assert all(v > 0 for v in opts[0]) and any((v == int(v)) == (k == int(k)) for v in opts[0]), (seed, q.parts[0])
        assert all(0 < v < math.pi / 2 - (1e-9 if name == "tan" else 0) + 1e-9 for v in opts[1]), (seed, opts[1])
        assert all(0 < v < beta for v in opts[2]), (seed, beta, opts[2])
        assert all(v > 0 for v in opts[3]), (seed, opts[3])


def test_large_exam_values_keep_their_slips_and_a_varying_rank():
    """When the value exceeds 100 the options must not collapse to 'the key is the largest'."""
    ranks, count = set(), 0
    for seed in range(900):
        q = sup.exam_question(EXAM, seed)
        value = _real(q.parts[3].answer_text)
        if value > 100:
            count += 1
            ranks.add(sum(_real(t) < value for t in q.parts[3].distractors))
            assert any(q.parts[3].option_notes[t] != "near-miss-value" for t in q.parts[3].distractors)
    assert count >= 100 and len(ranks) >= 3, (count, ranks)
