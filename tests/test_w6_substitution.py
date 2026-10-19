"""Tests the three substitution skills (topics `w6_subst_*` and their exam chains).

The oracles read the integrand, the limits and the substitution back out of the question TEXT, then differentiate the
typed key (indefinite integrals) or integrate the integrand numerically (definite integrals and chain parts); the
generators' own helpers are never used.
"""

from __future__ import annotations

import re
from collections import Counter

import mpmath
import pytest
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from tests import w7b_support as sup

X, U = sympy.symbols("x u")
_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_LOCAL = {"x": X, "u": U, "ln": sympy.log, "pi": sympy.pi, "e": sympy.E}

SKILLS = {
    "integrate-substitution-rewrite-x": ("w6_subst_rewrite_x", "exd_x_w6_subst_rewrite_x"),
    "integrate-substitution-exponential-trigonometric": ("w6_subst_exp_trig", "exd_x_w6_subst_exp_trig"),
    "definite-integral-substitution-changed-limits": ("w6_subst_definite", "exd_x_w6_subst_definite"),
}


def parse(text: str):
    """Learner-style maths text (as the questions print it) to a SymPy expression."""
    text = text.replace("√x", " sqrt(x)").replace("√", " sqrt").replace("−", "-").replace("θ", "t")
    text = re.sub(r"\be\^\(", "exp(", text)
    text = re.sub(r"\be\^([a-z])\b", r"exp(\1)", text)
    text = re.sub(r"(sin|cos|tan|sec)\^(\d+)\((\w+)\)", r"(\1(\3))**\2", text)
    text = text.replace("^", "**")
    return parse_expr(text, transformations=_TRANSFORMS, local_dict=_LOCAL)


def integrand_of(text: str):
    m = re.search(r"∫ from (.+?) to (.+?) of (.+?) dx", text)
    if m:
        return parse(m.group(3).strip()), parse(m.group(1)), parse(m.group(2))
    m = re.search(r"∫ (.+?) dx", text)
    return parse(m.group(1).strip()), None, None


def numeric(f, lo, hi) -> float:
    fn = sympy.lambdify(X, f, modules="mpmath")
    return float(mpmath.quad(fn, [float(sympy.N(lo)), float(sympy.N(hi))]))


def _defined_points(f, points=(0.37, 1.41, 2.63, 0.91, 3.17, 0.55, 1.9)):
    out = []
    for p in points:
        try:
            v = complex(f.subs(X, p).evalf())
        except (TypeError, ZeroDivisionError, ValueError):
            continue
        if abs(v.imag) < 1e-12 and abs(v.real) < 1e9:
            out.append((p, v.real))
    return out


def skill(skill_id):
    return sup.skill_of(skill_id)


def test_wiring():
    for sid, (topic, exam) in SKILLS.items():
        sk = skill(sid)
        sup.check_wiring(sk, topic, exam, "expression")
        steps = [sk.difficulty_progression[t].steps for t in ("easy", "medium", "hard", "exam")]
        assert steps == sorted(steps) and len(set(steps)) == 4, sid
        assert sk.misconceptions and sk.prerequisites


@pytest.mark.parametrize("sid", list(SKILLS))
@pytest.mark.parametrize("tier", sup.TIERS)
def test_key_accepted_and_distractors_rejected(sid, tier):
    sup.check_single_tier(skill(sid), SKILLS[sid][0], tier, seeds=60)


@pytest.mark.parametrize("sid", list(SKILLS))
def test_pools_and_shapes(sid):
    sup.check_pools(SKILLS[sid][0], min_prompts=15)


@pytest.mark.parametrize("sid", list(SKILLS))
@pytest.mark.parametrize("tier", sup.TIERS)
def test_at_least_twelve_distinct_answers_in_sixty_seeds(sid, tier):
    assert len(sup.answer_spread(SKILLS[sid][0], tier, 60)) >= 12


@pytest.mark.parametrize(
    "sid", ["integrate-substitution-rewrite-x", "integrate-substitution-exponential-trigonometric"]
)
@pytest.mark.parametrize("tier", sup.TIERS)
def test_indefinite_keys_differentiate_back_to_the_integrand_in_the_text(sid, tier):
    topic = SKILLS[sid][0]
    for seed in range(40):
        q = sup.single(topic, tier, seed)
        f, _, _ = integrand_of(q.text)
        key = parse(q.answer_text.replace(" + C", ""))
        points = _defined_points(f)
        assert len(points) >= 2, (seed, q.text)
        d = sympy.diff(key, X)
        for p, fv in points:
            got = complex(d.subs(X, p).evalf())
            assert abs(got - fv) < 1e-7 * max(1, abs(fv)), (tier, seed, q.text, q.answer_text, p, got, fv)
        assert q.answer_text.endswith("+ C")


@pytest.mark.parametrize("tier", sup.TIERS)
def test_definite_keys_match_numerical_integration_of_the_text(tier):
    topic = SKILLS["definite-integral-substitution-changed-limits"][0]
    for seed in range(40):
        q = sup.single(topic, tier, seed)
        f, lo, hi = integrand_of(q.text)
        want = numeric(f, lo, hi)
        got = float(sympy.N(parse(q.answer_text)))
        assert abs(got - want) < 1e-7 * max(1, abs(want)), (tier, seed, q.text, q.answer_text, got, want)
        assert "+ C" not in q.answer_text


def test_hard_tiers_give_no_hint_except_for_trigonometric_substitution():
    for sid in ("integrate-substitution-rewrite-x", "integrate-substitution-exponential-trigonometric"):
        topic = SKILLS[sid][0]
        for seed in range(40):
            assert "substitution" not in sup.single(topic, "hard", seed).text, (sid, seed)
        for tier in ("easy", "medium"):
            assert all("substitution" in sup.single(topic, tier, s).text for s in range(20)), (sid, tier)
    topic = SKILLS["definite-integral-substitution-changed-limits"][0]
    hard = [sup.single(topic, "hard", s).text for s in range(60)]
    assert any("x = " in t and "sin(θ)" in t for t in hard)
    assert any("substitution" not in t for t in hard)


def test_each_tier_uses_its_stated_forms():
    rx = [sup.single("w6_subst_rewrite_x", "medium", s).text for s in range(80)]
    assert any("√" in t for t in rx) and any("/(x" in t for t in rx) and any("(2x" in t or "(3x" in t for t in rx)
    rh = " ".join(sup.single("w6_subst_rewrite_x", "hard", s).text for s in range(80))
    assert "x^3" in rh and "x^5" in rh and "√x" in rh
    et = " ".join(sup.single("w6_subst_exp_trig", "hard", s).text for s in range(80))
    assert "sec^4" in et and "sec^6" in et and "sin^3(x)/cos^4(x)" in et and "cos^3(x)/sin^2(x)" in et
    em = " ".join(sup.single("w6_subst_exp_trig", "medium", s).text for s in range(80))
    assert "e^(2x)" in em and "sin(x)cos(x)" in em
    dd = " ".join(sup.single("w6_subst_definite", "hard", s).text for s in range(80))
    assert "from 0 to ln(" in dd and "from 0 to pi/" in dd and "x^2/√" in dd


def test_distractors_are_diverse_and_the_key_is_not_predictably_placed():
    for sid, (topic, _) in SKILLS.items():
        for tier in sup.TIERS:
            tags = Counter()
            for seed in range(60):
                q = sup.single(topic, tier, seed)
                tags.update(q.option_notes[d] for d in q.distractors)
            assert len(tags) >= 4, (sid, tier, tags)


# -- exam chains -------------------------------------------------------------------------------------------------------


def test_rewrite_exam_chain_shape_and_grading():
    sid = "integrate-substitution-rewrite-x"
    sup.check_exam_chain(skill(sid), SKILLS[sid][1], [1, 2, 1, 3], ["expression", "expression", "number", "expression"])


def test_rewrite_exam_chain_parts_match_an_independent_derivation():
    for seed in range(40):
        q = sup.exam(SKILLS["integrate-substitution-rewrite-x"][1], seed)
        f, lo, hi = integrand_of(q.text)
        a = parse(re.search(r"u = (x [+-] \d+)", q.text).group(1)) - X
        x_of_u, g, upper, value = (parse(p.answer_text) for p in q.parts)
        assert sympy.simplify(x_of_u.subs(U, X + a) - X) == 0, (seed, q.text)
        assert upper == hi + a, (seed, q.text)
        lower = lo + a
        fn_g = sympy.lambdify(U, g, modules="mpmath")
        assert abs(float(mpmath.quad(fn_g, [float(lower), float(upper)])) - numeric(f, lo, hi)) < 1e-7, (seed, q.text)
        assert abs(float(sympy.N(value)) - numeric(f, lo, hi)) < 1e-7, (seed, q.text)


def test_exp_trig_exam_chain_shape_and_grading():
    sid = "integrate-substitution-exponential-trigonometric"
    sk, exam = skill(sid), SKILLS[sid][1]
    families = set()
    for seed in range(60):
        q = sup.exam(exam, seed)
        assert [p.label for p in q.parts] == ["a", "b", "c"] and [p.marks for p in q.parts] == [1, 2, 3]
        assert sup.find_text_defects(q) == []
        canon = {p.label: p.answer_text for p in q.parts}
        assert sup.grade(sk, q, q.answer_text).correct
        for p in q.parts:
            assert len(p.distractors) >= 2
            for text in p.distractors:
                assert sup.grade(sk, q, {**canon, p.label: text}).parts[p.label].marks == 0, (seed, p.label, text)
                assert p.option_notes[text] in sup.CATALOGUE
        families.add("tan" if "tan(x)" in q.text else "exp")
    assert families == {"tan", "exp"}


def test_exp_trig_exam_chain_parts_match_an_independent_derivation():
    for seed in range(40):
        q = sup.exam(SKILLS["integrate-substitution-exponential-trigonometric"][1], seed)
        f, lo, hi = integrand_of(q.text)
        sub = parse(re.search(r"u = (.+?)\.", q.text).group(1))
        upper, g, value = (parse(p.answer_text) for p in q.parts)
        assert sympy.simplify(upper - sub.subs(X, hi)) == 0, (seed, q.text)
        lower = sub.subs(X, lo)
        fn_g = sympy.lambdify(U, g, modules="mpmath")
        want = numeric(f, lo, hi)
        assert abs(float(mpmath.quad(fn_g, [float(sympy.N(lower)), float(sympy.N(upper))])) - want) < 1e-7, (
            seed,
            q.text,
        )
        assert abs(float(sympy.N(value)) - want) < 1e-7, (seed, q.text)


def test_trig_substitution_exam_chain_shape_grading_and_follow_through():
    sid = "definite-integral-substitution-changed-limits"
    sk, exam = skill(sid), SKILLS[sid][1]
    sup.check_exam_chain(sk, exam, [1, 2, 3], ["number", "number", "expression"])
    # a wrong upper limit in (a) and a wrong k in (b) still earn (c) when (c) uses them correctly
    checked = 0
    for seed in range(40):
        q = sup.exam(exam, seed)
        canon = {p.label: p.answer_text for p in q.parts}
        a_text = q.text
        sign = -1 if "x^2/√" in a_text else 1
        wrong_a, wrong_b = 90 - int(canon["a"]), int(canon["b"]) + 3
        angle = sympy.pi * wrong_a / 180
        wrong_value = wrong_b * (angle + sign * sympy.sin(2 * angle) / 2)
        result = sup.grade(sk, q, {"a": str(wrong_a), "b": str(wrong_b), "c": str(sympy.simplify(wrong_value))})
        assert result.parts["c"].marks == 3 and result.parts["c"].follow_through is True, (seed, wrong_value)
        checked += 1
    assert checked == 40


def test_trig_substitution_exam_chain_parts_match_an_independent_derivation():
    for seed in range(40):
        q = sup.exam(SKILLS["definite-integral-substitution-changed-limits"][1], seed)
        f, lo, hi = integrand_of(q.text)
        a = int(re.search(r"x = (\d+)sin", q.text).group(1))
        degrees, k, value = int(q.parts[0].answer_text), int(q.parts[1].answer_text), parse(q.parts[2].answer_text)
        assert abs(float(sympy.sin(sympy.pi * degrees / 180) * a - sympy.N(hi))) < 1e-9, (seed, q.text)
        assert k * 2 == a * a, (seed, q.text)
        assert abs(float(sympy.N(value)) - numeric(f, lo, hi)) < 1e-7, (seed, q.text)


@pytest.mark.parametrize("sid", list(SKILLS))
def test_exam_chains_have_at_least_twelve_distinct_answer_sets_in_sixty_seeds(sid):
    exam = SKILLS[sid][1]
    seen = {tuple(p.answer_text for p in sup.exam(exam, s).parts) for s in range(60)}
    assert len(seen) >= 12
