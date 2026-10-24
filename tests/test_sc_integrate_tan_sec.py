"""Tests `integrate-tan-sec-powers`: keys differentiated back or integrated numerically against the printed stem."""

from __future__ import annotations

import math
import re

import pytest
import sympy

from tests import r3b_support as sup

SKILL = "integrate-tan-sec-powers"
TOPIC = "sc_integrate_tan_sec"
EXAM = "sc_integrate_tan_sec_exam"
_SEC_POWER = re.compile(r"sec\^(\d+)\(([^()]*)\)")
_SEC = re.compile(r"sec\(([^()]*)\)")


def readable(text: str) -> str:
    """The stem's maths with sec written as 1/cos, which the shared reader and numerical checks understand."""
    text = _SEC_POWER.sub(r"((1/cos(\2))**\1)", text)
    return _SEC.sub(r"(1/cos(\1))", text)


def key_without_sec(q):
    return sup.key_expression(q).rewrite(sympy.cos)


def check_key(q):
    stem = readable(q.text)
    if "from" in q.text:
        f, lo, hi = sup.definite_parts(stem)
        exact = sup.simpson(f, float(sympy.N(lo)), float(sympy.N(hi)))
        assert math.isclose(exact, sup.key_value(q), rel_tol=1e-6, abs_tol=1e-8), (q.text, exact, q.answer_text)
    else:
        sup.assert_derivative_is(key_without_sec(q), sup.integrand(stem))


def test_wired():
    sup.assert_wired(SKILL, TOPIC, EXAM, "expression", ["integrate-trig-powers", "integration-by-substitution"])


@pytest.mark.parametrize("tier", sup.TIERS)
def test_keys_match_the_stem(tier):
    for seed in range(120):
        check_key(sup.tier_question(TOPIC, tier, seed))


@pytest.mark.parametrize("tier", sup.TIERS)
def test_clean_graded_and_varied(tier):
    sup.assert_clean_and_deterministic(TOPIC, tier)
    sup.assert_single_graded(SKILL, TOPIC, tier)
    sup.assert_answer_space(TOPIC, tier, distinct=8)


def test_tiers_differ_and_cover_the_syllabus_shapes():
    text = {t: " ".join(sup.tier_question(TOPIC, t, s).text for s in range(160)) for t in sup.TIERS}
    assert re.search(r"tan\^2\(", text["easy"]) and "sec(" in text["easy"] and "from" not in text["easy"]
    assert re.search(r"sec\^\d\(\w*\) tan\(", text["medium"]) and re.search(r"tan\^\d\(\w*\) sec\^2", text["medium"])
    assert "in terms of" in text["medium"]
    assert "tan^4(" in text["hard"] and "sec^4(" in text["hard"] and "from 0 to" in text["hard"]
    assert re.search(r"sec\^\d\(x\) tan\^3\(x\)|sec\(x\) tan\^3\(x\)", text["hard"])
    sup.assert_tier_shapes(TOPIC, EXAM)


def test_definite_values_are_exact_and_include_pi_surds_and_logs():
    values = {sup.tier_question(TOPIC, "hard", s).answer_text for s in range(300)}
    assert any("pi" in v for v in values) and any("ln(2)" in v for v in values) and any("sqrt(3)" in v for v in values)


def test_indefinite_answers_ask_for_a_form_when_two_differ_by_a_constant():
    """tan^2/2 and sec^2/2 differ by a constant, so such a stem names the function the answer is in."""
    for tier in ("medium", "hard"):
        for seed in range(160):
            q = sup.tier_question(TOPIC, tier, seed)
            if "from" in q.text:
                continue
            if re.search(r"sec\^\d\(.*?\) tan\(|tan\^\d\(.*?\) sec\^2|tan\^2\(x\) sec\^4|tan\^3\(x\)", q.text):
                assert "in terms of" in q.text, q.text


def test_exam_shape_and_grading():
    sup.assert_chain_shape(EXAM, ["expression"] * 4, [2, 2, 2, 1], [False, True, True, True])
    sup.assert_exam_clean_and_deterministic(EXAM)
    sup.assert_exam_graded(SKILL, EXAM)
    sup.assert_exam_answer_space(EXAM, distinct=40, top=0.1)
    sup.assert_follow_through_holds(EXAM)


def test_exam_keys_match_quadrature_of_each_prompt():
    for seed in range(60):
        q = sup.exam_question(EXAM, seed)
        for part in q.parts:
            f, lo, hi = sup.definite_parts(part.prompt)
            exact = sup.simpson(f, float(lo), float(hi))
            assert math.isclose(exact, sup.part_value(part), rel_tol=1e-7), (seed, part.label, part.answer_text)


def test_exam_reduction_links_the_parts():
    """I(n) = 1/(n - 1) - I(n - 2): each of (b) and (c) is the stated reduction of the part before it."""
    for seed in range(60):
        q = sup.exam_question(EXAM, seed)
        a, b, c, d = (sup.part_value(p) for p in q.parts)
        s = int(re.search(r"tan\^(\d)\(x\) dx\. ", q.parts[0].prompt + " ")[1]) if "tan^" in q.parts[0].prompt else 1
        assert math.isclose(b, 1 / (s + 1) - a, rel_tol=1e-9)
        assert math.isclose(c, 1 / (s + 3) - b, rel_tol=1e-9)
        pq = re.search(r"\((\d+) tan\^\d\(x\) \+ (\d+) tan\^\d\(x\)\)", q.parts[3].prompt)
        assert math.isclose(d, int(pq[1]) * c + int(pq[2]) * b, rel_tol=1e-9)
