"""Exam tier of `senior-apply-the-product-quotient-and-chain-rule-to`: product, quotient and chain rules combined."""

import random

import exd_x_c3_support as sup
from sympy import diff, lambdify, symbols
from sympy.parsing.sympy_parser import parse_expr

SKILL = "senior-apply-the-product-quotient-and-chain-rule-to"
TOPIC = "exd_x_c3_product_quotient_chain"
x = symbols("x")
_NAMES = {"x": x}


def _parse(text: str):
    return parse_expr(text.replace("·", "*").replace("^", "**"), local_dict=_NAMES)


def test_registered():
    # The skill's first exam source is the multi-part chain; this single-answer preset is its second exam source,
    # re-added by the maintainer under key exam-single (bf2fca7).
    exam = [s for s in sup.skill(SKILL)["sources"] if s["difficulty"] == "exam"]
    assert [s["topic"] for s in exam] == ["exd_x_r2a_rules_tangent", TOPIC]
    assert all(s["stages"] == ["exam"] for s in exam)
    assert exam[-1]["key"] == "exam-single"


def test_text_and_determinism():
    sup.assert_clean_text_and_deterministic(TOPIC)


def test_shapes():
    sup.assert_shapes(SKILL, TOPIC)


def test_key_matches_independent_derivative_of_the_rendered_prompt():
    rng = random.Random(7)
    points = [rng.uniform(0.3, 1.4) for _ in range(3)]
    for seed in range(300):
        q = sup.exam_question(TOPIC, seed)
        body = q.text.removeprefix("Differentiate ").removesuffix(" with respect to x.")
        expected = lambdify(x, diff(_parse(body), x))
        got = lambdify(x, _parse(q.answer_text))
        for p in points:
            assert abs(expected(p) - got(p)) < 1e-9 * (1 + abs(expected(p))), (seed, q.text, q.answer_text)


def test_grader():
    sup.assert_grader(SKILL, TOPIC)
