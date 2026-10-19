"""Tests `uc2-irreducibility-test-over-zp`: each polynomial is read out of the question text and factored again with
SymPy over Z_p; the degree lists must match, and the patterns each tier promises must hold."""

from __future__ import annotations

import re

from sympy import Poly, factor_list, symbols

import mathgen.generators.r4c_irreducibility_test_over_zp_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from tests import r4c_support as sup

TOPIC = "r4c_irreducibility_test_over_zp"
SKILL_ID = "uc2-irreducibility-test-over-zp"
x = symbols("x")


def parse(text):
    total = 0
    for sign, coef, var, power in re.findall(r"([+-]?)\s*(\d*)(x(?:\^(\d+))?)?(?=\s|$)", text.strip()):
        if not coef and not var:
            continue
        c = int(coef) if coef else 1
        c = -c if sign == "-" else c
        total += c * (x ** (int(power) if power else 1) if var else 1)
    return total


def _read(text):
    poly = re.search(r"f = ([\dx^+ ]+)", text).group(1).strip()
    return parse(poly), int(re.search(r"Z_(\d+)", text).group(1))


def degrees(f, p):
    _, factors = factor_list(f, x, modulus=p)
    return sorted(Poly(g, x).degree() for g, m in factors for _ in range(m))


def test_wiring():
    sup.check_wiring(SKILL_ID, TOPIC, "ordered_list")


def test_easy_has_a_linear_factor_and_degree_three_or_four():
    seen = set()
    for seed in range(80):
        question = sup.q(TOPIC, "easy", seed)
        f, p = _read(question.text)
        assert p in (3, 5, 7)
        assert question.answer == degrees(f, p), (seed, question.text)
        assert question.answer[0] == 1 and sum(question.answer) in (3, 4)
        seen.add(tuple(question.answer))
    assert len(seen) >= 4


def test_medium_degree_four_or_five_over_small_fields():
    seen = set()
    for seed in range(80):
        question = sup.q(TOPIC, "medium", seed)
        f, p = _read(question.text)
        assert p in (2, 3, 5)
        assert question.answer == degrees(f, p), (seed, question.text)
        seen.add(tuple(question.answer))
    assert len(seen) >= 6


def test_hard_degree_six_and_root_free_cases():
    seen, rootfree = set(), 0
    for seed in range(120):
        question = sup.q(TOPIC, "hard", seed)
        f, p = _read(question.text)
        assert p in (2, 3) and Poly(f, x).degree() == 6
        assert question.answer == degrees(f, p), (seed, question.text)
        seen.add(tuple(question.answer))
        rootfree += question.answer[0] > 1
    assert len(seen) >= 6 and rootfree >= 10


def test_keys_and_distractors():
    sup.check_keys_and_distractors(SKILL_ID, TOPIC, "ordered_list")


def test_text_and_determinism():
    sup.check_text_and_determinism(TOPIC)


def test_answer_space():
    for tier, answers_min, texts_min in (("easy", 4, 40), ("medium", 6, 40), ("hard", 6, 50)):
        answers = {sup.q(TOPIC, tier, seed).answer_text for seed in range(100)}
        texts = {sup.q(TOPIC, tier, seed).text for seed in range(100)}
        assert len(answers) >= answers_min and len(texts) >= texts_min, (tier, len(answers), len(texts))


# -- exam -------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    sup.check_exam_shape(
        SKILL_ID, TOPIC, "abcd", (2, 1, 3, 2), ("number", "number", "ordered_list", "number"), (0, 1, 0, 0)
    )


def test_exam_keys_are_recomputed_from_the_text():
    for seed in range(80):
        question = sup.exam(TOPIC, seed)
        f, p = _read(question.stem)
        a, b, c, d = question.parts
        _, factors = factor_list(f, x, modulus=p)
        expected = degrees(f, p)
        assert c.answer == expected, (seed, question.stem)
        assert int(a.answer) == expected.count(1)
        assert int(b.answer) == 5 - expected.count(1)
        assert int(d.answer) == len(factors)


def test_exam_follow_through_for_the_remaining_degree():
    s = sup.skill(SKILL_ID)
    for seed in range(40):
        question = sup.exam(TOPIC, seed)
        wrong = (
            int(question.parts[0].answer) + 1
            if int(question.parts[0].answer) < 4
            else int(question.parts[0].answer) - 1
        )
        sent = {p.label: p.answer_text for p in question.parts} | {"a": str(wrong), "b": str(5 - wrong)}
        result = grade(s, question, sent)
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (seed, sent)
