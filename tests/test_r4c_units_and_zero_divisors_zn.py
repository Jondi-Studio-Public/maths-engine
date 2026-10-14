"""Tests `uc2-units-and-zero-divisors-zn`: units, zero divisors, nilpotents, idempotents and congruence solutions are
found by running through every residue, with the numbers read out of the question text."""

from __future__ import annotations

import re
from math import gcd

import mathgen.generators.r4c_units_and_zero_divisors_zn_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from tests import r4c_support as sup

TOPIC = "r4c_units_and_zero_divisors_zn"
SKILL_ID = "uc2-units-and-zero-divisors-zn"


def _modulus(text):
    return int(re.search(r"(?:\(mod |modulo |Z_)(\d+)", text).group(1))


def _zero_divisors(n):
    return [a for a in range(1, n) if any((a * b) % n == 0 for b in range(1, n))]


def _idempotents(n):
    return [e for e in range(n) if (e * e) % n == e]


def _nilpotents(n):
    return [a for a in range(n) if any(pow(a, k, n) == 0 for k in range(1, 12))]


def test_wiring():
    sup.check_wiring(SKILL_ID, TOPIC, "set")


def test_easy_nilpotents():
    for seed in range(80):
        question = sup.q(TOPIC, "easy", seed)
        n = _modulus(question.text)
        assert question.answer == _nilpotents(n), (seed, question.text)
        assert 2 <= len(question.answer) <= 14


def test_medium_congruence_solutions():
    for seed in range(80):
        question = sup.q(TOPIC, "medium", seed)
        n = int(re.search(r"(?:\(mod |modulo |Z_)(\d+)", question.text).group(1))
        a, b = (int(v) for v in re.search(r"(\d+)x [≡=] (\d+)", question.text).groups())
        assert question.answer == [x for x in range(n) if (a * x) % n == b % n], (seed, question.text)
        assert 2 <= gcd(a, n) and len(question.answer) == gcd(a, n)


def test_hard_idempotents():
    for seed in range(80):
        question = sup.q(TOPIC, "hard", seed)
        n = _modulus(question.text)
        assert question.answer == _idempotents(n), (seed, question.text)
        assert len(question.answer) >= 4


def test_keys_and_distractors():
    sup.check_keys_and_distractors(SKILL_ID, TOPIC, "set")


def test_text_and_determinism():
    sup.check_text_and_determinism(TOPIC)


def test_answer_space():
    for tier, answers_min, texts_min in (("easy", 10, 20), ("medium", 20, 40), ("hard", 15, 20)):
        answers = {sup.q(TOPIC, tier, seed).answer_text for seed in range(80)}
        texts = {sup.q(TOPIC, tier, seed).text for seed in range(80)}
        assert len(answers) >= answers_min and len(texts) >= texts_min, (tier, len(answers), len(texts))


# -- exam -------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    sup.check_exam_shape(SKILL_ID, TOPIC, "abcd", (2, 1, 2, 3), ("number", "number", "number", "set"), (0, 1, 0, 0))


def test_exam_keys_are_recomputed_from_the_text():
    for seed in range(80):
        question = sup.exam(TOPIC, seed)
        n = _modulus(question.stem)
        a, b, c, d = question.parts
        assert int(a.answer) == sum(1 for x in range(n) if gcd(x, n) == 1)
        assert int(b.answer) == len(_zero_divisors(n))
        assert int(c.answer) == len(_nilpotents(n))
        assert d.answer == _idempotents(n)


def test_exam_follow_through_for_the_zero_divisors():
    s = sup.skill(SKILL_ID)
    for seed in range(40):
        question = sup.exam(TOPIC, seed)
        n = _modulus(question.stem)
        wrong = int(question.parts[0].answer) + 3
        sent = {p.label: p.answer_text for p in question.parts} | {"a": str(wrong), "b": str(n - 1 - wrong)}
        result = grade(s, question, sent)
        assert result.parts["b"].marks == 1 and result.parts["b"].follow_through is True, (seed, sent)
