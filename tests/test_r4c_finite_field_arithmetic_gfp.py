"""Tests `uc2-finite-field-arithmetic-gfp`: every answer is found again by searching all p residues (inverses by trial,
roots by substitution), with the numbers read out of the question text."""

from __future__ import annotations

import re

import mathgen.generators.r4c_finite_field_arithmetic_gfp_build  # noqa: F401  (registers the topics)
from mathgen.engine import grade
from tests import r4c_support as sup

TOPIC = "r4c_finite_field_arithmetic_gfp"
SKILL_ID = "uc2-finite-field-arithmetic-gfp"


def _p(text):
    """The prime, from the residue range 0 to p - 1 every phrasing states."""
    return int(re.search(r"0 to (\d+)", text).group(1)) + 1


def _inverse(a, p):
    return next(u for u in range(1, p) if (a * u) % p == 1)


def _quadratic(text):
    """'72x² + x + 53' -> (72, 1, 53)."""
    a, b, c = re.fullmatch(r"(\d+)x² \+ (\d*)x \+ (\d+)", text).groups()
    return int(a), int(b or 1), int(c)


def _roots(a, b, c, p):
    return [x for x in range(p) if (a * x * x + b * x + c) % p == 0]


def test_wiring():
    sup.check_wiring(SKILL_ID, TOPIC, "number")


def test_easy_linear_equation():
    for seed in range(80):
        question = sup.q(TOPIC, "easy", seed)
        p = _p(question.text)
        a, b, c = (int(v) for v in re.search(r"(\d+)x \+ (\d+) [=≡] (\d+)", question.text).groups())
        solutions = [x for x in range(p) if (a * x + b) % p == c % p]
        assert solutions == [int(question.answer)], (seed, question.text)


def test_medium_fraction_expression():
    for seed in range(80):
        question = sup.q(TOPIC, "medium", seed)
        p = _p(question.text)
        a, b, c, d, e = (int(v) for v in re.search(r"(\d+)/(\d+) \+ (\d+)/(\d+) - (\d+)", question.text).groups())
        value = (a * _inverse(b, p) + c * _inverse(d, p) - e) % p
        # the value really is a/b + c/d - e: multiplying back by b*d recovers the numerator
        assert ((value + e) * b * d - (a * d + c * b)) % p == 0
        assert int(question.answer) == value, (seed, question.text)


def test_hard_smaller_root():
    for seed in range(80):
        question = sup.q(TOPIC, "hard", seed)
        p = _p(question.text)
        a, b, c = _quadratic(re.search(r"(\d+x² \+ \d*x \+ \d+)", question.text).group(1))
        roots = _roots(a, b, c, p)
        assert len(roots) == 2 and a > 1
        assert int(question.answer) == min(roots), (seed, question.text)


def test_keys_and_distractors():
    sup.check_keys_and_distractors(SKILL_ID, TOPIC, "number")


def test_text_and_determinism():
    sup.check_text_and_determinism(TOPIC)


def test_answer_space():
    sup.check_answer_space(TOPIC, {"easy": 30, "medium": 30, "hard": 25})


# -- exam -------------------------------------------------------------------------------------------------------


def test_exam_shape_and_marks():
    sup.check_exam_shape(SKILL_ID, TOPIC, "abcd", (2, 1, 1, 3), ("number", "number", "number", "set"), (0, 0, 1, 0))


def test_exam_keys_are_recomputed_from_the_text():
    for seed in range(80):
        question = sup.exam(TOPIC, seed)
        p = _p(question.stem)
        a, b, c = _quadratic(re.search(r"f\(x\) = (\d+x² \+ \d*x \+ \d+)", question.stem).group(1))
        pa, pb, pc, pd = question.parts
        u = _inverse(a, p)
        assert int(pa.answer) == u
        assert int(pb.answer) == (b * b - 4 * a * c) % p
        assert a * int(pa.answer) + p * int(pc.answer) == 1
        assert pd.answer == _roots(a, b, c, p), (seed, question.stem)
        assert sorted(pd.answer) == sorted(set(pd.answer)) and len(pd.answer) == 2


def test_exam_follow_through_for_the_bezout_cofactor():
    s = sup.skill(SKILL_ID)
    for seed in range(40):
        question = sup.exam(TOPIC, seed)
        p = _p(question.stem)
        a = int(re.search(r"f\(x\) = (\d+)x²", question.stem).group(1))
        wrong_u = (int(question.parts[0].answer) + 1) % p
        if wrong_u == 0:
            continue
        v = (1 - a * wrong_u) // p if (1 - a * wrong_u) % p == 0 else None
        if v is None:
            continue
        sent = {part.label: part.answer_text for part in question.parts} | {"a": str(wrong_u), "c": str(v)}
        result = grade(s, question, sent)
        assert result.parts["c"].marks == 1 and result.parts["c"].follow_through is True, (seed, sent)
