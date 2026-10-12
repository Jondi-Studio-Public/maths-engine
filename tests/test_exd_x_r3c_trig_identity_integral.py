"""Tests the multi-part exam source of `integrate-trig`: half-angle form, its oscillating part integrated, the whole integral.

The oracle reads f(x) and the upper limit from the rendered stem and works every part out by numerical integration
(mpmath), which uses neither the half-angle identity nor the generator's code.
"""

from __future__ import annotations

import math
import re
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import mpmath
from tests.exd_x_r3c_support import (
    check_all_wrong,
    check_answer_space,
    check_deterministic,
    check_follow_through,
    check_full_marks_and_distractors,
    check_oracle,
    check_shapes,
    check_text,
    check_wired,
    exam_questions,
    skill_for,
)

TOPIC = "exd_x_r3c_trig_identity_integral"
SKILL_ID = "integrate-trig"

_F = re.compile(r"f\(x\) = (\d*)(sin|cos)²\((\d*)x\)")
_LIMIT = re.compile(r"x = (\d*)π(?:/(\d+))?\.")


def parse(q):
    c, name, k = _F.search(q.stem).groups()
    num, den = _LIMIT.search(q.parts[1].prompt).groups()
    upper = Fraction(int(num or 1), int(den or 1))  # in multiples of pi
    return int(c or 1), name, int(k or 1), upper


def _fmt_frac(x):
    return str(Fraction(float(x)).limit_denominator(1000))


def oracle(q):
    c, name, k, upper = parse(q)
    fn = mpmath.sin if name == "sin" else mpmath.cos
    f = lambda x: c * fn(k * x) ** 2  # noqa: E731
    pi = mpmath.pi
    # f(x) = p + q cos(2kx) over one period pi/k: q is twice the mean of f(x) cos(2kx)
    q_val = 2 * mpmath.quad(lambda x: f(x) * mpmath.cos(2 * k * x), [0, pi / k]) / (pi / k)
    qf = Fraction(round(float(q_val) * 2), 2)
    b = mpmath.quad(lambda x: float(qf) * mpmath.cos(2 * k * x), [0, upper.numerator * pi / upper.denominator])
    total = mpmath.quad(f, [0, upper.numerator * pi / upper.denominator])
    c_text = str(Decimal(str(mpmath.nstr(total, 15))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
    return {"a": str(qf), "b": _fmt_frac(b), "c": c_text}


SKILL = skill_for(SKILL_ID)


def test_source_wired():
    check_wired(SKILL, TOPIC)


def test_shapes():
    check_shapes(SKILL, TOPIC)


def test_text_quality():
    check_text(TOPIC)


def test_key_rederived_from_stem():
    check_oracle(TOPIC, oracle, n=300)


def test_full_marks_and_distractors_rejected():
    check_full_marks_and_distractors(SKILL, TOPIC)


def test_all_wrong_earns_nothing():
    check_all_wrong(SKILL, TOPIC)


def test_follow_through_from_a_wrong_q():
    def chain(q, wrong):
        c, name, k, upper = parse(q)
        sigma = round(math.sin(2 * k * float(upper) * math.pi))
        b = Fraction(wrong) * sigma / (2 * k)
        total = Decimal(c) * Decimal(float(upper) * math.pi) / 2 + Decimal(float(b))
        return {"b": str(b), "c": str(total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))}

    check_follow_through(SKILL, TOPIC, chain, n=30)


def test_answer_space():
    check_answer_space(TOPIC)


def test_parts_shape():
    for q in exam_questions(TOPIC, 30):
        assert [p.label for p in q.parts] == ["a", "b", "c"]
        assert [p.marks for p in q.parts] == [2, 2, 2]
        assert [p.answer_kind for p in q.parts] == ["fraction", "fraction", "number"]
        assert [p.follow_through is not None for p in q.parts] == [False, True, True]


def test_deterministic():
    check_deterministic(TOPIC)
