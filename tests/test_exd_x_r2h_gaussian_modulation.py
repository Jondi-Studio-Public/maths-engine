"""Tests for the multi-part exam source `exd_x_r2h_gaussian_modulation` (skill `ucv1-fourier-transform-gaussian`).

The oracle reads the pulse f(x) = A exp(-a x^2) and the carrier frequency out of the rendered text and recomputes the
numeric parts by trapezoid quadrature of the defining integrals; the closed form in part (a) is compared by value.
"""

from __future__ import annotations

import math
import re

import sympy

from mathgen.engine import grade
from tests.exd_x_r2h_support import exam, follow_through_check, skill_of, standard_checks

TOPIC = "exd_x_r2h_gaussian_modulation"
SKILL_ID = "ucv1-fourier-transform-gaussian"
HARD_TOPIC = "p15e_fourier_transform_gaussian"
LABELS = list("abcd")
MARKS = [2, 2, 2, 1]
FT_LABELS = ["b", "c", "d"]


def parse(q):
    amp, a_text = re.search(r"f\(x\) = (-?\d+) exp\(-(\d*)x\^2\)", q.text).groups()
    a = int(a_text) if a_text else 1
    amp = int(amp)
    digits = re.search(r"cos\((\d*)x\)", q.text).group(1)
    return amp, a, int(digits) if digits else 1


def _trapezoid(func, half_width=9.0, steps=3600):
    h = 2 * half_width / steps
    return h * (sum(func(-half_width + i * h) for i in range(1, steps)) + (func(-half_width) + func(half_width)) / 2)


def _same_function(text, amp, a):
    got = sympy.sympify(text.replace("^", "**"), locals={"w": sympy.Symbol("w")})
    want = amp * sympy.sqrt(sympy.pi / a) * sympy.exp(-(sympy.Symbol("w") ** 2) / (4 * a))
    return all(abs(float(got.subs("w", v)) - float(want.subs("w", v))) < 1e-9 for v in (0, 0.7, 2.3))


def oracle(q):
    amp, a, w0 = parse(q)
    integral = _trapezoid(lambda x: amp * math.exp(-a * x * x) * math.cos(w0 * x))
    g_w0 = _trapezoid(lambda x: amp * math.exp(-a * x * x) * math.cos(w0 * x) ** 2)
    got_a = q.parts[0].answer_text
    return {
        "a": got_a if _same_function(got_a, amp, a) else "closed form differs",
        "b": f"{integral:.3f}",
        "c": f"{g_w0:.3f}",
        "d": f"{g_w0 / integral:.3f}",
    }


def test_follow_through_from_wrong_integral():
    def wrong_chain(q):
        wrong_b = float(q.parts[1].distractors[0])
        c = float(q.parts[2].answer_text)
        return (
            "b",
            {
                "a": q.parts[0].answer_text,
                "b": q.parts[1].distractors[0],
                "c": q.parts[2].answer_text,
                "d": f"{c / wrong_b:.3f}",
            },
            5,
        )

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_follow_through_from_wrong_modulated_value():
    def wrong_chain(q):
        b = float(q.parts[1].answer_text)
        wrong_c = float(q.parts[2].distractors[0])
        return (
            "c",
            {
                "a": q.parts[0].answer_text,
                "b": q.parts[1].answer_text,
                "c": q.parts[2].distractors[0],
                "d": f"{wrong_c / b:.3f}",
            },
            5,
        )

    follow_through_check(TOPIC, SKILL_ID, wrong_chain)


def test_all_wrong_earns_nothing():
    skill = skill_of(SKILL_ID)
    for s in range(30):
        q = exam(TOPIC, s)
        result = grade(skill, q, {"a": "7*w", "b": "1009.123", "c": "-307.456", "d": "401.789"})
        assert result.marks == 0


CHECKS = standard_checks(TOPIC, SKILL_ID, oracle, LABELS, MARKS, FT_LABELS, HARD_TOPIC)


def test_wired():
    CHECKS["wired"]()


def test_shape():
    CHECKS["shape"]()


def test_prompt_shapes():
    CHECKS["shapes"]()


def test_oracle_agrees_over_300_seeds():
    CHECKS["oracle_agrees"]()


def test_composed_answer_full_marks():
    CHECKS["composed_full_marks"]()


def test_distractors_tagged_and_rejected():
    CHECKS["distractors_rejected"]()
