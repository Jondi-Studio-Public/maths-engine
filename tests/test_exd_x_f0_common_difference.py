"""Exam source of `common-difference`: the key is re-derived from the rendered prompt with exact fractions."""

from __future__ import annotations

import re
from fractions import Fraction

from mathgen import make
from mathgen.grading import GRADERS
from mathgen.text_quality import find_text_defects, prompt_shape

TOPIC = "exd_x_f0_common_difference"
_NUM = r"(-?\d+)"


def _q(seed):
    return make(TOPIC, seed=seed, difficulty="exam", tier="exam")


def _hard(seed):
    return make("sequence", seed=seed, difficulty=None, kind="common_diff_nonconsecutive")


def _linear(expr: str) -> tuple[int, int]:
    """'-2x + 7' -> (-2, 7)."""
    expr = expr.replace(" ", "")
    m = re.fullmatch(r"(-?\d*)x([+-]\d+)?", expr)
    assert m, expr
    coef = {"": 1, "-": -1}.get(m.group(1), None)
    return (int(m.group(1)) if coef is None else coef), int(m.group(2) or 0)


def _ordinal_value(token: str) -> int:
    return int(re.match(r"\d+", token).group())


def derive(text: str) -> Fraction:
    if text.startswith("The expressions"):
        m = re.match(r"The expressions (.+?), (.+?) and (.+?) are three consecutive", text)
        (c1, k1), (c2, k2), (c3, k3) = (_linear(g) for g in m.groups())
        # 2(c2 x + k2) = (c1 x + k1) + (c3 x + k3)
        x = Fraction(k1 + k3 - 2 * k2, 2 * c2 - c1 - c3)
        return (c2 * x + k2) - (c1 * x + k1)
    if "for its first" in text:
        m = re.match(
            rf"An arithmetic sequence has a sum of {_NUM} for its first (\d+) terms and a sum of {_NUM} for its first (\d+)",
            text,
        )
        sp, p, sq, q = (int(g) for g in m.groups())
        # unknowns a, d:  n a + n(n-1)/2 d = S_n
        det = Fraction(p * q * (q - p), 2)
        return Fraction(p * sq - q * sp) / det
    m = re.match(
        rf"In an arithmetic sequence, the sum of the (\w+) and (\w+) terms is {_NUM} and the (\w+) term is {_NUM}", text
    )
    p, q, total, r, tr = (_ordinal_value(g) if i in (0, 1, 3) else int(g) for i, g in enumerate(m.groups()))
    # a + (r-1) d = tr and 2a + (p+q-2) d = total
    return Fraction(total - 2 * tr, p + q - 2 * r)


def test_exam_adds_shapes_over_hard_tier():
    hard = {prompt_shape(_hard(s)) for s in range(60)}
    exam = {prompt_shape(_q(s)) for s in range(60)}
    assert exam - hard, (exam, hard)


def test_key_matches_independent_derivation_over_300_seeds():
    bad = []
    for seed in range(300):
        q = _q(seed)
        d = derive(q.text)
        if d.denominator != 1 or int(d) != q.answer:
            bad.append((seed, q.text, d, q.answer))
    assert bad == [], bad[:3]


def test_grader_accepts_key_and_rejects_every_distractor():
    grade = GRADERS["number"]
    for seed in range(150):
        q = _q(seed)
        assert grade(q, q.answer_text).correct, (seed, q.text)
        assert len(q.distractors) >= 2, (seed, q.distractors)
        for d in q.distractors:
            assert not grade(q, d).correct, (seed, d, q.answer_text)
