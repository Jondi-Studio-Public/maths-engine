"""Tests for the exam source of senior-describe-the-association-between-two-numerical-variables-in.

r, r squared and the three-field description are re-derived from the sums and the residual-plot sentence in the stem.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import mathgen.generators.exd_x_r2k_association  # noqa: F401  (registers the topic)
from tests import exd_x_r2k_kit as kit

SKILL = "senior-describe-the-association-between-two-numerical-variables-in"
TOPIC = "exd_x_r2k_association"
_N = re.compile(r" for (\d+) ")
_SUMS = re.compile(r"Σx = (\d+), Σy = (\d+), Σx² = (\d+), Σy² = (\d+) and Σxy = (\d+)")


def oracle(q) -> dict[str, str]:
    n = int(_N.search(q.stem).group(1))
    sx, sy, sxx, syy, sxy = (int(v) for v in _SUMS.search(q.stem).groups())
    sp = Fraction(sxy) - Fraction(sx * sy, n)
    r = float(sp) / math.sqrt(float((sxx - Fraction(sx * sx, n)) * (syy - Fraction(sy * sy, n))))
    r2 = kit.half_up(r, 2)
    pct = kit.fixed(100 * r2 * r2, 1)
    direction = "positive" if r2 > 0 else "negative"
    if "clear curved pattern" in q.stem:
        form = "non-linear"
        if "almost exactly on the curve" in q.stem.split("Relative to the best-fitting curve")[1]:
            strength = "strong"
        elif "fairly close" in q.stem.split("Relative to the best-fitting curve")[1]:
            strength = "moderate"
        else:
            strength = "weak"
    else:
        form = "linear"
        strength = "strong" if abs(r2) >= Fraction(3, 4) else "moderate" if abs(r2) >= Fraction(1, 2) else "weak"
    return {"a": kit.fixed(r, 2), "b": pct, "c": f"{direction}, {form}, {strength}"}


def test_wiring_and_shapes():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_oracle_agrees_over_300_seeds():
    bad = [s for s in range(300) if oracle(kit.exam_q(SKILL, s)) != kit.canon(kit.exam_q(SKILL, s))]
    assert bad == []


def test_linear_and_curved_both_occur_with_every_strength():
    seen = {tuple(kit.canon(kit.exam_q(SKILL, s))["c"].split(", ")[1:]) for s in range(200)}
    assert {("linear", "strong"), ("linear", "moderate"), ("linear", "weak")} <= seen
    assert any(form == "non-linear" for form, _ in seen)


def test_grading_and_tags():
    kit.check_grading(SKILL)
    kit.check_tags(SKILL)


def test_answer_space():
    kit.check_answer_space(SKILL)


def test_follow_through_from_a_wrong_r():
    def chain(q):
        c = kit.canon(q)
        r_wrong = Fraction(c["a"]) + (Fraction(1, 10) if Fraction(c["a"]) < Fraction(8, 10) else -Fraction(1, 10))
        return "a", {**c, "a": kit.fixed(r_wrong, 2), "b": kit.fixed(100 * r_wrong * r_wrong, 1)}

    kit.check_follow_through(SKILL, chain)
