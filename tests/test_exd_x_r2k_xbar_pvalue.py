"""Tests for the exam source of senior-simulate-repeated-random-sampling-from-a-variety-of-acmsm139.

The standard error, z and the two-sided tail probability are re-derived from the sample summaries in the stem.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import mathgen.generators.exd_x_r2k_xbar_pvalue  # noqa: F401  (registers the topic)
from tests import exd_x_r2k_kit as kit

SKILL = "senior-simulate-repeated-random-sampling-from-a-variety-of-acmsm139"
TOPIC = "exd_x_r2k_xbar_pvalue"


def parse(q):
    n = int(re.search(r"sample of (\d+) ", q.stem).group(1))
    xbar = Fraction(re.search(r"x̄ = (\d+\.\d\d)", q.stem).group(1))
    s = Fraction(re.search(r"s = (\d+\.\d\d)", q.stem).group(1))
    mu = Fraction(re.search(r"μ = (\d+)", q.stem).group(1))
    return n, xbar, s, mu


def tail(z: Fraction) -> float:
    return math.erfc(abs(float(z)) / math.sqrt(2))


def oracle(q) -> dict[str, str]:
    n, xbar, s, mu = parse(q)
    se = kit.half_up(float(s) / math.sqrt(n), 3)
    z = kit.half_up((xbar - mu) / se, 2)
    return {"a": kit.fixed(se, 3), "b": kit.fixed(z, 2), "c": kit.fixed(tail(z), 4)}


def test_wiring_and_shapes():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_oracle_agrees_over_300_seeds():
    bad = [s for s in range(300) if oracle(kit.exam_q(SKILL, s)) != kit.canon(kit.exam_q(SKILL, s))]
    assert bad == []


def test_z_is_stable_whichever_route_is_taken():
    for seed in range(300):
        q = kit.exam_q(SKILL, seed)
        n, xbar, s, mu = parse(q)
        exact_z = float(xbar - mu) / (float(s) / math.sqrt(n))
        assert abs(exact_z - float(kit.canon(q)["b"])) < 0.0101, seed


def test_both_signs_occur():
    zs = [Fraction(kit.canon(kit.exam_q(SKILL, s))["b"]) for s in range(100)]
    assert any(z > 0 for z in zs) and any(z < 0 for z in zs)


def test_grading_and_tags():
    kit.check_grading(SKILL)
    kit.check_tags(SKILL)


def test_answer_space():
    kit.check_answer_space(SKILL)


def test_follow_through_from_a_wrong_standard_error():
    def chain(q):
        n, xbar, s, mu = parse(q)
        c = kit.canon(q)
        se_wrong = kit.half_up(Fraction(c["a"]) * Fraction(6, 5), 3)
        z = kit.half_up((xbar - mu) / se_wrong, 2)
        return "a", {**c, "a": kit.fixed(se_wrong, 3), "b": kit.fixed(z, 2), "c": kit.fixed(tail(z), 4)}

    kit.check_follow_through(SKILL, chain)
