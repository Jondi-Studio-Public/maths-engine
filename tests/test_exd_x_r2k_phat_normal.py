"""Tests for the exam source of senior-understand-the-concept-of-the-sample-proportion-widehat.

The standard deviation of the sample proportion, the z-score of the observed sample and the tail probability are
re-derived from the population percentage, sample size and count in the stem.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import mathgen.generators.exd_x_r2k_phat_normal  # noqa: F401  (registers the topic)
from tests import exd_x_r2k_kit as kit

SKILL = "senior-understand-the-concept-of-the-sample-proportion-widehat"
TOPIC = "exd_x_r2k_phat_normal"


def parse(q):
    pc = int(re.search(r"(\d+)% of", q.stem).group(1))
    n, x = (int(v) for v in re.search(r"sample of (\d+) [a-z]+ is taken, and (\d+) of them", q.stem).groups())
    return Fraction(pc, 100), n, x


def one_tail(z: Fraction) -> float:
    return 0.5 * math.erfc(abs(float(z)) / math.sqrt(2))


def oracle(q) -> dict[str, str]:
    p, n, x = parse(q)
    sd = kit.half_up(math.sqrt(float(p * (1 - p)) / n), 4)
    z = kit.half_up((Fraction(x, n) - p) / sd, 2)
    return {"a": kit.fixed(sd, 4), "b": kit.fixed(z, 2), "c": kit.fixed(one_tail(z), 4)}


def test_wiring_and_shapes():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_oracle_agrees_over_300_seeds():
    bad = [s for s in range(300) if oracle(kit.exam_q(SKILL, s)) != kit.canon(kit.exam_q(SKILL, s))]
    assert bad == []


def test_both_directions_occur():
    prompts = [kit.exam_q(SKILL, s).parts[2].prompt for s in range(80)]
    assert any("at least" in t for t in prompts) and any("at most" in t for t in prompts)


def test_grading_and_tags():
    kit.check_grading(SKILL)
    kit.check_tags(SKILL)


def test_follow_through_from_a_wrong_standard_deviation():
    def chain(q):
        p, n, x = parse(q)
        c = kit.canon(q)
        sd_wrong = kit.half_up(Fraction(c["a"]) * Fraction(6, 5), 4)
        z = kit.half_up((Fraction(x, n) - p) / sd_wrong, 2)
        return "a", {**c, "a": kit.fixed(sd_wrong, 4), "b": kit.fixed(z, 2), "c": kit.fixed(one_tail(z), 4)}

    kit.check_follow_through(SKILL, chain)
