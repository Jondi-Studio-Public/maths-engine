"""Tests for the exam source of senior-identify-relative-frequency-as-probability.

The pooled relative frequency, the expected count and the at-least-one probability are re-derived from the counts in
the rendered stem and the numbers in the part prompts.
"""

from __future__ import annotations

import re
from fractions import Fraction

import mathgen.generators.exd_x_r2k_relfreq  # noqa: F401  (registers the topic)
from tests import exd_x_r2k_kit as kit

SKILL = "senior-identify-relative-frequency-as-probability"
TOPIC = "exd_x_r2k_relfreq"
# Each context states count then total in a different order; the four stems are matched one by one.
_PATTERNS = (
    re.compile(r"first stage, (\d+) of (\d+) items were faulty. In the second stage, (\d+) of (\d+) items"),
    re.compile(
        r"took (\d+) shots at goal in the first half of a season and scored (\d+)\. In the second half she took (\d+) shots and scored (\d+)"
    ),
    re.compile(
        r"timed (\d+) trips in Term 1, and (\d+) arrived late. In Term 2 it timed (\d+) trips, and (\d+) arrived"
    ),
    re.compile(
        r"planted (\d+) seeds in a first tray, and (\d+) germinated. It then planted (\d+) seeds in a second tray, and (\d+) germinated"
    ),
)


def counts(stem: str) -> tuple[int, int]:
    """(total successes, total trials)."""
    m = _PATTERNS[0].search(stem)
    if m:
        c1, n1, c2, n2 = (int(v) for v in m.groups())
    else:
        for pat in _PATTERNS[1:]:
            m = pat.search(stem)
            if m:
                n1, c1, n2, c2 = (int(v) for v in m.groups())
                break
        else:
            raise AssertionError(stem)
    return c1 + c2, n1 + n2


def oracle(q) -> dict[str, str]:
    successes, trials = counts(q.stem)
    a = kit.half_up(Fraction(successes, trials), 3)
    big_m = int(re.search(r"of the next (\d+) ", q.parts[1].prompt).group(1))
    k = int(re.search(r"at least one of the next (\d+) ", q.parts[2].prompt).group(1))
    return {
        "a": kit.fixed(a, 3),
        "b": kit.fixed(big_m * a, 0),
        "c": kit.fixed(1 - (1 - a) ** k, 3),
    }


def test_wiring_and_shapes():
    kit.check_wiring(SKILL, TOPIC)
    kit.check_shapes(SKILL)
    kit.check_no_defects(SKILL)
    kit.check_determinism(SKILL)


def test_oracle_agrees_over_300_seeds():
    bad = [s for s in range(300) if oracle(kit.exam_q(SKILL, s)) != kit.canon(kit.exam_q(SKILL, s))]
    assert bad == []


def test_pooled_value_is_not_the_average_of_the_stage_rates():
    for seed in range(100):
        q = kit.exam_q(SKILL, seed)
        assert q.parts[0].answer_text not in q.parts[0].distractors


def test_grading_and_tags():
    kit.check_grading(SKILL)
    kit.check_tags(SKILL)


def test_follow_through_from_a_wrong_pooled_value():
    def chain(q):
        c = kit.canon(q)
        a_wrong = Fraction(c["a"]) + Fraction(3, 100)
        big_m = int(re.search(r"of the next (\d+) ", q.parts[1].prompt).group(1))
        k = int(re.search(r"at least one of the next (\d+) ", q.parts[2].prompt).group(1))
        return "a", {
            **c,
            "a": kit.fixed(a_wrong, 3),
            "b": kit.fixed(big_m * a_wrong, 0),
            "c": kit.fixed(1 - (1 - a_wrong) ** k, 3),
        }

    kit.check_follow_through(SKILL, chain)
